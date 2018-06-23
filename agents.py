from __future__ import division
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from utils import excluding_keys, normalized, softmax
from copy import copy, deepcopy
from utils import HashableDict, _issubclass
from itertools import chain, product, combinations
import networkx as nx


PRETTY_KEYS = {"RA_prior": "prior",
               "RA_K": 'K'}

def add_tremble(p, tremble):
    if tremble == 0:
        return p
    else:
        return (1 - tremble) * p + tremble * np.ones(len(p)) / len(p)


class HashableSet(set):
    def __hash__(self):
        return hash(tuple(self))

def is_agent_type(instance, base):
    try:
        return _issubclass(instance, base)
    except TypeError:
        return _issubclass(instance.type, base)

class AgentType(type):
    def __str__(cls):
        return cls.__name__

    def __repr__(cls):
        return str(cls)

    def __hash__(cls):
        return hash(cls.__name__)

    def __eq__(cls, other):
        return str(cls) == str(other)

    def short_name(cls, *args):
        return cls.__name__

    # def __reduce__(self):
    #    return (eval,(self.__name__,))


class Agent(object):
    __metaclass__ = AgentType

    def __new__(cls, genome=None, world_id=None, **kwargs):
        if not genome and kwargs:
            return PrefabAgent(cls, **kwargs)
        else:
            return object.__new__(cls)

    def __init__(self, genome, world_id=None):
        self.genome = genome
        self.beta = self.genome['beta']
        self.world_id = world_id
        self.belief = dict()
        self.likelihood = dict()

    def utility(self, payoffs, agent_ids):
        raise NotImplementedError

    def decide_likelihood(self, game, agents, tremble=0, **kwargs):
        # The first agent is always the deciding agent
        # Only have one action so just pick it
        if len(game.actions) == 1:
            # Returning a probability a vector
            return np.array([1])

        Us = np.array([self.utility(game.payoffs[action], agents)
                       for action in game.actions])

        return add_tremble(softmax(Us, self.beta), tremble)

    def likelihood_of(self, game, participant_ids, tremble = 0, action = None, **kwargs):
        likelihoods = self.decide_likelihood(game, participant_ids, tremble)
        if action == None:
            return likelihoods
        else:
            return likelihoods[game.actions.index(action)]

    def decide(self, game, agent_ids):
        # Tremble is always 0 for decisions since tremble happens in
        # the world, not the agent
        ps = self.decide_likelihood(game, agent_ids, tremble=0)
        action_id = np.squeeze(np.where(np.random.multinomial(1, ps)))
        return game.actions[action_id]

    # For uniformity with Rational Agents
    def observe(self, observations):
        pass

    def observe_k(self, observations, k, tremble=0):
        pass

    # def short_name(self, *args):
        # return self.__name__

class PrefabAgent(Agent):
    def __init__(self, a_type, **genome_kwargs):
        self.type = a_type


        try:
            self._nickname = genome_kwargs['subtype_name']
        except:
            pass

        self.__name__ = self.indentity = str(a_type) + "(%s)" % ",".join(
            ["%s=%s" % (key, val) for key, val in sorted(genome_kwargs.iteritems())])

        self.genome = HashableDict(genome_kwargs)

    def __call__(self, genome, world_id=None):
        temp_genome = copy(self.genome)
        try:
            tom = temp_genome['agent_types']
            temp_genome['agent_types'] = tuple(t if t != 'self' else self for t in tom)

        except:
            pass

        return self.type(dict(genome, **temp_genome), world_id=world_id)

    def short_name(self, *without):
        try:
            return self._nickname
        except:
            genome = excluding_keys(self.genome, *without)
            return str(self.type) + "(%s)" % ",".join(["%s=%s" % (PRETTY_KEYS.get(key, key), val) for key, val in sorted(genome.iteritems())])

    def __str__(self):
        try:
            return self._nickname
        except:
            return self.__name__

    def __repr__(self):
        # return self.short_name('agent_types')
        return self.short_name()#str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if str(self) == str(other):
            return True
        if str(self.type) == str(other):
            return True
        return False

        # if hash(self) == hash(other):
        #     return True
        # if hash(self.type) == hash(other):
        #     return True
        # return False

    def ingroup(self):
        return self.type.ingroup()

class Puppet(Agent):
    def __init__(self, world_id='puppet'):
        self.world_id = world_id
        self.belief = self.likelihood = None

    def decide(self, decision, agent_ids):
        print decision.name
        for i, (action, payoff) in enumerate(decision.payoffs.iteritems()):
            print i, action, payoff
        choice = decision.actions[int(input("enter a number: "))]
        print ""
        return choice

class SelfishAgent(Agent):
    def __init__(self, genome, world_id=None):
        super(SelfishAgent, self).__init__(genome, world_id)

    def utility(self, payoffs, agent_ids):

        weights = [1 if agent_id == self.world_id else 0
                   for agent_id in agent_ids]

        return np.dot(weights, payoffs)

class AltruisticAgent(Agent):
    def __init__(self, genome, world_id=None):
        super(AltruisticAgent, self).__init__(genome, world_id)

    def utility(self, payoffs, agent_ids):
        weights = [1] * len(agent_ids)
        return np.dot(weights, payoffs)

class ConstantDefaultDict(dict):
    def __init__(self, val):
        self.val = val

    def __missing__(self, key):
        ret = self[key] = copy(self.val)
        return ret
    
class TypeDict(dict):
    def __init__(self, genome, agent_id=None):
        agent_types = genome['agent_types']
        rational_types = filter(lambda t: _issubclass(
            t, RationalAgent), agent_types)
        model = self.agent = RationalAgent(genome, agent_id)
        belief = self.belief = model.belief
        likelihood = self.likelihood = model.likelihood
        self.observers = observers = [model]
        for agent_type in agent_types:
            m = agent_type(genome, agent_id)
            if agent_type in [gTFT, Pavlov]:
                observers.append(m)
            dict.__setitem__(self, agent_type, m)

        for rational_type in rational_types:
            r_model = dict.__getitem__(self, rational_type)
            r_model.belief = belief
            r_model.likelihood = likelihood

    def observe_k(self, observations, k, tremble):
        for observer in self.observers:
            observer.observe_k(observations, k, tremble)


class AgentDict(dict):
    def __init__(self, genome):
        self.genome = genome

    def __missing__(self, agent_id):
        ret = self[agent_id] = TypeDict(self.genome, agent_id)
        return ret


class RationalAgent(Agent):
    def __init__(self, genome, world_id=None):  # , *args, **kwargs):
        super(RationalAgent, self).__init__(genome, world_id)
        self._type_to_index = dict(map(reversed, enumerate(genome['agent_types'])))
        self.pop_prior = copy(self.genome['prior'])
        self.model = AgentDict(genome)

        self.likelihood = ConstantDefaultDict(self.initialize_likelihood())
        self.belief = ConstantDefaultDict(self.initialize_prior())

    def belief_that(self, a_id, a_type):
        if a_type in self._type_to_index:
            return self.belief[a_id][self._type_to_index[a_type]]
        else: 
            return 0

    def likelihood_that(self, a_id, a_type):
        return self.likelihood[a_id][self._type_to_index[a_type]]

    def k_belief(self, a_ids, a_type):
        """
        Recursive function. 
        call with list of agent_ids. If you want A's belief that B believes A is reciprocal
        agent_A.k_belief(('B','A'),ReciprocalAgent)
        """
        if len(a_ids) == 1:
            return self.belief_that(a_ids[0], a_type)
        else:
            a_id = a_ids[0]
            a_ids = a_ids[1:]
            return self.model[a_id].agent.k_belief(a_ids, a_type)

    def initialize_prior(self):
        # This needs to initialize the data structure that is used for the
        # online update
        return self.pop_prior

    def initialize_likelihood(self):
        # return normalized(np.zeros_like(self.pop_prior)+1)
        return np.zeros_like(self.pop_prior)
    
    def utility(self, payoffs, agent_ids):
        weights = map(self.sample_alpha, agent_ids)
        return np.dot(weights, payoffs)

    def sample_alpha(self, agent_id):
        """
        this function basically tells us how much we care about
        a particular agent's payoff as a function of our beliefs about them
        every reciprocal agent type is defined by just defining this function
        """

        # return int(flip(belief))
        print "Rational Agents don't know how to choose, subclasses do"
        raise NotImplementedError

    def observe(self, observations):
        if self.genome['RA_prior'] in [1, 0]:
            return
        self.observe_k(observations, self.genome['RA_K'], self.genome['tremble'])

    def observe_k(self, observations, K, tremble=0):
        """
        takes in
        observations = [(game, agent_ids, observer_ids, action), ...]
        k = an integer. (function has special behavior for k =,<,> 0)

        """
        # Key assumption: everyone who observes the action, observes
        # who observes the action. Thus observation of action is
        # common-knowledge among those who observe the action. First
        # order beliefs: I believe that you are X. Second order
        # beliefs: I believe that you believe that I am X OR I believe
        # that you believe that she is Y. These are needed to learn
        # from observation. Third order beliefs: I believe that you
        # believe that she believes he is Z. My guess is that third
        # order beliefs only diverge from second order beliefs if who
        # observes who is unknown.

        # if K < 0: return
        genome = self.genome
        agent_types = genome['agent_types']
        rational_types = filter(lambda t: _issubclass(t, RationalAgent), agent_types)

        observations = filter(lambda obs: self.world_id in obs[2], observations)
        for observation in observations:
            observers = observation[2]

            for agent_id in observers:
                if agent_id == self.world_id:
                    continue

                if agent_id not in self.model:
                    model = self.model[agent_id]
                    for o_id in observers:
                        model.belief[o_id]
                        model.likelihood[o_id]

        for observation in observations:
            game, participants, observers, action = observation

            decider_id = participants[0]

            if decider_id == self.world_id:
                continue

            likelihood = []
            action_index = game.action_lookup[action]

            # calculate the normalized likelihood for each type
            for agent_type in agent_types:
                model = self.model[decider_id][agent_type]
                likelihood.append(model.decide_likelihood(game, participants, tremble)[action_index])

            # # Not using log-likelihoods
            # self.likelihood[decider_id] *= likelihood
            # self.likelihood[decider_id] = normalized(self.likelihood[decider_id])
            # prior = self.pop_prior
            # likelihood = self.likelihood[decider_id]
            # self.belief[decider_id] = prior*likelihood/np.dot(prior,likelihood)
            self.likelihood[decider_id] += np.log(likelihood)
            prior = np.log(self.pop_prior)
            likelihood = self.likelihood[decider_id]

            self.belief[decider_id] = np.exp(prior + likelihood)
            self.belief[decider_id] = normalized(self.belief[decider_id])

        # Observe the other person, when this code runs at K=0
        # nothing will happen because of the return at the top
        # of the function.

        if K == 0:
            return
        for agent_id, models in self.model.iteritems():
            models.observe_k(observations, K - 1, tremble)

        # if K>0:
    #     self.update_prior()

class IngroupAgent(RationalAgent):
    def __init__(self, genome, world_id=None):
        super(IngroupAgent, self).__init__(genome, world_id)
        my_ingroup = self.ingroup()

        # The indices of the agent_types who are in my in-group
        self.ingroup_indices = list()
        for agent_type in genome['agent_types']:
            if agent_type in my_ingroup:
                self.ingroup_indices.append(self._type_to_index[agent_type])
        self.ingroup_indices = np.array(self.ingroup_indices)

    def sample_alpha(self, agent_id):
        # If its me
        if agent_id == self.world_id:
            return 1

        try:
            return sum(self.belief[agent_id][self.ingroup_indices])
        except Exception as e:
            print 'Alejandro promised this wouldn\'t happen. Look at the commented code below for a fix'
            print e
            raise e
        
        # try:
        #     return sum(self.belief[agent_id][self.ingroup_indices])
        # except IndexError:
        #     return sum([self.belief_that(agent_id, t) for t in self.genome['agent_types'] if self.is_in_ingroup(t)])
        # except KeyError:
        #     self.belief[agent_id] = self.initialize_prior()
        #     return sum(self.belief[agent_id][self.ingroup_indices])

    def is_in_ingroup(self, a_type):
        for i in self.ingroup():
            
            # If its the same in-group exactly
            if a_type == i:
                return True

            # Or if its a subclass of the ingroup
            if _issubclass(a_type, i):
                return True
            
        return False

class ReciprocalAgent(IngroupAgent):
    @staticmethod
    def ingroup():
        return [ReciprocalAgent]



### feasible
# kappa | 0 <= kappa <= B-C
# chi | -1 <= max((kap-B)/(kap+C),(kap+C)/(kap-B)) <= chi <= 1
# phi | 0 < phi <= chi*B/(chi*B/chi*C+B)
# lamda |
# (chi+1)/C + (B-C) <= lambda <= (chi+1)/(-B) + (B-C)
# -(chi*B+C)<= lambda <= (B+chi*C)

### robust
# X is robust against mutant Y
# if the odds off Y replacing X are <1/N
# for N->inf robustness reduces to ESS

### extortion
# kap = 0 = P
# chi > 0

### cooperative
# kappa = R = B-C

### generous
# cooperative
# chi > 0

### good
# cooperative
#
#(lambda > -(B-C)*chi AND lambda > -(B+C)*chi)
# OR
#lambda | -(chi*B+C) <= lambda <= (B+chi*C)

### Good+Robust (in pop = N)
# feasible
# cooperative (implied)
# lambda |
# lambda >(B-C)/(3N)* (N+1-(2n-1)*chi)
# AND
# lambda >(B+C)/(N-2)*(N+1-(2n-1)*chi)

### robust ZD for N>2
# cooperative
# generous (implied)
# 1 > chi >= (N+1)/(2N-1)

### ZD
# lambda = 0

### WSLS at least has
# chi = -C/B < 0


class HyperAgent(Agent):
    ####ONLY WORKS FOR THE DEFAULT CASE
    def __init__(self, genome, world_id=None):
        defaults = dict(chi=2/3, kap = None, lam = 0, phi = 3/11)
        keys = ["B","C","chi","kap","lam",'phi']
        B,C,chi,kap,lam,phi = [genome[k] if k in genome else defaults[k] for k in keys]
        self.world_id = world_id

        if kap == None:
            # make strategies cooperative
            kap = B-C

        assert 0 <= kap and kap <= B-C
        assert max((kap-B)/(kap+C),(kap+C)/(kap-B)) <= chi and chi <= 1

        if phi == None:
            #max out phi, at bottom it's TFT
            #phi = (chi*B)/(chi*C+B)
            phi = C/(C+chi*B)

        #print phi
        #assert (0 < phi) and (phi <= (chi*B)/(chi*C+B))
        #assert (0 < phi) and (phi <= C/(chi*B+C))
        #assert (lam > -(B-C)*chi) and (lam > -(B+C)*chi)
        assert (lam >= -B*(chi+1)+B-C) and (lam <= C*(chi+1)+B-C)
        p_vec= (1-phi*(1-chi)*(B-C-kap),
                1-phi*(chi*C+B-kap*(1-chi)+lam),
                phi*(chi*B+C+(1-chi)*kap-lam),
                phi*(1-chi)*kap)
        

        for i,p in enumerate(p_vec):
            if not p<=1 and p>=0:
                print chi,kap,lam,phi
                print B,C
                print i,p
                raise Exception("p out of bounds: %s" % str(p_vec))
        
        joint_actions = [('give','give'),
                         ('give','keep'),
                         ('keep','give'),
                         ('keep','keep')]

        self.reaction = strats = {a:{'give':p,'keep':1-p} for a,p in zip(joint_actions,p_vec)}
        self.memory = defaultdict(lambda:('give','give'))

    def observe(self, observations):
        # Can only judge a case where there are two observations
        # TODO: This will not work on sequential PD since it requires two simultaneous observations
        obs1, obs2 = observations
        actions = (obs1['action'], obs2['action'])
        players = (obs1['participant_ids'][0], obs2['participant_ids'][0])
        me = self.world_id
        if me in players:
            if me == players[1]:
                players = tuple(reversed(players))
                actions = tuple(reversed(actions))
            self.memory[players[1]] = actions


    def decide_likelihood(self, game, agents=None, tremble=None):
        me, other = agents
        assert me == self.world_id
        return add_tremble(np.array([self.reaction[self.memory[other]][action] for action in game.actions]), tremble)

class ClassicAgent(Agent):
    
    def decide(self, game, agent_ids):
        ps = self.decide_likelihood(game, tremble=0)
        action_id = np.squeeze(np.where(np.random.multinomial(1, ps)))
        action = game.actions[action_id]
        return action

    def observe(self, *args, **kwargs):
        pass

    def observe_k(self, observations, *args, **kwargs):
        self.observe(observations)

    def k_belief(self, *args, **kwargs):
        return 0


class Standing(Agent):
    def __init__(self, genome, world_id = None):
        self.genome = genome
        self.world_id = world_id
        self.image = ConstantDefaultDict(True)
        self.action_dict = genome['action_dict']
        self.assesment_dict = genome['assesment_dict']

    def observe(self, observations):
        # ASSUMPTION: You only see every agent act once, in a round. 
        for obs in observations:
            assert self.world_id in set(obs[2])
            decider, recipient = obs[1]
            action = obs[3]
            image = self.image
            assesment = self.assesment_dict
            image[decider] = assesment[(action,image[decider],image[recipient])]

    def decide_likelihood(self, game, agents = None, tremble = 0):
        action_dict = self.action_dict
        image = self.image
        [decider, recipient] = agents
        action = action_dict[(image[decider], image[recipient])]
        return add_tremble(np.array([1 if a == action else 0 for a in game.actions]),tremble)

def make_assesment_dict(assesment_list):
    """refer to order of situations in table p98 calculus of selfishness"""
    situation = product(['give','keep'],[True,False],[True,False])
    return dict(zip(situation,assesment_list))

def make_action_dict(action_list):
    """refer to order of situations in table p98 calculus of selfishness"""
    strategies = product([True,False],repeat = 2)
    return dict(zip(strategies,action_list))

STANDING_SHORTHAND_TRANSLATOR = {
    'g': True,
    'b': False,
    'y': 'give',
    'n': 'keep'
}
def shorthand_to_standing(shorthand):
    translated = [STANDING_SHORTHAND_TRANSLATOR[s] for s in shorthand]
    assesments,actions = translated[:8],translated[8:12]

    assert all([a in [True,False] for a in assesments])
    assert len(assesments) == 8
    
    assert all([a in ['keep','give'] for a in actions])
    assert len(actions) == 4
    standing_type = Standing(assesment_dict = make_assesment_dict(assesments),
                             action_dict = make_action_dict(actions))
    standing_type._nickname = "Standing("+shorthand+")"
    return standing_type

def leading_8_dict():
    #this is the transpose of the table in p 98 of TCoS
    shorthands = [
        'ggggbgbbynyy',
        'gbggbgbbynyy',
        'ggggbgbgynyn',
        'gggbbgbgynyn',
        'gbggbgbgynyn',
        'gbgbbgbgynyn',
        'gggbbgbbynyn',
        'gbgbbgbbynyn'
    ]
    types = map(shorthand_to_standing,shorthands)
    names = ["L"+str(n) for n in range(1,9)]
    for t,n in zip(types,names):
        t._nickname = n
    return dict(zip(names,types))

class Pavlov(ClassicAgent):
    def __init__(self, genome, world_id=None):
        self.genome = genome
        self.world_id = world_id
        self.strats = strats = [
            {'give': 1, 'keep': 0},
            {'keep': 1, 'give': 0}]
        self.strat_index = 0

    def observe(self, observations):
        # Can only judge a case where there are two observations
        # TODO: This will not work on sequential PD since it requires two simultaneous observations
        obs1, obs2 = observations
        actions = [obs1['action'], obs2['action']]
        players = [obs1['participant_ids'][0], obs2['participant_ids'][0]]
        me = self.world_id
        if me in players:
            a = players.index(me)
            o = (a + 1) % 2
            if actions[o] == 'keep':
                self.strat_index = (self.strat_index + 1) % 2

    def decide_likelihood(self, game, agents=None, tremble=None):
        return add_tremble(np.array([self.strats[self.strat_index][action] for action in game.actions]), tremble)

WSLS = Pavlov(subtype_name = "WSLS")
    
class gTFT(ClassicAgent):
    def __init__(self, genome, world_id=None):
        self.world_id = world_id
        self.genome = genome

        self.y = y = genome['y']
        self.p = p = genome['p']
        self.q = q = genome['q']

        self.rules = {None: {"give": y, 'keep': 1 - y},
                      True: {"give": p, 'keep': 1 - p},
                      False: {'give': q, 'keep': 1 - q}}

        self.cooperation = 'give'
        self.cooperated = None

    def decide_likelihood(self, game, agents=None, tremble=None):
        rules = self.rules
        last_action = self.cooperated
        return add_tremble(np.array([rules[last_action][action] for action in game.actions]), tremble)

    def observe_k(self, observations, *args, **kwargs):
        self.observe(observations)

    def observe(self, observations):
        assert len(observations) == 2
        for observation in observations:
            game, participants, observers, action = [observation[k] for k in ['game', 'participant_ids', 'observer_ids', 'action']]

            if self.world_id not in observers:
                continue
            if self.world_id not in participants:
                continue
            decider_id = participants[0]
            if decider_id == self.world_id:
                continue

            self.cooperated = action is "give"

TFT = gTFT(y=1, p=1, q=0, subtype_name = "TFT")
# benefit = 3.; cost = 1.
# q = min(1 - (benefit - (benefit - cost)) / (benefit - cost - -cost), (benefit - cost - 0) / (benefit - 0))
# print q
GTFT = gTFT(y=1, p=1, q=.66, subtype_name = "GTFT")

class AllC(ClassicAgent):
    def decide_likelihood(self, game, agents=None, tremble=None):
        odds = {'give': 1,
                'keep': 0}
        return add_tremble(np.array([odds[action] for action in game.actions]), tremble)


class AllD(ClassicAgent):
    def decide_likelihood(self, game, agents=None, tremble=None):
        odds = {'give': 0,
                'keep': 1}
        return add_tremble(np.array([odds[action] for action in game.actions]), tremble)


class RandomAgent(ClassicAgent):
    def decide_likelihood(self, game, *args, **kwargs):
        l = len(game.actions)
        return (1 / l,) * l


class wTypeDict(dict):
    """test that modeled types are captured correctly"""
    def __init__(self, genome, agent_id=None):
        agent_types = tuple(a_type for a_type in genome['agent_types'] if a_type != WeAgent)
        self.observers = {}#list()  # [model]
        for agent_type in agent_types:
            m = agent_type(genome, agent_id)
            must_observe = False
            for modeled_supertype in [gTFT, Pavlov, Standing]:
                if _issubclass(agent_type, modeled_supertype):
                    must_observe = True

            if must_observe:
                self.observers[agent_type] = m
            self[agent_type] = m
        
    def observe_k(self, observations, k, tremble):
        for observer in self.observers.values():
            observer.observe_k(observations, k, tremble)


class wAgentDict(dict):
    def __init__(self, genome):
        self.genome = genome

    def __missing__(self, agent_id):
        ret = self[agent_id] = wTypeDict(self.genome, agent_id)
        return ret

def power_set(s):
    return chain.from_iterable(combinations(s,r) for r in range(len(s)+1))


class Mimic(RationalAgent):
    def decide_likelihood(self, game, agents, tremble):
        agents[1]
        likelihood = 0
        flipped = list(reversed(agents))
        for agent_type, model in self.model[agent_id].iteritems():
            if agent_type == Mimic:
                model_likelihood
            model.decide_likelihood(game, flipped, 0)
            likelihood += self.belief_that(agent_id, agent_type)

    def utility(self, payoffs, agents):
        pass

class UniverseSet(frozenset):
    def __contains__(self,x):
        return True
    def __and__(self,s):
        return s
    def __rand__(self,s):
        return s
    def __or__(self,s):
        return self
    def __ror__(self,s):
        return self
    def __len__(self):
        # TODO: This needs to be infinity but using approx infinity
        # for now due to some other bug somewhere
        return 10000000
    def __le__(self,other):
        return False
    def __lt__(self,other):
        return False
    def __ge__(self,other):
        return True
    def __gt__(self,other):
        return True
    def issuperset(self,other):
        return True
    def issubset(self,other):
        return False
    def __rsub__(self,other):
        return other-other


class ModelNode(object):
    def __init__(self, genome, id_set):
        self.ids = id_set
        self.beliefs = None
        self.likelihood = None
        
        
        #route to the 'everyone' by default
        self.models = defaultdict(lambda: self)

        self.genome = genome = genome
        self.need_to_observe = True#gTFT in genome['agent_types'] or Pavlov in genome['agent_types']

        self.beta = genome['beta']

        self._type_to_index = dict(map(reversed,enumerate(genome['agent_types'])))
        self._my_type_index = self._type_to_index[genome['type']]
        
        tmp_genome = dict(genome)
        tmp_genome['agent_types'] = tuple(a_type for a_type in tmp_genome['agent_types'] if a_type != genome['type'])
        self.other_models = wAgentDict(tmp_genome)

        #print genome['agent_types']
        prior = genome['prior']
        non_WA_prior = (1-prior)/(len(genome['agent_types'])-1)
        self.pop_prior = prior = np.array([prior if t is genome['type'] else non_WA_prior for t in genome['agent_types']])

        self.belief = ConstantDefaultDict(prior)
        self.likelihood = ConstantDefaultDict(np.zeros_like(prior))
        self.new_likelihoods = ConstantDefaultDict(np.zeros_like(prior))
        #self.new_likelihoods = defaultdict(int)
        
    def copy(self,new_id_set):
        cpy = copy(self)
        cpy.belief = deepcopy(self.belief)
        cpy.likelihood = deepcopy(self.likelihood)
        cpy.models = models = copy(self.models)
        cpy.other_models = deepcopy(self.other_models)
        cpy.ids = new_id_set
        for i in new_id_set:
            models[i] = cpy

        return cpy

    def belief_that(self, a_id, a_type):
        if a_type in self._type_to_index:
            return self.belief[a_id][self._type_to_index[a_type]]
        else:
            return 0

    def utility(self,payoffs, agent_ids):
        t = self._my_type_index
        weights = [1]+[self.models[agent_ids[0]].belief[a][t] for a in agent_ids[1:]]
        return np.dot(payoffs, weights)
    
    def decide_likelihood(self, game, agents, tremble = 0):
        if len(game.actions) == 1:
            return np.array([1])

        Us = np.array([self.utility(game.payoffs[action], agents)
                       for action in game.actions])
        #print tremble
        return add_tremble(softmax(Us, self.beta), tremble)

    def observe(self, observations):
        #print self.ids
        agent_types = self.genome['agent_types']
        self.new_likelihoods = ConstantDefaultDict(np.zeros_like(self.pop_prior))

        for observation in observations:
            game, participants, observers, action = [observation[k] for k in ['game', 'participant_ids', 'observer_ids', 'action']]
            tremble = game.tremble
            if not self.ids <= set(observers): continue
            decider_id = participants[0]
            action_index = game.action_lookup[action]

            likelihood = []
            for agent_type in agent_types:
                if agent_type == self.genome['type']:
                    model = self
                else:
                    model = self.other_models[decider_id][agent_type]
                likelihood.append(model.decide_likelihood(game,participants,tremble)[action_index])

            self.new_likelihoods[decider_id] += np.log(likelihood)
        #self.nl_cache = copy(new_likelihoods)

        prior = np.log(self.pop_prior)
        for decider_id, new_likelihood in self.new_likelihoods.iteritems():
            self.likelihood[decider_id] += new_likelihood
            likelihood = self.likelihood[decider_id]
            #print 'prior', prior
            #print 'likelihood', likelihood
            self.belief[decider_id] = np.exp(prior+likelihood)
            self.belief[decider_id] = normalized(self.belief[decider_id])
        #self.l_cache = deepcopy(self.likelihood)

        if self.need_to_observe:
            for model in self.other_models.itervalues():
                model.observe_k(observations, 0, tremble)


class JoinLatticeModel(object):
    def __init__(self,genome):
        """
        sets is the set of all known sets S
        top is the set that is a subset of all other elements
        bottom is the set that is the superset of all others, i.e. the universe
        subsets maps from a A to all B in S such that B<=A
        supersets maps from A to all C such that C>A
        size_to_sets maps from n to all known sets of size n

        """
        #Top<s for all s in S
        self.top = self.bottom = U = UniverseSet()
        self.sets = set([U])
        #if supersets[A] contains B, A<B
        self.supersets = {U:set()}
        #if subsets[A] contains B, A>=B
        self.subsets = {U:set((U,))}
        self.model = {U:ModelNode(genome,U)}
        self.size_to_sets = defaultdict(set)
        self.size_to_sets[len(U)].add(U)

    def make_path_to(lattice, new_set):
        """
        insert a new set into the lattice
        start with the largest potential subsets
        if the new set is a known superset of one of your supersets
        it is a superset of yours
        """
        size = len(new_set)

        subsets = lattice.subsets
        supersets = lattice.supersets
        model = lattice.model
        supersets = lattice.supersets
        #we only need to check the smaller sets for potential subsets
        smaller_set_sizes = sorted((i for i in lattice.size_to_sets.iterkeys() if i < size), reverse = True)
        for n in smaller_set_sizes:
            for small_set in lattice.size_to_sets[n]:
                if (small_set < new_set) and (new_set not in supersets[small_set]):
                    connection_needed = True
                    for s in supersets[small_set]:
                        if new_set in supersets[s]:
                            connection_needed = False

                    models = model[small_set].models
                    new_model = model[new_set]
                    if connection_needed:
                        for i in new_set - small_set:
                            models[i] = new_model
                    else:
                        for i in new_set-small_set:
                            if new_set < models[i].ids:
                                models[i] = new_model
                    subsets[new_set].add(small_set)
                    supersets[small_set].add(new_set)

    def insert_new_set(self, new_set):
        """
        insert any intersections between this set and any known sets
        every new intersection's associated node is a copy of
        the node of the largest known set whose union with the new set produces the intersection

        make a path to the new sets, starting with the smallest new sets

        return True if any of the new sets is smaller than the smallest known set
        """
        
        
        model = self.model
        size_to_sets = self.size_to_sets
        new_set = frozenset(new_set)
        new_sets = {}
        subsets = self.subsets
        supersets = self.supersets
        #sets must be sorted from largest to smallest
        #this way we make sure that new_sets[i] has the value of the smallest set that makes that intersection
        sets = sorted(self.sets, key = len, reverse = True)

        for s in sets:
            i = new_set&s
            if i not in sets:
                new_sets[i] = s

        #here the new sets are actually inserted
        for subset, superset in new_sets.iteritems():
            size_to_sets[len(subset)].add(subset)
            self.sets.add(subset)
            subsets[superset].add(subset)
            subsets[subset] = copy(subsets[superset])
            subsets[subset].remove(superset)

            supersets[subset] = copy(supersets[superset])
            supersets[subset].add(superset)

            new_model = model[subset] = model[superset].copy(subset)

        new_sets = sorted(new_sets.iterkeys(), key = len)
        new_smallest = new_sets[0]

        new_top = False
        if len(new_smallest)<len(self.top):
            self.top = new_smallest
            new_top = True

        for s in new_sets:
            ###THIS CAN BE OPTIMIZED
            self.make_path_to(s)

        return new_top

    def observe(self, observations):
        """
        check which observers need to be inserted into the lattice and do it
        then have all subsets of observers observe
        """
        sets = self.sets
        insert = self.insert_new_set


        observers = set(frozenset(o['observer_ids']) for o in observations)
        
        
        #this must be a list, because 'any' will short-circuit as it expands an iterator
        new_top = any([insert(s) for s in observers if s not in sets])
        

        #observer_sets = sorted(set().union(self.subsets[o] for o in observers), key = len)
        observer_subsets = sorted(set().union(*[self.subsets[o] for o in observers]), key = len)
        for s in observer_subsets:
            self.model[s].observe(observations)
        return new_top

    def draw_hasse(self,my_id,ids):
        nodes = self.sets
        edges = []
        U = self.bottom

        def set_to_str(s):
            return "".join(sorted(s))
        for a in self.sets:
            print "the set",a
            for e,b in self.model[a].models.iteritems():
                print "\tthe link",e
                print "\tthe ids",b.ids
                if b.ids != U:
                    edges.append((set_to_str(a),set_to_str(b.ids)))
            if len(self.supersets[a]) == 1:
                print "set in question", a
                print 'should be singleton', self.supersets[a]
                edges.append((set_to_str(a),'U'))
        nodes = [set_to_str(s) for s in self.sets if s != U]

        pos = make_pos_dict(my_id,ids)
        nodes+=["U"]
        labels = {i:i for i in nodes}
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        f = plt.figure(figsize = (5,5))
        plt.ylim((0,1))
        plt.xlim((0,1))
        nx.draw(G,pos,labels = labels)
        plt.show()

def make_pos_dict(my_id,all_ids):
    """
    Used by JoinLattice's draw_hasse method
    requires a single id that will be the head, and 'all_ids' which is the list of ids of other players
    """
    my_id = frozenset(my_id)
    ids = frozenset(all_ids)
    #ids = frozenset().union(*all_ids)
    powerset = ["".join(sorted(my_id|set(s))) for s in power_set(set(ids).difference(my_id))]
    size_to_strs = defaultdict(set)
    for s in powerset:
        size_to_strs[len(s)].add(s)

    for n in size_to_strs:
        size_to_strs[n] = sorted(size_to_strs[n])

    y_levels = max(size_to_strs.iterkeys())+1

    pos = {}
    for size, y in zip(sorted(size_to_strs.keys()+["inf"]),reversed(np.linspace(0,1,y_levels+2)[1:-1])):
        if size != "inf":
            strs = size_to_strs[size]
        else:
            strs = ["U"]
        for s, x in zip(strs,np.linspace(0,1,len(strs)+2)[1:-1]):
            if size == "inf":
                print len(strs)
                print np.linspace(0,1,len(strs)+2)[1:-1]
                print s
                print x
                print y
            pos[s] = (x,y)

    return pos

class WeAgent(Agent):
    """The lord of all RAs. Bow before her. She rules over all."""
    def __init__(self, genome, world_id):
        self.world_id = world_id
        self.shared_model = self.lattice = lattice = JoinLatticeModel(genome)
        self.me = me = lattice.model[lattice.top]
        self.belief = me.belief
        self.pop_prior = me.pop_prior
        self.likelihood = me.likelihood
        self.models = me.models
        self.new_likelihoods = me.new_likelihoods

        self._type_to_index = dict(map(reversed, enumerate(genome['agent_types'])))

    def decide_likelihood(self,*args,**kwargs):
        return self.me.decide_likelihood(*args,**kwargs)

    def observe(self,observations):
        """
        calls the lattice.observe function
        if it returns true, meaning the top element has changed,
        then point to the new top element
        """
        observations = list(o for o in observations if self.world_id in o['observer_ids'])
        lattice = self.shared_model
        if lattice.observe(observations):
            self.me = me = lattice.model[lattice.top]
            self.belief = me.belief
            self.likelihood = me.likelihood
            self.models = me.models
            self.new_likelihoods = me.new_likelihoods
            #self.l_cache = me.l_cache
            #self.nl_cache = me.nl_cache

    def belief_that(self, a_id, a_type):
        if a_type in self._type_to_index:
            return self.belief[a_id][self._type_to_index[a_type]]
        else:
            return 0

class FullLattice(dict):
    def __init__(self, agent_set, common_knowledge):
        agent_ids = common_knowledge['agent_ids']
        for subset in subsets_with(agent_ids, agent_set):
            lattice[subset] = model_node(subset, common_knowledge)

class UniverseSet(frozenset):
    def __contains__(self,x):
        return True
    def __and__(self,s):
        return s
    def __rand__(self,s):
        return s
    def __or__(self,s):
        return self
    def __ror__(self,s):
        return self
    def __len__(self):
        # TODO: This needs to be infinity but using approx infinity
        # for now due to some other bug somewhere
        return 10000000
    def __le__(self,other):
        return False
    def __lt__(self,other):
        return False
    def __ge__(self,other):
        return True
    def __gt__(self,other):
        return True
    def issuperset(self,other):
        return True
    def issubset(self,other):
        return False
    def __rsub__(self,other):
        return other-other
Universe = UniverseSet()

def power_set(s):
    return chain.from_iterable(combinations(s,r) for r in range(len(s)+1))

def subsets_with(base_set, common):
    assert  common <= base_set
    subsets = power_set(base_set-common)
    return map(common.union, subsets)

class ModelDict(dict):
    def __init__(self, common_knowledge):
        self.model_types = [t for t in common_knowledge['agent_types'] if not issubclass(getattr(t,'type',t), RationalAgent)]
        self.common_knowledge = common_knowledge

    def __missing__(self, agent_id):
        model_dict = {model_type : model_type(self.common_knowledge, agent_id) for model_type in self.model_types}
        self[agent_id] = model_dict
        return model_dict

class JointState(object):
    def __init__(self, prior, common_knowledge):
        self.belief = defaultdict(lambda:copy(prior))
        self.likelihood = defaultdict(lambda:np.zeros_like(prior))
        self.model = ModelDict(common_knowledge)

class ObserverLattice(dict):
    def __init__(self, prior, common_knowledge):
        self[Universe] = JointState(prior, common_knowledge)

    def insert(self, new_set):
        known_sets = list(self.keys())
        new_sets = defaultdict(list)

        for known_set in known_sets:
            intersect = known_set&new_set
            if intersect not in known_sets:
                new_sets[intersect].append(known_set)

        for new_set, supersets in new_sets.iteritems():
            smallest_superset = sorted(supersets, key = len)[0]
            self[new_set] = deepcopy(self[smallest_superset])

    def __missing__(self, agent_set):
        """
        when asked for the joint state of an uknown set,
        return the state of the smallest superset of the given set
        because this is a join-semilattice, there is guaranteed to be a least upper bound
        """
        smallest_superset = sorted(filter(lambda s: s > agent_set, self.keys()), key = len)[0]
        return self[smallest_superset]


class RationalAgent(Agent):
    """
    This subclasses of RationalAgent can simulate Agent sublcasses and other RationalAgent subclasses
    Rational subclasses only require a 'utility' method to be defined
    """
    def __init__(self, genome, world_id = None):
        if world_id:
            self.agent_id = self.world_id = agent_id = world_id
            agent_set = self.agent_set = frozenset([agent_id])
        else:
            self.agent_id = self.world_id = "Overmind"
            agent_set = self.agent_set = frozenset()

        self.common_knowledge = common_knowledge = genome
        self.beta = common_knowledge['beta']

        try:
            self.prior = self.__class__.make_prior(common_knowledge)
        except Exception as e:
            pass

        try:
            lattice = common_knowledge['overmind'].lattice
        except KeyError:
            lattice = ObserverLattice(self.prior, common_knowledge)

        self.lattice = lattice
        self.point_to_top()

    def decide(self, game, participant_ids):
        return self.act(game, participant_ids)

    def act(self, game, participant_ids):
        action_likelihoods = self.__class__._likelihood_of(game, participant_ids, self.belief, self.common_knowledge)
        return np.random.choice(game.actions, p = action_likelihoods)

    def point_to_top(self):
        me = self.lattice[self.agent_set]
        self.belief = me.belief
        self.likelihood = me.likelihood

    def observe(self, simultaneous_observations):
        agent_set = self.agent_set
        agent_types = self.common_knowledge['agent_types']

        #this is a map of type set(id)->id->array
        #it keeps track of the likelihoods that need updating for which subsets of observers
        new_likelihoods = defaultdict(lambda: defaultdict(lambda:np.zeros_like(self.prior)))

        for observation in simultaneous_observations:
            #actor is in participants whose elements are a subset of observers
            participants, observers = [observation[attr] for attr in ['participant_ids','observer_ids']]
            #game, participants, observers, action = observation

            actor = participants[0]

            if not(agent_set <= observers):
                continue

            if observers not in self.lattice:
                self.lattice.insert(observers)

            known_subsets = [s for s in subsets_with(observers, agent_set) if s in self.lattice]
            for subset in known_subsets:

                joint = self.lattice[subset.union(set([actor]))]

                type_likelihoods = []
                for potential_type in agent_types:
                    real_type = getattr(potential_type,'type', potential_type)
                    if not issubclass(real_type, RationalAgent):
                        likelihood = joint.model[actor][potential_type].likelihood_of(**observation)
                    else:
                        likelihood = real_type._likelihood_of(belief = joint.belief,
                                                              common_knowledge = self.common_knowledge,
                                                              **observation)
                    type_likelihoods.append(likelihood)

                new_likelihoods[subset][actor]+= np.log(type_likelihoods)

        #update lattice beliefs using new_likelihoods
        prior =  np.log(self.prior)
        for observers, actor_likelihood in new_likelihoods.iteritems():
            joint = self.lattice[observers]
            for actor, new_likelihood in actor_likelihood.iteritems():
                joint.likelihood[actor] += new_likelihood
                #print "prior", prior
                #print "likelihood", joint.likelihood[actor]
                joint.belief[actor] = normalized(np.exp(prior+joint.likelihood[actor]))

            for agent_type in agent_types:
                real_type = getattr(agent_type,'type', agent_type)
                if not issubclass(real_type, RationalAgent):
                    for observer in observers:
                        joint.model[observer][agent_type].observe(simultaneous_observations)

        self.point_to_top()

    def likelihood_of(self, game, participant_ids, belief = None, common_knowledge = None, tremble = 0, action = None, **kwargs):
        if belief == None:
            belief = self.lattice[self.agent_set].belief
        if common_knowledge == None:
            common_knowledge = self.common_knowledge
        return self.__class__._likelihood_of(game, participant_ids, belief, common_knowledge, tremble, action, **kwargs)

    @classmethod
    def _likelihood_of(cls, game, participant_ids, belief, common_knowledge, tremble = 0, action = None, **kwargs):
        """
        this is a class method so that 'RationalAgents' can model each other without needing to
        build entire copies of each other.
        Here we assume that all 'RationalAgents' share the same theory of mind and priors.
        """
        utility_per_action = [cls.utility(game.payoffs[a], participant_ids, belief, common_knowledge) for a in game.actions]
        likelihoods = add_tremble(softmax(utility_per_action, common_knowledge['beta']), tremble)

        if action == None:
            return likelihoods
        else:
            return likelihoods[game.actions.index(action)]

    @classmethod
    def utility(cls, payoffs, participant_ids, belief, common_knowledge):
        """
        This is a class method for the same reason 'likelihood_of' is
        """
        raise Warning("RationalAgent is a non-functional ABC. Create a subclass with 'utility' method defined")

    @staticmethod
    def make_prior(common_knowledge):
        me = common_knowledge['type']
        prior = common_knowledge['prior']
        agent_types = common_knowledge['agent_types']

        outgroup_prior = (1-prior)/(len(agent_types)-1)
        return  np.array([prior if issubclass(getattr(t,'type',t),RationalAgent) else outgroup_prior for t in agent_types])

### The Rational Agent of interest
class ReciprocalAgent(RationalAgent):
    """
    ReciprocalAgent cares about an agent's payoffs proportional to its belief that they are
    also ReciprocalAgents
    """
    @classmethod
    def utility(cls, payoffs, participant_ids, belief, common_knowledge):
        cls_index = common_knowledge['agent_types'].index(cls)
        weights = [belief[p_id][cls_index] for p_id in participant_ids]
        weights[0] = 1
        return np.dot(payoffs,weights)

    @staticmethod
    def make_prior(common_knowledge):
        me = common_knowledge['type']
        prior = common_knowledge['prior']
        agent_types = common_knowledge['agent_types']

        outgroup_prior = (1-prior)/(len(agent_types)-1)
        return  np.array([prior if t is me else outgroup_prior for t in agent_types])

#class SelfishAgent(RationalAgent):
#"""must change default_genome"""
#    @classmethod
#    def utility(cls, payoffs, participant_ids, belief, common_knowledge):
#        weights = [0 for p_id in participant_ids]
#        weights[0] = 1
#        return np.dot(payoffs,weights)


