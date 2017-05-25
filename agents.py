from __future__ import division
import scipy as sp
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
from utils import softmax, sample_softmax, softmax_utility, flip, normalized, excluding_keys
from copy import deepcopy
from copy import copy, deepcopy
from utils import unpickled, pickled, HashableDict, issubclass

PRETTY_KEYS = {"RA_prior": "prior",
               "RA_K": 'K'}

def is_agent_type(instance, base):
    try:
        return issubclass(instance, base)
    except TypeError:
        return issubclass(instance.type, base)

class AgentType(type):
    def __str__(cls):
        return cls.__name__

    def __repr__(cls):
        return str(cls)

    def __hash__(cls):
        return hash(cls.__name__)

    def __eq__(cls, other):
        return hash(cls) == hash(other)

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
        self.genome = deepcopy(genome)
        self.beta = self.genome['beta']
        self.world_id = world_id
        self.belief = dict()
        self.likelihood = dict()

    def utility(self, payoffs, agent_ids):
        return sum(self._utility(payoff, id) for payoff, id in itertools.izip(payoffs, agent_ids))

    def _utility(self, payoffs, agent_ids):
        raise NotImplementedError

    def decide_likelihood(self, game, agents, tremble=0):
        # The first agent is always the deciding agent
        # Only have one action so just pick it
        if len(game.actions) == 1:
            # Returning a probability a vector
            return np.array([1])

        Us = np.array([self.utility(game.payoffs[action], agents)
                       for action in game.actions])

        return self.add_tremble(softmax(Us, self.beta), tremble)

    def add_tremble(self, p, tremble):
        if tremble == 0:
            return p
        else:
            return (1 - tremble) * p + tremble * np.ones(len(p)) / len(p)

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
        rational_types = filter(lambda t: issubclass(
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
        self.uniform_likelihood = normalized(self.pop_prior * 0 + 1)
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
        sample_alpha = self.sample_alpha
        weights = map(sample_alpha, agent_ids)
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
        rational_types = filter(lambda t: issubclass(
            t, RationalAgent), agent_types)

        my_id = self.world_id
        observations = filter(lambda obs: my_id in obs[2], observations)
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

            if decider_id == my_id:
                continue

            likelihood = []
            append_to_likelihood = likelihood.append
            action_index = game.action_lookup[action]

            # calculate the normalized likelihood for each type
            for agent_type in agent_types:
                model = self.model[decider_id][agent_type]
                append_to_likelihood(model.decide_likelihood(
                    game, participants, tremble)[action_index])

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
            self.belief[decider_id] = belief = normalized(
                self.belief[decider_id])

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
            if issubclass(a_type, i):
                return True
            
        return False

class ReciprocalAgent(IngroupAgent):
    @staticmethod
    def ingroup():
        return [ReciprocalAgent]

class PrefabAgent(Agent):
    def __init__(self, a_type, **genome_kwargs):
        self.type = a_type
        self.__name__ = str(a_type) + "(%s)" % ",".join(
            ["%s=%s" % (key, val) for key, val in sorted(genome_kwargs.iteritems())])
        try:
            tom = genome_kwargs['agent_types']
            genome_kwargs['agent_types'] = tuple(
                t if t is not 'self' else self for t in tom)
        except:
            pass
        self.genome = HashableDict(genome_kwargs)

    def __call__(self, genome, world_id=None):
        return self.type(dict(genome, **self.genome), world_id=world_id)

    def short_name(self, *without):
        genome = excluding_keys(self.genome, *without)
        return str(self.type) + "(%s)" % ",".join(["%s=%s" % (PRETTY_KEYS.get(key, key), val) for key, val in sorted(genome.iteritems())])

    def __str__(self):
        return self.__name__

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        if hash(self.type) == hash(other):
            return True
        return False

    def ingroup(self):
        return self.type.ingroup()

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

    
class Pavlov(ClassicAgent):
    def __init__(self, genome, world_id=None):
        self.genome = deepcopy(genome)
        self.world_id = world_id
        self.strats = strats = [
            {'give': 1, 'keep': 0},
            {'keep': 1, 'give': 0}]
        self.strat_index = 0

    def observe(self, observations):
        # Can only judge a case where there are two observations
        # TODO: This will not work on sequential PD since it requires two simultaneous observations
        obs1, obs2 = observations
        actions = [obs1[3], obs2[3]]
        players = [obs1[1][0], obs2[1][0]]
        me = self.world_id
        if me in players:
            a = players.index(me)
            o = (a + 1) % 2
            if actions[o] == 'keep':
                self.strat_index = (self.strat_index + 1) % 2

    def decide_likelihood(self, game, agents=None, tremble=None):
        return self.add_tremble(np.array([self.strats[self.strat_index][action] for action in game.actions]), tremble)


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
        return self.add_tremble(np.array([rules[last_action][action] for action in game.actions]), tremble)

    def observe_k(self, observations, *args, **kwargs):
        self.observe(observations)

    def observe(self, observations):
        assert len(observations) == 2
        for observation in observations:
            game, participants, observers, action = observation
            if self.world_id not in observers:
                continue
            if self.world_id not in participants:
                continue
            decider_id = participants[0]
            if decider_id == self.world_id:
                continue

            self.cooperated = action is "give"


class AllC(ClassicAgent):
    def decide_likelihood(self, game, agents=None, tremble=None):
        odds = {'give': 1,
                'keep': 0}
        return self.add_tremble(np.array([odds[action] for action in game.actions]), tremble)


class AllD(ClassicAgent):
    def decide_likelihood(self, game, agents=None, tremble=None):
        odds = {'give': 0,
                'keep': 1}
        return self.add_tremble(np.array([odds[action] for action in game.actions]), tremble)


class RandomAgent(ClassicAgent):
    def decide_likelihood(self, game, *args, **kwargs):
        l = len(game.actions)
        return (1 / l,) * l


class wTypeDict(dict):
    def __init__(self, genome, agent_id=None):
        agent_types = tuple(a_type for a_type in genome['agent_types'] if a_type != WeAgent)
        self.observers = list()  # [model]
        for agent_type in agent_types:
            m = agent_type(genome, agent_id)
            if agent_type in [gTFT, Pavlov]:
                self.observers.append(m)
                
            self[agent_type] = m

    def observe_k(self, observations, k, tremble):
        for observer in self.observers:
            observer.observe_k(observations, k, tremble)


class wAgentDict(dict):
    def __init__(self, genome):
        self.genome = genome

    def __missing__(self, agent_id):
        ret = self[agent_id] = wTypeDict(self.genome, agent_id)
        return ret


class WeAgent(Agent):
    def __init__(self, genome, world_id):
        self.world_id = world_id
        self.genome = genome = copy(genome)
        self.genome['agent_types'] = tuple(t if t != 'self' else genome['type'] for t in genome['agent_types'])

        self.beta = genome['beta']

        self._type_to_index = dict(map(reversed, enumerate(genome['agent_types'])))

        self.models = wAgentDict(genome)

        RA_prior = genome['RA_prior']
        non_WA_prior = (1 - RA_prior) / (len(genome['agent_types']) - 1)
        self.pop_prior = np.array([RA_prior if t is WeAgent else non_WA_prior for t in genome['agent_types']])

        self.belief = ConstantDefaultDict(self.pop_prior)
        self.likelihood = ConstantDefaultDict(np.zeros_like(self.pop_prior))

    def utility(self, payoffs, agent_ids):
        t = self.genome['agent_types'].index(self.genome['type'])
        weights = [1] + [self.belief[a][t] for a in agent_ids[1:]]
        #weights = [self.belief[a][t] for a in agent_ids]
        return sum(p * w for p, w in zip(payoffs, weights))

    def belief_that(self, a_id, a_type):
        if a_type in self._type_to_index:
            return self.belief[a_id][self._type_to_index[a_type]]
        else:
            return 0

    def decide_likelihood(self, game, agents, tremble):
        if len(game.actions) == 1:
            return np.array([1])

        Us = np.array([self.utility(game.payoffs[action], agents)
                       for action in game.actions])
        return self.add_tremble(softmax(Us, self.beta), tremble)

    def observe(self, observations):
        agent_types = self.genome['agent_types']
        tremble = self.genome['tremble']
        new_likelihoods = defaultdict(int)
        for observation in observations:
            game, participants, observers, action = observation
            decider_id = participants[0]
            action_index = game.action_lookup[action]

            likelihood = []
            for agent_type in agent_types:
                if agent_type == WeAgent:
                    model = self
                else:
                    model = self.models[decider_id][agent_type]
                likelihood.append(model.decide_likelihood(game, participants, tremble)[action_index])

            new_likelihoods[decider_id] += np.log(likelihood)

        prior = np.log(self.pop_prior)
        for decider_id, new_likelihood in new_likelihoods.iteritems():
            self.likelihood[decider_id] += new_likelihood
            likelihood = self.likelihood[decider_id]
            self.belief[decider_id] = np.exp(prior + likelihood)
            self.belief[decider_id] = normalized(self.belief[decider_id])

        # Update the models of the other agents
        for a_id, model in self.models.iteritems():
            model.observe_k(observations, 0, tremble)


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

