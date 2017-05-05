from __future__ import division
import scipy as sp
import numpy as np
import pandas as pd
import itertools
from collections import Counter,defaultdict,Iterable
import matplotlib.pyplot as plt
import seaborn as sns
import math
# import dirichlet
from utils import softmax, sample_softmax, softmax_utility, flip, namedArrayConstructor, normalized, constraint_min
from copy import deepcopy
from pprint import pprint
from scipy.spatial.distance import cosine
import operator
import os.path
from scipy.special import (psi, polygamma, gammaln)
from operator import mul as multiply
from copy import copy, deepcopy
from functools import partial
from utils import unpickled, pickled, HashableDict,issubclass


sns.set_style('white')
sns.set_context('paper', font_scale=1.5)

from collections import MutableMapping

import warnings
warnings.filterwarnings("ignore",category=np.VisibleDeprecationWarning)

from itertools import ifilterfalse
#from games import RepeatedPrisonersTournament


"""
Begin:
Agent Definitions

Agents play games and make observations.
They are characteristically defined by their utilities and whether or not they can observe.
"""



class AgentType(type):
    def __str__(cls):
        return cls.__name__
    def __repr__(cls):
        return str(cls)
    def __hash__(cls):
        return hash(cls.__name__)
    #def __reduce__(self):
    #    return (eval,(self.__name__,))

class Agent(object):
    __metaclass__ = AgentType
    def __new__(cls, genome=None, world_id=None, **kwargs):
        if not genome and kwargs:
            return PrefabAgent(cls,**kwargs)
        else:
            return object.__new__(cls)

    def __init__(self, genome, world_id=None):
        self.genome = deepcopy(genome)
        self.beta = self.genome['beta']
        self.world_id = world_id
        self.belief = dict()
        self.likelihood = dict()
        
    def utility(self, payoffs, agent_ids):
        return sum(self._utility(payoff,id) for payoff,id in itertools.izip(payoffs,agent_ids))

    def _utility(self, payoffs, agent_ids):
        raise NotImplementedError

    @staticmethod
    def decide_likelihood(deciding_agent, game, agents, tremble = 0):
        """
        recieves:
        1)deciding_agent, an agent of a particular type

        returns:
        a probability vector representing what I think the deciding agent would do in this game?
        """

        # The first agent is always the deciding agent
        # Only have one action so just pick it
        if len(game.actions) == 1:
            # Returning a probability a vector
            return np.array([1])

        # TODO: Get probability of the other players taking an action,
        # otherwise this only works for dictator game. For now just
        # assume its uniform since it doesn't matter for dictator
        # game. Can/Should use the belief distribution. May need to do
        # a logit response for simultaneous move games.
        Us = np.array([deciding_agent.utility(game.payoffs[action], agents)
                       for action in game.actions])

        return (1-tremble) * softmax(Us, deciding_agent.beta) + tremble * np.ones(len(Us))/len(Us)

    def decide(self, game, agent_ids):
        # Tremble is always 0 for decisions since tremble happens in
        # the world, not the agent
        ps = self.decide_likelihood(self, game, agent_ids, tremble = self.genome['tremble'])
        action_id = np.squeeze(np.where(np.random.multinomial(1,ps)))
        return game.actions[action_id]

    #For uniformity with Rational Agents

    def observe(self,observations):
        pass
    def observe_k(self, observations, k, tremble = 0):
        pass

class Puppet(Agent):
    def __init__(self, world_id = 'puppet'):
        self.world_id = world_id
        self.belief = self.likelihood = None
        
    def decide(self,decision,agent_ids):
        print decision.name
        for i,(action,payoff) in enumerate(decision.payoffs.iteritems()):
            print i,action, payoff
        choice = decision.actions[int(input("enter a number: "))]
        print ""
        return choice
class TitForTat(Agent):
    """
    Abstract Tit-For-Tat agent
    collaboration is maximizing the sum of payoffs
    defection is anything else
    """
    def __init__(self, genome, world_id=None):
        self.world_id = world_id
        self.genome = genome
        self.grudge = defaultdict(int)
        self.punishments = 1
        self.belief = {}
        self.likelihood = {}
    def utility(self,payoffs,agent_ids):
        if agent_ids[0] == self.world_id and self.grudge[agent_ids[1]]:
            return -payoffs[1]
        else:
            return self.social_good(payoffs)
        
    def social_good(self,payoffs):
        return sum(payoffs)
    
    def decide(self,game,agent_ids):
        if self.grudge[agent_ids[1]]:
            action = min(game.actions, key = lambda a: game.payoffs[a][1])
            self.grudge[agent_ids[1]] -= 1
        else:
            action = max(game.actions, key = lambda a: sum(game.payoffs[a]))
            
        return action
        
    def observe_k(self,observations,k,tremble=0):
        for observation in observations:
            decision,participants,observers,actions = observation
            decider_id = participants[0]
            my_id = self.world_id
            
            if decider_id == my_id or my_id not in participants: continue

            best_action = max(decision.actions,key=lambda a: sum(decision.payoffs[a]))

            if actions is not best_action:
                self.grudge[decider_id] += self.punishments
    
    
class SelfishAgent(Agent):
    def __init__(self, genome, world_id=None):
        super(SelfishAgent, self).__init__(genome, world_id)
    
    def utility(self, payoffs, agent_ids):

        weights = [1 if agent_id == self.world_id else 0
                   for agent_id in agent_ids]   
        return sum(itertools.imap(multiply,weights,payoffs))

class AltruisticAgent(Agent):

    def __init__(self, genome, world_id=None):
        super(AltruisticAgent, self).__init__(genome, world_id)

    def utility(self,payoffs,agent_ids):
        weights = [1]*len(agent_ids)
        return sum(itertools.imap(multiply,weights,payoffs))

class keydefaultdict(defaultdict):
    def __missing__(self,key):
        ret = self[key] = self.default_factory(key)
        return ret

class constantdefaultdict(dict):
    def __init__(self,val):
        self.val = val
    def __missing__(self,key):
        ret = self[key] = copy(self.val)
        return ret
    
class TypeDict(dict):
    def __init__(self,genome,agent_id=None):
        agent_types = genome['agent_types']
        rational_types = filter(lambda t: issubclass(t,RationalAgent),agent_types)
        model = self.agent = RationalAgent(genome,agent_id)
        belief = self.belief = model.belief #= defaultdict(model.initialize_prior)
        likelihood = self.likelihood = model.likelihood# = defaultdict(model.initialize_likelihood)
        
        for agent_type in agent_types:
            dict.__setitem__(self,agent_type,agent_type(genome,agent_id))

        for rational_type in rational_types:
            r_model = dict.__getitem__(self,rational_type)
            r_model.belief = belief
            r_model.likelihood = likelihood
            
class AgentDict(dict):
    def __init__(self,genome):
        self.genome = genome
    def __missing__(self,agent_id):
        ret = self[agent_id] = TypeDict(self.genome,agent_id)
        return ret
        
class RationalAgent(Agent):
    #def __new__(cls, genome = None, world_id = None, prior = None):
    #    if not (genome or world_id) and prior:
    #        return AgentWithPrior(cls,prior)
    #    else:
    #        return Agent.__new__(cls,genome,world_id)
    def __init__(self, genome, world_id=None):#, *args, **kwargs):
        super(RationalAgent, self).__init__(genome, world_id)
        #print genome
        #print world_id

        #NamedArray mapping agent_type to odds that an arbitrary agent is of that type
        self._type_to_index = dict(map(reversed,enumerate(genome['agent_types'])))
        self.pop_prior = copy(self.genome['prior'])

        self.uniform_likelihood = normalized(self.pop_prior*0+1)
        self.model = AgentDict(genome)
        
        self.likelihood = constantdefaultdict(self.initialize_likelihood())
        self.belief = constantdefaultdict(self.initialize_prior())

    """
    the following are wrappers for convenience
    """
    def belief_that(self, a_id,a_type):
        try:
            return self.belief[a_id][self._type_to_index[a_type]]
        except KeyError:
            return 0

    def likelihood_that(self, a_id,a_type):
        return self.likelihood[a_id][self._type_to_index[a_type]]
    def k_belief(self,a_ids,a_type):
        """
        call with list of agent_ids. If you want A's belief that B believes A is reciprocal
        agent_A.k_belief(('B','A'),ReciprocalAgent)
        """
        try:
            [a_id] = a_ids
            return self.belief_that(a_id, a_type)
        except ValueError:
            a_id = a_ids[0]
            a_ids = a_ids[1:]
            return self.model[a_id].agent.k_belief(a_ids,a_type)

    def purge_models(self, ids):
        #must explicitly use .keys() below because mutation
        for id in (id for id in ids if id in set(self.model.keys())): 
            del self.model[id]
            del self.belief[id]
            del self.likelihood[id]
        for model in self.model.itervalues():
            model.purge_models(ids)
        
    def initialize_prior(self):
        # This needs to initialize the data structure that is used for the online update
        return self.pop_prior
    
    def initialize_likelihood(self):
        return normalized(self.pop_prior*0+1)

    def initialize_model(self, agent_id):
        genome = self.genome
        agent_types = genome['agent_types']
        rational_types = filter(lambda t: issubclass(t,RationalAgent),agent_types)
        
        self.agent = rational_model = RationalAgent(genome,agent_id)
        self.rational_models[agent_id] = rational_model
        self.models[agent_id] = {}
        
        for rational_type in rational_types:
            model = self.models[agent_id][rational_type] = rational_type(genome,agent_id)
            model.belief = rational_model.belief
            model.likelihood = rational_model.likelihood
            model.models = rational_model.models
        return self.models[agent_id]

    def _use_defaultdicts(self):
        belief,likelihood,models = copy(self.belief),copy(self.likelihood),copy(self.models)
        
        self.belief.update(belief)
        
        self.likelihood = defaultdict(self.initialize_likelihood)
        self.likelihood.update(likelihood)
        
        self.models = keydefaultdict(self.initialize_model)
        self.models.update(models)

        for model in self.rational_models.values():
            model._use_defaultdicts()
        return self

        
    def utility(self, payoffs, agent_ids):
        sample_alpha = self.sample_alpha
        weights = map(sample_alpha,agent_ids)
        return sum(itertools.imap(multiply,weights,payoffs))

    def sample_alpha(self, agent_id):
        """
        this function basically tells us how much we care about
        a particular agent's payoff as a function of our beliefs about them
        every reciprocal agent type is defined by just defining this function
        """
                   
        # TODO: Decide what to do here? Should this be sampling and
        # then giving full weight? Or should it be weighting how much
        # we care? The weighted version is worse at punishing the bad
        # guys since it will still weight them a bit even when its
        # highly unlikely that they are reciprocal... Maybe this is
        # just a downside of being a nice person?0
        
        # return int(flip(belief))
        print "Rational Agents don't know how to choose, subclasses do"
        raise NotImplementedError

    #@profile
    def observe(self,observations):
        if self.genome['RA_prior'] in [1,0]:
            return
        self.observe_k(observations,self.genome['RA_K'],self.genome['tremble'])

    def observe_k(self, observations, K, tremble = 0):
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

        #if K < 0: return
        genome = self.genome
        agent_types = genome['agent_types']
        rational_types = filter(lambda t: issubclass(t,RationalAgent),agent_types)

        my_id = self.world_id 
        observations = filter(lambda obs: my_id in obs[2], observations)
        for observation in observations:
            observers = observation[2]
            
            for agent_id in observers:
                if agent_id == self.world_id: continue
                
                if agent_id not in self.model:
                    model = self.model[agent_id]
                    #rational_model = RationalAgent(genome,agent_id)
                    #self.rational_models[agent_id] = rational_model
                    #self.models[agent_id] = {}
                    #self.belief[agent_id] = self.initialize_prior()
                    #self.likelihood[agent_id] = self.initialize_likelihood()
                    
                    #for rational_type in rational_types:
                    #    model = self.models[agent_id][rational_type] = rational_type(genome,agent_id)
                    #    model.belief = rational_model.belief
                    #    model.likelihood = rational_model.likelihood

                    for o_id in observers:
                        model.belief[o_id]# = model.agent.initialize_prior()
                        model.likelihood[o_id]# = model.agent.initialize_likelihood()
                
        for observation in observations:
            game, participants, observers, action = observation

            decider_id= participants[0]
            
            if decider_id == my_id: continue

            likelihood = []
            append_to_likelihood = likelihood.append
            decide_likelihood = Agent.decide_likelihood
            action_index = game.action_lookup[action]

            #calculate the normalized likelihood for each type
            for agent_type in agent_types:
                model = self.model[decider_id][agent_type]
                #if agent_type in rational_types:
                    #fetch model
                #    model = self.model[decider_id][agent_type]
                #else:
                    #make model
                #    model = agent_type(genome, world_id = decider_id)
                append_to_likelihood(decide_likelihood(model,game,participants,tremble)[action_index])
                

            self.likelihood[decider_id] *= likelihood
            self.likelihood[decider_id] = normalized(self.likelihood[decider_id])
            
            prior = self.pop_prior
            likelihood = self.likelihood[decider_id]
            self.belief[decider_id] = prior*likelihood/np.dot(prior,likelihood)     


        # Observe the other person, when this code runs at K=0
        # nothing will happen because of the return at the top
        # of the function.

        
        if K == 0:return
        for model in self.model.values():
            model.agent.observe_k(observations, K-1, tremble)

            
        # if K>0:
        #     self.update_prior()

    def update_prior(self):
        if self.genome['prior_precision'] == 0: return
        
        D = list()
        n_agent_types = len(self.genome['agent_types'])
        NamedArray = namedArrayConstructor(self.genome['agent_types'])
        
        prior = self.genome['prior'] * self.genome['prior_precision']

        def ll(p):
            p[-1] = 1-sum(p[0:-1])
            like = 0
            for order in itertools.product(range(n_agent_types), repeat=len(self.likelihood)):
                agent_counts = [sum(np.array(order)==t) for t in range(n_agent_types)]
                counts = np.array(prior + agent_counts)

            #     # lnB = np.sum(gammaln(counts)) - gammaln(np.sum(counts))
            #     # pdf = np.exp( - lnB + np.sum((np.log(p.T) * (counts - 1)).T, 0) )
            #     # term = np.array(pdf) #FIXME: Shouldn't need this... named array issue
            
                term = sp.stats.dirichlet.pdf(p, counts)
                for a_id, agent_type in zip(self.likelihood.keys(), order):
                    # wrong not a dependence on p (theta) a dependence on alpha which is the prior
                    # belief = (self.likelihood[a_id][agent_type]*p[agent_type]) / np.dot(p, self.likelihood[a_id])

                    # still wrong since this just using the mean and not doing the full integration
                    belief = (self.likelihood[a_id][self._type_to_index[agent_type]]*prior[self._type_to_index[agent_type]]) / np.dot(prior, self.likelihood[a_id])

                    term *= belief

                like += term

            return -(np.log(like))

        out = constraint_min(ll, np.ones(n_agent_types)/n_agent_types)
        
        print out
        
        # FIXME: Need to save these out to the pop_prior and then update the belief of all the agents by using the new prior when combining the likelihood and prior. 
        
        # self.pop_prior = {
            # ReciprocalAgent.__name__: out.x[0],
            # SelfishAgent.__name__ : 1-out.x[0]
        # }

class IngroupAgent(RationalAgent):
    def __init__(self, genome, world_id=None):
        super(IngroupAgent, self).__init__(genome, world_id)
        self.ingroup_indices = np.array([self._type_to_index[agent_type]
                                         for agent_type in genome['agent_types']
                                         if self.is_in_ingroup(agent_type)])

    def sample_alpha(self,agent_id):
        if agent_id == self.world_id:
            return 1
        try:
            return sum(self.belief[agent_id][self.ingroup_indices])
        except IndexError:
            return sum([self.belief_that(agent_id,t) for t in self.ingroup()])
        except KeyError:
            self.belief[agent_id] = self.initialize_prior()
            return sum(self.belief[agent_id][self.ingroup_indices])

    def is_in_ingroup(self,a_type):
        for i in self.ingroup():
            if is_agent_type(a_type,i):
                return True
        return False

class ReciprocalAgent(IngroupAgent):
    @staticmethod
    def ingroup():
        return [ReciprocalAgent]
class NiceReciprocalAgent(IngroupAgent):
    @staticmethod
    def ingroup():
        return [NiceReciprocalAgent,AltruisticAgent]

class prefabABC(Agent):
    def __new__(self,genome,world_id = None):
        return self.type(dict(genome,**self.genome), world_id = world_id)
import copy_reg
def pickle_prefab(prefab):
    return eval,prefab.name

class PrefabAgent2(type):
    def __new__(cls,a_type,**genome_kwargs):
        name = str(a_type)+"(%s)" % ",".join(["%s=%s" % (key,val) for key,val in sorted(genome_kwargs.iteritems())])
        t = AgentType(name,(prefabABC,a_type),{"genome":genome_kwargs,"type":a_type})
        class prefab(a_type):
            def __new__(self,genome,world_id = None):
                return a_type(dict(genome,**genome_kwargs), world_id = world_id)
        prefab.name = name
        copy_reg.pickle(prefab,pickle_prefab)
        return prefab
class PrefabAgent(Agent):
    def __init__(self,a_type,**genome_kwargs):
        self.type = a_type
        self.genome = HashableDict(genome_kwargs)
        self.__name__ = str(a_type)+"(%s)" % ",".join(["%s=%s" % (key,val) for key,val in sorted(genome_kwargs.iteritems())])

    def __call__(self,genome,world_id = None):
        return self.type(dict(genome,**self.genome), world_id = world_id)
    def __str__(self):
        return self.__name__
    def __repr__(self):
        return str(self)
    def __hash__(self):
        return hash(str(self))
    def __eq__(self,other):
        return hash(self)==hash(other)

    def ingroup(self):
        return self.type.ingroup()

def is_agent_type(instance,base):
    try:
        return issubclass(instance,base)
    except TypeError:
        return issubclass(instance.type, base)

class gTFT(Agent):
    def __init__(self, genome, world_id = None):
        self.y = genome['y']
        self.p = genome['p']
        self.q = genome['q']
        self.world_id = world_id
        self.genome = genome
        self.cooperated = "First"

    def decide_likelihood(self,game):
        if self.cooperated == "First":
            coop_prob = self.y
        else:
            coop_prob = self.p if self.cooperated else self.q
        ps = []
    
        for action in game.actions:
            if action == "give":
                ps.append(coop_prob)
            elif action == "keep":
                ps.append(1-coop_prob)
            else:
                assert False

        return ps

    def decide(self, game, agent_ids):
        # Tremble is always 0 for decisions since tremble happens in
        # the world, not the agent
        ps = self.decide_likelihood(game)
        action_id = np.squeeze(np.where(np.random.multinomial(1,ps)))
        action = game.actions[action_id]

        return action

    def observe(self,observations):
        assert len(observations)==2
        for observation in observations:
            game, participants, observers, action = observation
            decider_id = participants[0]
            if decider_id == self.world_id: continue
            assert participants[1] == self.world_id
            self.cooperated = action is "give"

class OpportunisticRA(RationalAgent):
    """
    The idea here is "nice to anyone who would be nice to me".
    It's interesting to try to define this guy's behavior
    should it be anyone who would give him 100% of his payoff?
    or should he take whatever he can get?
    What about "nice to anyone who would be nice to me if they knew the real me"?
    
    try these utilities
    summing over types: 
    (belief that they are of a type)*(how much that type values me, according to their beliefs about me)
    (belief that they are of a type)*(how much that type would value me if they knew the real me)
    (belief in type) if (they absolutely value my type)
    (belief in type) if (they absolutely value who they think I am)
    
    """
    pass

"""
End Agent Definitions
"""

"""
Start Essential Auxiliary Definitions

These define structures that are necessary to define other structures in this file, namely params and genomes
"""

def discount_stop_condition(x,n):
    return not flip(x)
def constant_stop_condition(x,n):
    return n >= x

"""
End Essential Auxiliary Definitions
"""

class World(object):
    # TODO: spatial or interaction probabilities
    
    def __init__(self, params, genomes):
        self.agents = []
        self.counter = itertools.count()
        self.id_to_agent = {}
        
        self.add_agents(genomes)

        #self.agent_types = self.agents[0].genome['agent_types']
        
        self.pop_size = len(self.agents)
        self.tremble = params['p_tremble']
        self.game = params['games']
        #self._stop_condition = params['stop_condition']
        #self.stop_condition = partial(*params['stop_condition'])
        self.params = params
        self.last_run_results = {}

        for id, agent in enumerate(self.agents):
            if isinstance(agent,RationalAgent):
                for a_id in range(self.pop_size):
                    if a_id is not id:
                        agent.belief[a_id] = agent.initialize_prior()



    def add_agents(self, genomes):
        self.agents = list(self.agents)
        for genome, world_id in itertools.izip(genomes,self.counter):
            agent = genome['type'](genome,world_id)
            self.agents.append(agent)
            self.id_to_agent[world_id] = agent
        self.agents = np.array(self.agents)

    def purge_agents(self, ids):
        pass
            
    def evolve(self, fitness, p=1, beta=1, mu=0.05):
        # FIXME: Need to increment the id's. Can't just make new
        # agents, otherwise new agents will be treated as old agents
        # if they share the same ID


        #problem solved?
        assert 0 # BROKEN (See above)
        
        die = np.random.choice(range(self.pop_size), int(p*self.pop_size), replace=False)
        random = np.random.choice(range(self.pop_size), int(mu*self.pop_size), replace=False)
        for a in die:
            copy_id = sample_softmax(fitness, beta)
            self.agents[a] = self.agents[copy_id].__class__(self.agents[copy_id].genome)

        new_genomes = generate_random_genomes(len(random), self.params['agent_types'], self.params['RA_prior'], self.params['beta'])
        for a, ng in zip(random, new_genomes):
            self.agents[a] = ng['type'](ng)

    def cull_list(self):
        """
        returns a list of agent_ids to be removed from the population
        
        goes through every agent and adds them to list with probability proportional
        to age and maybe fitness
        
        """
        pass
    
    def reproduce(self):
        """
        returns a list of agent_ids to reproduce

        runs through list of agents and selects for reproduction with probability 
        proportional to fitness
        """
        pass

    def run(self,agents = None,observers = None,notes = {}):
        agents = np.array(self.agents)
        if notes:
            payoff, observations, record = self.game.play(agents, agents, tremble=self.params['p_tremble'],notes = notes)
        else:
            payoff, observations, record = self.game.play(agents ,agents, tremble=self.params['p_tremble'])
        return payoff, record

def diagnostics():

    params = default_params()

    
    typesClassic = agent_types = (ReciprocalAgent,SelfishAgent)
    
    params['stop_condition'] = [constant_stop_condition,10]
    params['p_tremble'] = 0
    params['RA_prior'] = 0.8
    params['RA_prior_precision'] = 0
    prior = prior_generator(agent_types,params['RA_prior'])

    w = World(params, [default_genome(params,NiceReciprocalAgent)])

    #g = RepeatedPrisonersTournament().game
    #print g.name
    #for key,val in g.__dict__.items():
    #    print key,val
    #    pickled(val,"./world.pkl")

    #
    pickled(w,"./world.pkl")
    w = unpickled("./world.pkl")
    return "YAY"
    
    
    K = 0
    
    observations= [
        (w.game, [0, 1], [0, 1], 'give'),
        (w.game, [1, 0], [0, 1], 'keep'),
    ]

    for a in w.agents:
        a.observe_k(observations, K)
    for a in w.agents:
        print a.world_id,a.belief
    assert_almost_equal = np.testing.assert_almost_equal
    assert_almost_equal(w.agents[0].belief[1],[0.37329758,  0.62670242])
    
    observations= [
        (w.game, [0, 2], [0, 2], 'give'),
        (w.game, [2, 0], [0, 2], 'give'),
    ]

    for a in w.agents:
        a.observe_k(observations, K)
    for a in w.agents:
        print a.world_id,a.belief
        assert_almost_equal(w.agents[0].belief[1],[ 0.37329758,  0.62670242])
        assert_almost_equal(w.agents[0].belief[2],[ 0.98637196,  0.01362804])
        assert_almost_equal(w.agents[1].belief[0],[ 0.98637196,  0.01362804])

if __name__ == '__main__':
    NRA = NiceReciprocalAgent
    a = NiceReciprocalAgent(RA_prior = 1)
    print a
    print type(a)
    print a.__name__
    print type(eval(a.__name__))
    d = {a:True}
    print d[a]
    #print a.hi
    #print a.__str__
    #print a.__reduce__(a)

    #pickled(NiceReciprocalAgent,"./agent.pkl")
    #print unpickled("./agent.pkl")
    #import ipdb; ipdb.set_trace()
    #forgiveness_experiment(overwrite=True)
    #forgiveness_plot() 

    #protection_experiment(overwrite=True)
    #protection_plot()

    #fitness_rounds_experiment(20,overwrite=True)
    #fitness_rounds_plot()

    #diagnostics()
    pass

"""
Trust game
public goods
chicken?
"""
