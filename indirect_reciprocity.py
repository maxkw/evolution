
from __future__ import division
import scipy as sp
import numpy as np
import pandas as pd
import networkx as nx
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
from joblib import Parallel, delayed
import operator
import os.path
from scipy.special import (psi, polygamma, gammaln)
from operator import mul as multiply
from copy import copy, deepcopy
from functools import partial
from utils import unpickled, pickled
from games import RepeatedPrisonersTournament

print
sns.set_style('white')
sns.set_context('paper', font_scale=1.5)

from collections import MutableMapping

import warnings
warnings.filterwarnings("ignore",category=np.VisibleDeprecationWarning)

from itertools import ifilterfalse
from games import RepeatedPrisonersTournament


"""
Begin:
Agent Definitions

Agents play games and make observations.
They are characteristically defined by their utilities and whether or not they can observe.
"""
class AgentType(type):
    def __str__(cls):
        return cls.__name__
    
class Agent(object):
    __metaclass__ = AgentType
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
        ps = self.decide_likelihood(self, game, agent_ids, tremble = 0)
        action_id = np.squeeze(np.where(np.random.multinomial(1,ps)))
        return game.actions[action_id]

    #For uniformity with Rational Agents
    
    def observe_k(self, observations, k, tremble = 0):
        pass

class Puppet(Agent):
    def __init__(self,world_id = 'puppet'):
        self.world_id = world_id
        
    def decide(self,decision,agent_ids):
        print decision.name
        for i,(action,payoff) in enumerate(decision.payoffs.iteritems()):
            print i,action, payoff
        choice = decision.actions[int(input("enter a number: "))]
        print ""
        return choice
    
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

class RationalAgent(Agent):
    def __init__(self, genome, world_id=None):
        super(RationalAgent, self).__init__(genome, world_id)
        
        #NamedArray mapping agent_type to odds that an arbitrary agent is of that type
        self._type_to_index = dict(map(reversed,enumerate(genome['agent_types'])))
        self.pop_prior = copy(self.genome['prior'])
        
        self.uniform_likelihood = normalized(self.pop_prior*0+1)
        self.rational_models = {}
        self.models = {}
        self.likelihood = {}
        self.belief = {}

    """
    the following are wrappers for convenience
    """
    def belief_that(self, a_id,a_type):
        return self.belief[a_id][self._type_to_index[a_type]]
    def likelihood_that(self, a_id,a_type):
        return self.likelihood[a_id][self._type_to_index[a_type]]

    def purge_models(self, ids):
        #must explicitly use .keys() below because mutation
        for id in (id for id in ids if id in set(self.models.keys())): 
            del self.models[id]
            del self.belief[id]
            del self.likelihood[id]
        for model in agentModels.itervalues():
            model.purge_models(ids)
        
    def initialize_prior(self):
        # This needs to initialize the data structure that is used for the online update
        return self.pop_prior
    
    def initialize_likelihood(self):
        return normalized(self.pop_prior*0+1)

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
        return NotImplementedError

    #@profile
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
                
                if agent_id not in self.rational_models:
                    rational_model = RationalAgent(genome,agent_id)
                    self.rational_models[agent_id] = rational_model
                    self.models[agent_id] = {}
                    self.belief[agent_id] = self.initialize_prior()
                    self.likelihood[agent_id] = self.initialize_likelihood()
                    
                    for rational_type in rational_types:
                        model = self.models[agent_id][rational_type] = rational_type(genome,agent_id)
                        model.belief = rational_model.belief
                        model.likelihood = rational_model.likelihood

                    for o_id in observers:
                        rational_model.belief[o_id] = rational_model.initialize_prior()
                        rational_model.likelihood[o_id] = rational_model.initialize_likelihood()
                
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
                if agent_type in rational_types:
                    #fetch model
                    model = self.models[decider_id][agent_type]
                else:
                    #make model
                    model = agent_type(genome, world_id = decider_id)
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
        for model in self.rational_models.values():
            model.observe_k(observations, K-1, tremble)

            
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

    def use_npArrays(self):
        self.genome['prior'] = np.array(self.genome['prior'])
        
        self.uniform_likelihood = np.array(self.uniform_likelihood)
        self.pop_prior = np.array(self.pop_prior)
        for id in self.belief:
            self.belief[id] = np.array(self.belief[id])
        for id in self.likelihood:
            self.likelihood[id] = np.array(self.likelihood[id])
        for id, model in self.models.items():
            model.use_npArrays()

    def use_NamedArrays(self):
        NamedArray = namedArrayConstructor(tuple(self.genome['agent_types']))
        self.genome['prior'] = NamedArray(self.genome['prior'])
        self.uniform_likelihood = NamedArray(self.uniform_likelihood)
        self.pop_prior = NamedArray(self.pop_prior)
        for id in self.belief:
            self.belief[id] = NamedArray(self.belief[id])
        for id in self.likelihood:
            self.likelihood[id] = NamedArray(self.likelihood[id])
        for id, model in self.models.items():
            model.use_NamedArrays()

class IngroupAgent(RationalAgent):
    def __init__(self, genome, world_id=None):
        super(IngroupAgent, self).__init__(genome, world_id)
        self.ingroup_indices = np.array([self._type_to_index[member] for member in self.ingroup()])
        
    def sample_alpha(self,agent_id):
        if agent_id == self.world_id:
            return 1
        try:
            return sum(self.belief[agent_id][self.ingroup_indices])
        except KeyError:
            self.belief[agent_id] = self.initialize_prior()
            return sum(self.belief[agent_id][self.ingroup_indices])
            
class ReciprocalAgent(IngroupAgent):
    def ingroup(self):
        return [ReciprocalAgent]
        
class NiceReciprocalAgent(IngroupAgent):
    def ingroup(self):
        return [NiceReciprocalAgent,AltruisticAgent]
       

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

def default_params():
    """
    generates clean dict containing rules that determine agent qualities
    this dict provides a simple way to change conditions of experiments

    `games`: Instance of a StageGame object to be used as the default if no game is specified. 

    `stop_condition`: The current `World` matches players together for either a single or a repeated interaction. This expects a reference to a function that takes in the number of rounds played (`n`). The function returns True if the interaction should stop and false if the interaction should continue. 

    `constant_stop_condition`: means that each dyad plays a fixed number of round together up to X. 

    `discount_stop condition`: randomly terminates the interaction with probability `x` and hence does not depend on the round number. 

    `agent_types`: Default agent types which will be initialized through `generate_random_genomes`.

    `beta`: is the beta in the softmax function for the agent decision making. When beta is Inf the agents always select the option that maximizes utility when beta is 0, the agents choose actions randomly. 

    `moran_beta`: is the beta in the evolutionary selection. When moran_beta is 0 the next generation randomly copies from the previous generation. When moran_beta is Inf, all agents in the next generation copy the best agent in the previous generation. 

    I've been using the `RA_` prefix to denote parameters that are special to the reciprocal agent time. 

    `RA_prior`: The prior probability that a ReciprocalAgent has on another agent also being a ReciprocalAgent. The higher this value, the more likely the ReciprocalAgent is to believe the other agent is also ReciprocalAgent without observing any of their actions. 
    TODO: This will probability need to be generalized a bit to handle other Agent types.

    `prior_precision`: This is how confident the agent is in its prior (its like a hyperprior). If it is 0, there is no uncertainty and the prior never changes. If this is non-zero, then `RA_prior` is just the mean of a distribution. The agent will learn the `RA_prior` as it interacts with more and more agents. 

    `p_tremble`: probability of noise in between forming a decision and making an action. 

    `RA_K`: is the number of theory-of-mind recursions to carry out. When RA_K is 0 the agent just tries to infer the type directly, when it is 1, you first infer what each agent knows and then infer what you know based on those agents and so on. 

    """
    agent_types =  [
        SelfishAgent,
        NiceReciprocalAgent,
        AltruisticAgent
    ]
    return {
        'N_agents':2,
        'games': RepeatedPrisonersTournament(10), 
        'stop_condition': [constant_stop_condition,10],
        'agent_types' : agent_types,
        'beta': 3,
        'moran_beta': .1,
        'RA_prior': .8,
        'prior_precision': 0, # setting this to 0 turns off updating the prior
        'p_tremble': 0.0,
        'RA_K': 1,
        'agent_types_world': agent_types
    }



def generate_random_genomes(N_agents, agent_types_world, agent_types, RA_prior, prior_precision,
                            beta, **keys):
    genomes = []
    
    prior = prior_generator(agent_types,RA_prior)
    for _ in range(N_agents):
        genomes.append({
            'type': np.random.choice(agent_types_world),
            'agent_types': agent_types,
            'prior': prior,
            'RA_prior': RA_prior,
            'prior_precision': prior_precision,
            'beta': beta,
            'RA_K': 1
        })
        
    return genomes

def prior_generator(agent_types,RA_prior=False):
    """
    if not given RA_prior  it generates a uniform prior over types
    else splits RA_prior uniformly among all rational types
    """
    
    agent_types = tuple(agent_types)
    type2index = dict(map(reversed,enumerate(agent_types)))
    size = len(agent_types)
    NamedArray = namedArrayConstructor(agent_types)
    rational_types = filter(lambda t: issubclass(t,RationalAgent),agent_types)
    if not (RA_prior or rational_types):
        return np.array(np.ones(size)/size)
    else:
        try:
            normal_prior = (1.0-sum(RA_prior.values()))/(size-len(RA_prior))
            prior = [RA_prior[agent_type] if agent_type in RA_prior
                     else normal_prior for agent_type in agent_types]
            #print prior
        except AttributeError:
            rational_size = len(rational_types)
            rational_prior = RA_prior/float(rational_size)
            normal_prior = (1.0-RA_prior)/(size-rational_size)
            prior = [rational_prior if agent_type in rational_types
                     else normal_prior for agent_type in agent_types]
        return np.array(prior)
    
def default_genome(params = default_params() ,agent_type = False):
    agent_types = params["agent_types"]
    
    if agent_type:
        assert agent_type in agent_types
    else:
        agent_type = np.random.choice(agent_types)
        
    return {
        'type': agent_type,
        'RA_prior': params['RA_prior'],
        'prior_precision': params['prior_precision'],
        'beta': params['beta'],
        'prior': prior_generator(agent_types,params['RA_prior']),
        "agent_types":agent_types,
        'RA_K':params['RA_K']
    }

def generate_proportional_genomes(params = default_params(), agent_proportions = None):
    """
    returns a number of genomes roughly proportional to 'N_agents' in the supplied params.

    'params' is a dict
    'agent_proportions' is a dict of agent_type to a fraction
    ideally all fractions add up to 1 but this is not enforced nor will anything break if unobserved
    
    WARNING:
    this function does not try to preserve the value of "N_agents" it will round up fractions of the population.
    
    for example, if given 'N_agents' = 1, and agent proportions of 1/3,1/3,1/3 the actual population will be 3.
    """
    if not agent_proportions:
        return generate_random_genomes(**params)
    agent_list = []
    pop_size = params['N_agents']
    for agent_type in agent_proportions:
        number = int(math.ceil(pop_size*agent_proportions[agent_type]))
        agent_list.extend([default_genome(params,agent_type) for _ in xrange(number)])
    #print agent_list
    return agent_list




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

        self.agent_types = self.agents[0].genome['agent_types']
        
        self.pop_size = len(self.agents)
        self.tremble = params['p_tremble']
        self.game = params['games']
        self._stop_condition = params['stop_condition']
        #self.stop_condition = partial(*params['stop_condition'])
        self.params = params
        self.last_run_results = {}

        for id, agent in enumerate(self.agents):
            if isinstance(agent,RationalAgent):
                for a_id in range(self.pop_size):
                    if a_id is not id:
                        agent.belief[a_id] = agent.initialize_prior()



    def add_agents(self, genomes):
        for genome, world_id in itertools.izip(genomes,self.counter):
            agent = genome['type'](genome,world_id)
            self.agents.append(agent)
            self.id_to_agent[world_id] = agent

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

    def run(self):
        payoff,observations,record = self.game.play(np.array(self.agents),np.array(self.agents),tremble=self.params['p_tremble'])
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

    g = RepeatedPrisonersTournament().game
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

    #pickled(NiceReciprocalAgent,"./agent.pkl")
    #print unpickled("./agent.pkl")
    #import ipdb; ipdb.set_trace()
    #forgiveness_experiment(overwrite=True)
    #forgiveness_plot() 

    #protection_experiment(overwrite=True)
    #protection_plot()

    #fitness_rounds_experiment(20,overwrite=True)
    #fitness_rounds_plot()

    diagnostics()

"""
Trust game
public goods
chicken?
"""
