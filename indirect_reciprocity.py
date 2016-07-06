from __future__ import division
import scipy as sp
import numpy as np
import pandas as pd
import networkx as nx
import itertools
import random
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

from copy import copy, deepcopy

print
sns.set_style('white')
sns.set_context('paper', font_scale=1.5)

from collections import MutableMapping

import warnings
warnings.filterwarnings("ignore",category=np.VisibleDeprecationWarning)

from itertools import ifilterfalse
def without(source,*blacklists):
    try:
        [blacklist] = blacklists
        if isinstance(blacklist,Iterable):
            blacklist = list(blacklist)
            return ifilterfalse(lambda x: x in blacklist, source)
        else:
            return ifilterfalse(lambda x: x is blacklist, source)
    except ValueError:
        return without(without(source,blacklists[0]),blacklists[1:])

def printing(obj):
    print obj
    return obj

class SerialGame(object):
    def __init__(self, *decisions):
        self.decisions = decisions
    def start(self):
        self.payoff = []
        self.actions = []
        self.stage = 0
    def play(self,action):
        decision,stage = self.decisions[stage]
        
class Game(object):
    def __init__(self, payoffs):
        pass

class StageGame(object):
    """
    base class for all games, initialized with a payoff dict
    TODO: Build a game generator that can capture an ultimatum game by stringing together simple games. For instance,
    GAME 1: P1 gets 10 and can share X [0 through 10] with P2
    GAME 2: P2 can do nothing and keep X, or lose -X and have P1 lose 10-X
    
    """
    def __init__(self, payoffs):
        self.N_players = len(payoffs.values()[0])
     # Create the action space
        self.actions = payoffs.keys()
        self.action_lookup = {a:i for i, a in enumerate(self.actions)}
        self.payoffs = payoffs

    def __call__(self,action):
        return action, self.payoffs[action]
        
class BinaryDictator(StageGame):
    """
    dictator game in which cooperation produces more rewards in total
    """
    def __init__(self, endowment = 0, cost = 1, benefit = 2):
        payoffs = {
            "keep": (endowment, 0),
            "give": (endowment-cost, benefit),
        }
        super(BinaryDictator, self).__init__(payoffs)

class UltimatumPropose(StageGame):
    def __init__(self, endowment):
        payoffs = {"keep {}/give {}".format(keep, give) : (keep, give) for keep, give in
                   ((endowment - give, give) for give in xrange(endowment))}
        super(UltimatumPropose, self).__init__(payoffs)
        

class CostlyAllocationGame(StageGame):
    """
    Three player version of dictator
    """
    def __init__(self, endowment = 0, cost = 1, benefit = 2):
        payoffs = {
            "give 1": (endowment-cost, benefit, 0),
            "give 2": (endowment-cost, 0, benefit),
            "keep": (endowment, 0, 0),
        }
        super(CostlyAllocationGame, self).__init__(payoffs)

        
class AllocationGame(StageGame):
    """
    Three player game, first player must give to one of two others
    """
    def __init__(self, endowment = 0, cost = 1, benefit = 2):
        payoffs = {
            "give 1": (endowment-cost, benefit, 0),
            "give 2": (endowment-cost, 0, benefit),
        }
        super(AllocationGame, self).__init__(payoffs)

class AgentType(type):
    def __repr__(cls):
        return cls.__name__
    
class Agent(object):
    __metaclass__ = AgentType
    def __init__(self, genome, world_id=None):
        self.genome = deepcopy(genome)
        self.beta = self.genome['beta']
        self.world_id = world_id
        self.belief = dict()
        
    def utility(self, payoffs, agent_ids):
        return sum(self._utility(payoff,id) for payoff,id in itertools.izip(payoffs,agent_ids))

    def _utility(self, payoffs, agent_ids):
        raise NotImplementedError

    def observe_k(self, observations, k, tremble = 0):
        pass

    def __repr__(self):
        return type(self).__name__

    def __str__(self):
        return type(self).__name__
    
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

        # OBSOLETE?
        #Us = np.zeros(len(game.actions)) # Utilities for each action
        #for action in game.actions:
        #    action_index = game.action_lookup[action]
        #    Us[action_index] += deciding_agent.utility(game.payoffs[action], agents)
        #    print Us[action_index]

        Us = np.array([deciding_agent.utility(game.payoffs[action], agents)
                       for action in game.actions])
        try:
            return (1-tremble) * softmax(Us, deciding_agent.beta) + tremble * np.ones(len(Us))/len(Us)
        except ValueError:
            print "Us",Us
            print "beta",deciding_agent.beta
            raise
        
    def decide(self, game, agent_ids):
        # Tremble is always 0 for decisions since tremble happens in
        # the world, not the agent
        ps = self.decide_likelihood(self, game, agent_ids, tremble = 0)
        action_id = np.squeeze(np.where(np.random.multinomial(1,ps)))
        return game.actions[action_id]

class SelfishAgent(Agent):
    def __init__(self, genome, world_id=None):
        super(SelfishAgent, self).__init__(genome, world_id)
        
    def _utility(self, payoff, agent_id):
        if agent_id == self.world_id:
            return payoff
        else:
            return 0

class AltruisticAgent(Agent):

    def __init__(self, genome, world_id=None):
        super(AltruisticAgent, self).__init__(genome, world_id)

    def utility(self, payoffs, agent_ids):
        return sum(payoffs)
    
class RationalAgent(Agent):
    def __init__(self, genome, world_id=None):
        super(RationalAgent, self).__init__(genome, world_id)
        
        #NamedArray mapping agent_type to odds that an arbitrary agent is of that type
        self.pop_prior = self.genome['prior']
        
        self.uniform_likelihood = normalized(self.pop_prior*0+1)
        
        self.agentModels = {}
        self.likelihood = {}
        self.belief = {}
        

    def initialize_models(self, agent_ids):
        for agent_id in without(agent_ids, self.world_id, self.agentModels):
            self.agentModels[agent_id] = type(self)(self.genome, world_id = agent_id)

    def get_models(self, agent_id):
        models = []
        for agent_type in self.genome['agent_types']:
            if agent_type is type(self):
                try:
                    models.append(self.agentModels[agent_id])
                except KeyError:
                    self.agentModels[agent_id] = agent_type(self.genome, world_id = agent_id)
                    models.append(self.agentModels[agent_id])
            else:
                models.append(agent_type(self.genome, world_id = agent_id))
        return models
        
    def purge_models(self, ids):
        #must explicitly use .keys() below because mutation
        for id in (id for id in ids if id in set(self.agentModels.keys())): 
            del self.agentModels[id]
            del self.belief[id]
            del self.likelihood[id]
        for model in agentModels.itervalues():
            model.purge_models(ids)

    def use_default_dicts(self):
        """
        adding this function to __init__(self) makes all internal representations default-dicts
        this is guaranteed to not fail throughout and renders initializations redundant
        """
        #modelDict stores this agent's models of other agents
        #it maps agent ids to a model of that agent
        #will initialize any unseen agents to a new ReciprocalAgent
        class modelDict(dict):
            def __missing__(agentModels,agent_id):
                typeDict = {}
                for agent_type in self.genome['agent_types']:
                    if issubclass(agent_type,RationalAgent):
                        typeDict[agent_type] = agent_type(self.genome, world_id = agent_id)
                    else:
                        typeDict[agent_type] = self.genome['generic_models'][agent_type]
                        
                agentModels[agent_id] = type(self)(self.genome,world_id = agent_id)
                return agentModels[agent_id]
            
        self.agentModels = modelDict()

        #dict mapping a particular agent to the belief that they are of a given type
        #the belief is represented as a NamedArray
        self.belief = defaultdict(self.initialize_prior)

        #basically the same as belief
        self.likelihood = defaultdict(self.initialize_likelihood) 
        
    def initialize_prior(self):
        # This needs to initialize the data structure that is used for the online update
        return self.pop_prior
    
    def initialize_likelihood(self):
        return self.uniform_likelihood

    def utility(self, payoffs, agent_ids):
        return sum(itertools.imap(self.utility_try,payoffs,agent_ids))
    
    def utility_try(self,payoff,agent_id):
        #tries to return the defined utility
        #if it fails it initializes belief and tries again
        try:
            return self._utility(payoff,agent_id)
        except KeyError:
            self.belief[agent_id] = self.initialize_prior()
            return self._utility(payoff,agent_id)
            

    def sample_alpha(self, belief):
        # TODO: Decide what to do here? Should this be sampling and
        # then giving full weight? Or should it be weighting how much
        # we care? The weighted version is worse at punishing the bad
        # guys since it will still weight them a bit even when its
        # highly unlikely that they are reciprocal... Maybe this is
        # just a downside of being a nice person?0
        
        # return int(flip(belief))
        return belief

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

        if K < 0: return
                
        for observation in observations:
            game, participants, observers, action = observation


            deciding_agent = participants[0]

            # Can't have a belief about what I think about what I think. Beliefs about others are first order beliefs.
            #so if i'm considering myself, skip to the next round of the loop
            if deciding_agent == self.world_id: continue

            
            #if im not one of the observers this round, skip to the next round
            if self.world_id not in observers: continue


            #initialize models for all observers
            self.initialize_models(observers)

            #generate a list of models for every type of agent
            models = self.get_models(deciding_agent)

            action_index = game.action_lookup[action]
            
            #calculate the normalized likelihood for each type
            likelihood = [
                Agent.decide_likelihood(model, game, participants, tremble)[action_index]
                for model in models]

            try:
                self.likelihood[deciding_agent] *= likelihood
            except KeyError:
                self.likelihood[deciding_agent] = self.initialize_likelihood()
                self.likelihood[deciding_agent] *= likelihood

            # Update the priors after getting the likelihood estimate for each agent
            # TODO: Should the K-1 agents also have priors that get updated?
            
            self.likelihood[deciding_agent] = normalized(self.likelihood[deciding_agent])
            
            self.belief[deciding_agent] = (self.pop_prior*self.likelihood[deciding_agent]) / np.dot(self.pop_prior,self.likelihood[deciding_agent])


        # Observe the other person, when this code runs at K=0
        # nothing will happen because of the return at the top
        # of the function.
        for agent in self.agentModels:
            self.agentModels[agent].observe_k(observations, K-1, tremble)

            
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
                agent_counts = NamedArray([sum(np.array(order)==t) for t in range(n_agent_types)])
                counts = np.array(prior + agent_counts)

            #     # lnB = np.sum(gammaln(counts)) - gammaln(np.sum(counts))
            #     # pdf = np.exp( - lnB + np.sum((np.log(p.T) * (counts - 1)).T, 0) )
            #     # term = np.array(pdf) #FIXME: Shouldn't need this... named array issue
            
                term = sp.stats.dirichlet.pdf(p, counts)
                for a_id, agent_type in zip(self.likelihood.keys(), order):
                    # wrong not a dependence on p (theta) a dependence on alpha which is the prior
                    # belief = (self.likelihood[a_id][agent_type]*p[agent_type]) / np.dot(p, self.likelihood[a_id])

                    # still wrong since this just using the mean and not doing the full integration
                    belief = (self.likelihood[a_id][agent_type]*prior[agent_type]) / np.dot(prior, self.likelihood[a_id])

                    term *= belief

                like += term

            return -(np.log(like))

        out = constraint_min(ll, np.ones(n_agent_types)/n_agent_types)
        
        print out
        print NamedArray(out.x)
        # FIXME: Need to save these out to the pop_prior and then update the belief of all the agents by using the new prior when combining the likelihood and prior. 
        
        # self.pop_prior = {
            # ReciprocalAgent.__name__: out.x[0],
            # SelfishAgent.__name__ : 1-out.x[0]
        # }

class ReciprocalAgent(RationalAgent):
    def _utility(self, payoff, agent_id):
        if agent_id == self.world_id:
            return payoff
        else:
            alpha = self.sample_alpha(self.belief[agent_id][ReciprocalAgent])
            return alpha * payoff

class NiceReciprocalAgent(RationalAgent):
    def _utility(self, payoff, agent_id):
        if agent_id == self.world_id:
            return payoff
        else:
            alpha = self.sample_alpha(self.belief[agent_id][NiceReciprocalAgent]+
                                      self.belief[agent_id][AltruisticAgent])
            return alpha * payoff

class OpportunisticRA(RationalAgent):
    def _utility(self, payoff, agent_id):
        if agent_id == self.world_id:
            return payoff
        else:
            beliefs = []
            for agent_type in self.belief.fields:
                agent_type(self.genome,world_id = agent_id)._utility(1,self.world_id)
            alpha = self.sample_alpha(self.belief[agent_id][NiceReciprocalAgent]+
                                      self.belief[agent_id][AltruisticAgent])
            return alpha * payoff

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
        ReciprocalAgent,
        AltruisticAgent
    ]
    return {
        'N_agents':2,
        'games': BinaryDictator(0, 1, 2), 
        'stop_condition': constant_stop_condition(10),
        'agent_types' : agent_types,
        'beta': 3,
        'moran_beta': .1,
        'RA_prior': .8,
        'prior_precision': 0, # setting this to 0 turns off updating the prior
        'p_tremble': 0.0,
        'RA_K': 1,
        'agent_types_world': agent_types
    }

def prior_generator(agent_types,RA_prior=False):
    agent_types = tuple(agent_types)
                                           
    if ReciprocalAgent in agent_types:
        uniform = (1.0-RA_prior)/(len(agent_types)-1)
    else:
        uniform = 1.0/len(agent_types)

    if not RA_prior:
        RA_prior = uniform
        
    return namedArrayConstructor(tuple(agent_types))(
        [
            uniform if agentType is not ReciprocalAgent
            else RA_prior for agentType in agent_types
        ])

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
        })
        
    return genomes
    

class World(object):
    # TODO: spatial or interaction probabilities
    
    def __init__(self, params, genomes):
        self.agents = []
        self.counter = itertools.count()
        self.id_to_agent = {}
        
        self.add_agents(genomes)

        self.population = len(self.agents)
        
        self.game = params['games']
        self.stop_condition = params['stop_condition']
        self.params = params

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
        
        die = np.random.choice(range(len(self.agents)), int(p*len(self.agents)), replace=False)
        random = np.random.choice(range(len(self.agents)), int(mu*len(self.agents)), replace=False)
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
        # take in a sampling function
        fitness = np.zeros(len(self.agents))
        history = []
        
        #Get all matchups (sets of players)
        #players are represented by their position in the list of agents
        matchups = list(itertools.combinations(xrange(self.population), self.game.N_players))
        np.random.shuffle(matchups)

        for players in matchups:
            players= list(players)
            rounds = 0
                      
            while True:
                rounds += 1
                
                observations = []
                payoff = np.zeros(len(self.agents))


                #We want every possible significant matchup to happen
                #everyone gets to be a deciding agents exactly once
                
                #We assume that:
                #the order of non-deciding players doesn't matter
                #games only have a single deciding player
                player_orderings = [players[n:n+1]+players[:n]+players[n+1:]
                                    for n in range(len(players))]

                for player_order in player_orderings:
                    
                    agents, agent_ids = [], []
                    for nth in player_order:
                        agents.append(self.agents[nth])
                        agent_ids.append(self.agents[nth].world_id)

                    deciders = agents[:1]
                    
                    # Intention -> Trembling Hand -> Action
                    intentions = [decider.decide(self.game, agent_ids)
                                  for decider in deciders]

                    # translate intentions into actions applying tremble
                    actions = [np.random.choice(self.game.actions)
                               if flip(params['p_tremble']) else intention
                               for intention in intentions]
                    
                    #accumulate the payoff
                    for action in actions:
                        payoff[list(player_order)] += self.game.payoffs[action]

                    # Determine who gets to observe this action. 
                    
                    # Reveal observations to update the belief state. This
                    # is where we can include more agents to increase the
                    # amount of observability
                    observer_ids = players

                    # Record observations
                    observation = (self.game, agent_ids, observer_ids, actions[0])
                    observations.append(observation)

                # Update fitness
                fitness += payoff

                # All observers see who observed the action. 
                # for o in observations:
                    # Iterate over all of the observers

                for agent in self.agents:
                    agent.observe_k(observations, self.params['RA_K'], self.params['p_tremble'])

                history.append({
                    'round': rounds,
                    'players': tuple(self.agents[player] for player in players),
                    'actions': tuple(observation[2][0] for observation in observations),
                    'payoff': payoff,
                    'belief': tuple(copy(self.agents[player].belief) for player in players)
                })

                if self.stop_condition(rounds): break
                
        return fitness, history
                
discount_stop_condition = lambda x: lambda n: not flip(x)
constant_stop_condition = lambda x: lambda n: n >= x

# adding multistep games and games that depend on the outcome of previosu games (e.g., ultimatum game)
# TODO: This currently expects a single game instance. TODO: Make this accept a list of games or a game generator function. 


def forgiveness_experiment(path = 'sims/forgiveness.pkl', overwrite = False):
    """
    When two reciprocal agents interact, how likely are they to figure out that they are both reciprocal agent types?
    This will depend on the RA_prior. 

    Compare with something like TFT which if it gets off on the wrong foot will never recover. There are forgiving versions of TFT but they are not context sensitive. Experiments here should explore how the ReciprocalAgent is a more robust cooperator since it can reason about types. 

    TODO: This could be an interesting place to explore p_tremble, and show that agents can recover. 
    """
    print 'Running Forgiveness Experiment'
    if os.path.isfile(path) and not overwrite: 
        print path, 'exists. Delete or set the overwrite flag.'
        return
    
    params = default_params()
    params['N_agents'] = 2
    params['agent_types_world'] = [ReciprocalAgent]
    params['agent_types_model'] = [ReciprocalAgent,SelfishAgent]
    N_round = 10
    params['stop_condition'] = constant_stop_condition(N_round)
    data = []
    N_runs = 500
    for RA_prior in np.linspace(0.5, 0.95, 4):
        params['RA_prior'] = RA_prior
        print 'running prior', RA_prior

        for r_id in range(N_runs):
            np.random.seed(r_id) # Increment a new seed for each run
            w = World(params, generate_random_genomes(params['N_agents'],
                                                      params['agent_types_world'],
                                                      params['agent_types_model'],
                                                      params['RA_prior'],
                                                      params['prior_precision'],
                                                      params['beta']))
            fitness, history = w.run()
            for nround in range(len(history)):
                avg_beliefs = np.mean([history[nround]['belief'][0][w.agents[1].world_id][ReciprocalAgent],
                                       history[nround]['belief'][1][w.agents[0].world_id][ReciprocalAgent]])
                #print avg_beliefs.dtype
                data.append({
                    'RA_prior': RA_prior,
                    'avg_beliefs': list(avg_beliefs),
                    'round': nround+1
                })

            data.append({
                'RA_prior': RA_prior,
                'avg_beliefs': RA_prior,
                'round': 0
            })


    df = pd.DataFrame(data)
    df.to_pickle(path)

def forgiveness_plot(in_path = 'sims/forgiveness.pkl', out_path='writing/evol_utility/figures/forgiveness.pdf'):
    df = pd.read_pickle(in_path)

    sns.factorplot(x='round', y='avg_beliefs', hue='RA_prior', data=df)
    sns.despine()
    plt.ylim([0,1.05])
    plt.ylabel('P(Other is reciprocal | Round)'); plt.xlabel('Round')
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

    
    
def protection_experiment(path = 'sims/protection.pkl', overwrite = False):
    """
    If a ReciprocalAgent and a Selfish agent are paired together. How quickly will the
    ReicprocalAgent detect it. Look at how fast this is learned as a function of the prior. 
    """
    
    print 'Running Protection Experiment'
    if os.path.isfile(path) and not overwrite: 
        print path, 'exists. Delete or set the overwrite flag.'
        return

    params = default_params()
    params['agent_types_world'] = agent_types =  [ReciprocalAgent, SelfishAgent]
        
    params['stop_condition'] = constant_stop_condition(10)
    data = []
    N_runs = 500
    for RA_prior in np.linspace(0.5, .95, 4):
        params['RA_prior'] = RA_prior
        print 'running prior', RA_prior

        for r_id in range(N_runs):
            np.random.seed(r_id)
            w = World(params, [
                {'type': ReciprocalAgent,
                 'RA_prior': RA_prior,
                 'agent_types':[ReciprocalAgent,SelfishAgent],
                 'agent_types_world':[ReciprocalAgent,SelfishAgent],
                 'prior':prior_generator(agent_types,RA_prior),
                 'agent_types_model':[ReciprocalAgent,SelfishAgent],
                 'prior_precision': params['prior_precision'],
                 'beta': params['beta']
                },
                {'type': SelfishAgent, 'beta': params['beta']},
            ])
            
            fitness, history = w.run()
            for h in history:
                data.append({
                    'round': h['round'],
                    'RA_prior': RA_prior,
                    'belief': h['belief'][0][1][ReciprocalAgent],
                })

            data.append({
                'round': 0,
                'RA_prior': RA_prior,
                'belief': RA_prior,
            })

    df = pd.DataFrame(data)
    df.to_pickle(path)

def protection_plot(in_path = 'sims/protection.pkl',
                    out_path='writing/evol_utility/figures/protection.pdf'):
    df = pd.read_pickle(in_path)

    sns.factorplot('round', 'belief', hue='RA_prior', data=df, ci=68)
    sns.despine()
    plt.ylim([0,1])
    plt.ylabel('P(Other is reciprocal | Interactions)'); plt.xlabel('Round #')
    plt.tight_layout()
    plt.savefig(out_path); plt.close()
    
def fitness_rounds_experiment(path = 'sims/fitness_rounds.pkl', overwrite = False):
    """
    Repetition supports cooperation. Look at how the number of rounds each dyad plays together and 
    the average fitness of the difference agent types. 
    """
    
    print 'Running Fitness Rounds Experiment'
    if os.path.isfile(path) and not overwrite: 
        print path, 'exists. Delete or set the overwrite flag.'
        return


    params = default_params()
    params.update({
        "N_agents":50,
        "RA_K": 1
    })
    N_runs = 10
    data = []
    for rounds in np.linspace(1, 8, 8, dtype=int):
        print "Round:",rounds
        for r_id in range(N_runs):
            np.random.seed(r_id)

            params['stop_condition'] = constant_stop_condition(rounds)
                  
            w = World(params, generate_random_genomes(**params))
            fitness, history = w.run()
            
            genome_fitness = Counter()
            genome_count = Counter()

            for a_id, a in enumerate(w.agents):
                genome_fitness[type(a)] += fitness[a_id]
                genome_count[type(a)] += 1

            average_fitness = {a:genome_fitness[a]/genome_count[a] for a in genome_fitness}

            moran_fitness = softmax_utility(average_fitness, params['moran_beta'])

            for a in moran_fitness:
                data.append({
                    'rounds': rounds,
                    'genome': a,
                    'fitness': moran_fitness[a]
                })

        df = pd.DataFrame(data)
        df.to_pickle(path)

def fitness_rounds_plot(in_path = 'sims/fitness_rounds.pkl', out_path='writing/evol_utility/figures/fitness_rounds.pdf'):

    df = pd.read_pickle(in_path)
    sns.factorplot('rounds', 'fitness', hue='genome', data=df,)
    sns.despine()
    plt.ylim([0,1.05])
    plt.ylabel('Fitness ratio'); plt.xlabel('# of repetitions')
    plt.tight_layout()
    plt.savefig(out_path); plt.close()






params = default_params()

    
typesClassic = agent_types = (ReciprocalAgent,SelfishAgent)
typesAll = (ReciprocalAgent,AltruisticAgent,SelfishAgent)








params['stop_condition'] = constant_stop_condition(10)
params['p_tremble'] = 0
params['RA_prior'] = 0.8
params['prior_precision'] = 0
prior = prior_generator(agent_types,params['RA_prior'])

w = World(params, [
    {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'prior_precision': params['prior_precision'], 'beta': params['beta'],'prior' : prior, 'agent_types': agent_types},
    {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'prior_precision': params['prior_precision'], 'beta': params['beta'],'prior' : prior, 'agent_types': agent_types},
    {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'prior_precision': params['prior_precision'], 'beta': params['beta'],'prior' : prior, 'agent_types': agent_types},
    {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'prior_precision': params['prior_precision'], 'beta': params['beta'],'prior' : prior, 'agent_types': agent_types},
    {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'prior_precision': params['prior_precision'], 'beta': params['beta'],'prior' : prior, 'agent_types': agent_types},
    {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'prior_precision': params['prior_precision'], 'beta': params['beta'],'prior' : prior, 'agent_types': agent_types},
    {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'prior_precision': params['prior_precision'], 'beta': params['beta'],'prior' : prior, 'agent_types': agent_types},
    {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'prior_precision': params['prior_precision'], 'beta': params['beta'],'prior' : prior, 'agent_types': agent_types},
    
])


observations= [
    #(w.game, [0, 1], range(len(w.agents)), 'give'),
    (w.game, [0, 1], range(len(w.agents)), 'keep'),
    #(w.game, [0, 1], range(len(w.agents)), 'keep'),
    #(w.game, [0, 1], range(len(w.agents)), 'keep'),
    #(w.game, [0, 1], range(len(w.agents)), 'keep'),
    #(w.game, [0, 1], range(len(w.agents)), 'keep'),
    #(w.game, [0, 1], range(len(w.agents)), 'keep'),
    #(w.game, [0, 1], range(len(w.agents)), 'keep'),
    # (w.game, [0, 1], range(len(w.agents)), 'keep'),
    # (w.game, [0, 1], range(len(w.agents)), 'keep'),
    # (w.game, [0, 1], range(len(w.agents)), 'keep'),
    #(w.game, [1, 2], range(len(w.agents)), 'keep'),
    #(w.game, [1, 2], range(len(w.agents)), 'keep'),
    # (w.game, [0, 1], range(len(w.agents)), 'keep'),
    # (w.game, [1, 0], range(len(w.agents)), 'keep'),
    # (w.game, [0, 1], range(len(w.agents)), 'keep'),
    # (w.game, [1, 0], range(len(w.agents)), 'keep'),
    # (w.game, [1, 2], range(len(w.agents)), 'give'),
    #(w.game, [2, 1], range(len(w.agents)), 'keep'),
    #(w.game, [2, 1], range(len(w.agents)), 'keep'),
    #(w.game, [4, 1], range(len(w.agents)), 'keep'),
    #(w.game, [5, 1], range(len(w.agents)), 'keep'),
    #(w.game, [6, 1], range(len(w.agents)), 'keep'),
    #(w.game, [7, 1], range(len(w.agents)), 'keep'),

    # (w.game, [2, 1], range(len(w.agents)), 'give'),

    
    # (w.game, [0, 1], range(len(w.agents)), 'keep'),
    #(w.game, [2, 3], range(len(w.agents)), 'give'),
    # (AllocationGame(), [2, 0, 1], range(len(w.agents)), 'give 1'),
]


K = 1
i = 3


print "prior: %s" % prior
w.agents[i].belief[0] = w.agents[i].pop_prior
print "belief(type(0)=RA|Nothing) = ", w.agents[i].belief[0][ReciprocalAgent]

w.agents[i].observe_k(observations[:], 1)
print "belief(type(0)=RA|Obs) = ", w.agents[i].belief[0][ReciprocalAgent]
print "likelihood(0)",w.agents[i].likelihood[0][ReciprocalAgent]

w.agents[i].update_prior()
for aid,l in w.agents[i].likelihood.iteritems():
    print aid,l


# for i in range(len(w.agents)):
#     w.agents[i].observe_k(observations, K)
#     print i
#     pprint(w.agents[i].belief)
    # for k in w.agents[i].agents:
    #     print k,
    #     pprint( w.agents[i].agents[k].belief)



#forgiveness_experiment(overwrite=True)
#forgiveness_plot() 

#protection_experiment(overwrite=True)
#protection_plot()

fitness_rounds_experiment(overwrite=True)
fitness_rounds_plot()


"""
added 

class
RationalAgent(Agent)
ReciprocalAgent(RationalAgent)

methods for RationalAgent

initialize_models
get_models
initialize_likelihood
use_default_dicts

attributes

uniform_likelihood
"""
