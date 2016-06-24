from __future__ import division
import scipy as sp
import numpy as np
import pandas as pd
import networkx as nx
import itertools
import random
from collections import Counter,defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import math
# import dirichlet
from utils import softmax, sample_softmax, softmax_utility, flip
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

def normalized(array):
    return array/np.sum(array)
assert np.sum(normalized(np.array([.2,.3]))) ==1

def memoized(f):
    """ Memoization decorator for a function taking one or more arguments. """
    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret
            
    return memodict().__getitem__

def namedArrayConstructor(fields):
    """
    this takes a canonical sequence of hashable objects and returns a class of arrays
    where each of the elements in the first dimension of the array can be indexed 
    using the object in the corresponding position in the canonical sequence

    these classes are closed on themselves, operating two arrays with the same seed sequence 
    will produce an array that can be indexed using the same sequence.
   
    Note: This constructor should always be memoized to avoid creation of functionally identical classes
    """

    assert len(fields) == len(set(fields))
    
    reference = dict(map(reversed,enumerate(fields)))
    class NamedArray(np.ndarray):
        """
        these arrays function exactly the same as normal np.arrays
        except that they can also be indexed using the elements provided to the constructor
        in short, if we assume that the sequence 'fields' contains the element 'field'

        namedArray = namedArrayConstructor(fields)

        array = namedArray(sequence)

        array[fields.index(field)] == array[field]
        """
        def __new__(self, seq):
            return np.asarray(seq).view(self)
        
        def __getitem__(self,*keys):
            try:
                return super(NamedArray,self).__getitem__(*keys)
            except IndexError:
                return super(NamedArray,self).__getitem__(reference[keys[0]])

        def __setitem__(self,*keys,**vals):
            try:
                super(NamedArray,self).__setitem__(*keys,**vals)
            except IndexError:
                keys = (reference[keys[0]],)+keys[1:]
                super(NamedArray,self).__setitem__(*keys,**vals)
    return NamedArray

namedArrayConstructor = memoized(namedArrayConstructor)


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


class Agent(object):
    def __init__(self, genome, world_id=None):
        self.genome = deepcopy(genome)
        self.beta = self.genome['beta']
        self.world_id = world_id
        self.belief = dict()

    def utility(self, payoff, agent):
        raise NotImplementedError

    def observe_k(self, observations, k, tremble = 0):
        pass

    def __repr__(self):
        return type(self).__name__

    def __str__(self):
        return type(self).__name__

    def __hash__(self):
        return hash(self.__repr__())
    
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

        Us = np.zeros(len(game.actions)) # Utilities for each action
        for action in game.actions:
            action_index = game.action_lookup[action]
            for payoff, agent in zip(game.payoffs[action], agents):
                Us[action_index] += deciding_agent.utility(payoff, agent)
        
        return (1-tremble) * softmax(Us, deciding_agent.beta) + tremble * np.ones(len(Us))/len(Us)

    def decide(self, game, agent_ids):
        # Tremble is always 0 for decisions since tremble happens in
        # the world, not the agent
        ps = self.decide_likelihood(self, game, agent_ids, tremble = 0)
        action_id = np.squeeze(np.where(np.random.multinomial(1,ps)))
        return game.actions[action_id]


class SelfishAgent(Agent):
    def __init__(self, genome, world_id=None):
        super(SelfishAgent, self).__init__(genome, world_id)

    def utility(self, payoff, agent_id):
        if agent_id == self.world_id:
            return payoff
        else:
            return 0

class AltruisticAgent(Agent):
    def __init__(self, genome, world_id=None):
        super(AltruisticAgent, self).__init__(genome, world_id)

    def utility(self, payoff, agent_id):
        return payoff

        
class ReciprocalAgent(Agent):
    def __init__(self, genome, world_id=None):
        super(ReciprocalAgent, self).__init__(genome, world_id)

        #NamedArray mapping agent_type to odds that an arbitrary agent is of that type
        self.pop_prior = self.genome['prior']

        #modelDict stores this agent's models of other agents
        #it maps agent ids to a model of that agent
        #will initialize any unseen agents to a new ReciprocalAgent
        class modelDict(dict):
            def __missing__(agentModels,agent_id):
                agentModels[agent_id] = ReciprocalAgent(self.genome,world_id = agent_id)
                return agentModels[agent_id]
            
        self.agentModels = modelDict()

        #dict mapping a particular agent to the belief that they are of a given type
        #the belief is represented as a NamedArray
        self.belief = defaultdict(self.initialize_prior)

        #basically the same as belief
        self.initial_likelihood = initial_likelihood = lambda: namedArrayConstructor(genome['agent_types'])(np.ones(len(genome['agent_types'])))
        self.likelihood = defaultdict(initial_likelihood)
        
    def initialize_prior(self):
        # This needs to initialize the data structure that is used for the online update
        return self.pop_prior
    
    def utility(self, payoff, agent_id):
        if agent_id == self.world_id:
            return payoff
        else:
            alpha = self.sample_alpha(self.belief[agent_id][ReciprocalAgent])
            return alpha * payoff

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

            action_index = game.action_lookup[action]

            #make a model for every agent type
            models = [
                agent_type(self.genome, deciding_agent)
                if agent_type is not ReciprocalAgent
                else self.agentModels[deciding_agent]
                for agent_type in self.genome['agent_types']]
            
            #calculate the normalized likelihood for each type
            likelihood = [
                Agent.decide_likelihood(model, game, participants, tremble)[action_index]
                for model in models]

            self.likelihood[deciding_agent] *= likelihood

            # Update the priors after getting the likelihood estimate for each agent
            # TODO: Should the K-1 agents also have priors that get updated?
            
            self.likelihood[deciding_agent] = normalized(self.likelihood[deciding_agent])
            
            marginal = np.dot(self.pop_prior,self.likelihood[deciding_agent])
            self.belief[deciding_agent] = (self.pop_prior*self.likelihood[deciding_agent])/marginal

        # Observe the other person, when this code runs at K=0
        # nothing will happen because of the return at the top
        # of the function.
        for agent in self.agentModels:
            self.agentModels[agent].observe_k(observations, K-1, tremble)

            
        if K>0:
            self.update_prior()

    def update_prior(self):
        if self.genome['RA_prior_precision'] == 0: return
        
        D = list()
        length = len(self.genome['agent_types'])
        namedArray = namedArrayConstructor(self.genome['agent_types'])

        for agent, likelihood in self.likelihood.iteritems():
            p = likelihood[ReciprocalAgent]/sum(likelihood)
            w = abs(p - (1 - p))
            uniform = (1.0-p)/(length-1)
            array = namedArray(np.ones(length))*uniform
            array[ReciprocalAgent] = p
            
            D.append(array*w)

        if not len(D): return # No observations

        D = np.array(D)
        
        prior = self.genome['prior'] * self.genome['RA_prior_precision']

        def ll(p):
            # Beta Prior on Mean
            logprior = sum((prior-1) * np.log([p[0], 1-p[0]]))

            # Binomial likelihood
            RA, S = D.sum(axis=0)
            return -(
                RA * np.log(p[0]) + S * np.log(1-p[0]) 
                + logprior)
        
        out = sp.optimize.minimize(ll, D.sum(axis=0)[0], bounds = [(.001, .999)])
        
        self.pop_prior = {
            ReciprocalAgent.__name__: out.x[0],
            SelfishAgent.__name__ : 1-out.x[0]
        }

                                                                                                                                            
def generate_random_genomes(N, agent_types_world, agent_types, RA_prior, RA_prior_precision, beta):
    genomes = []
    agent_types = tuple(agent_types)
                                                    
    if ReciprocalAgent in agent_types:
        uniform = (1.0-RA_prior)/(len(agent_types)-1)
    else:
        uniform = 1.0/len(agent_types)

    prior = namedArrayConstructor(agent_types)(
        [uniform if agentType is not ReciprocalAgent else RA_prior for agentType in agent_types]
        )

    for _ in range(N):
        genomes.append({
            'type': np.random.choice(agent_types_world),
            'agent_types': agent_types,
            'prior': prior,
            'RA_prior': RA_prior,
            'RA_prior_precision': RA_prior_precision,
            'beta': beta,
        })

    return genomes

class World(object):
    # TODO: spatial or interaction probabilities
    
    def __init__(self, params, genomes):
        self.agents = []
        self.id_to_agent = {}
        # Generate some Genomes and instantiate them as agents
        for world_id, genome in enumerate(genomes):
            self.agents.append(
                genome['type'](genome, world_id)
            )
            self.id_to_agent[world_id] = self.agents[-1]
        
        self.game = params['games']
        self.stop_condition = params['stop_condition']
        self.params = params
        
    def evolve(self, fitness, p=1, beta=1, mu=0.05):
        # FIXME: Need to increment the id's. Can't just make new
        # agents, otherwise new agents will be treated as old agents
        # if they share the same ID
        assert 0 # BROKEN (See above)
        
        die = np.random.choice(range(len(self.agents)), int(p*len(self.agents)), replace=False)
        random = np.random.choice(range(len(self.agents)), int(mu*len(self.agents)), replace=False)
        for a in die:
            copy_id = sample_softmax(fitness, beta)
            self.agents[a] = self.agents[copy_id].__class__(self.agents[copy_id].genome)

        new_genomes = generate_random_genomes(len(random), self.params['agent_types'], self.params['RA_prior'], self.params['beta'])
        for a, ng in zip(random, new_genomes):
            self.agents[a] = ng['type'](ng)

    def run(self):
        # take in a sampling function
        fitness = np.zeros(len(self.agents))
        history = []

        # Get all pairs
        pairs = list(itertools.combinations(range(len(self.agents)), 2))
        np.random.shuffle(pairs)

        for pair in pairs:
            rounds = 0
            while True:
                rounds += 1
                
                observations = []
                payoff = np.zeros(len(self.agents))
                # Have both players play both roles in the dictator game
                for p0, p1 in [[pair[0], pair[1]], [pair[1], pair[0]]]:
                    agents = [self.agents[p0], self.agents[p1]]
                    agent_ids = [self.agents[p0].world_id, self.agents[p1].world_id]

                    # Intention -> Trembling Hand -> Action
                    intentions = agents[0].decide(self.game, agent_ids)
                    actions = copy(intentions)

                    #does this assume that everyone has the same actions?
                    #does everyone tremble independently?
                    #are these joint actions?
                    
                    for i in range(len(intentions)):
                        if flip(params['p_tremble']):
                            actions = np.random.choice(self.game.actions)
                            
                    intentions = intentions
                    actions = actions

                    payoff[[p0, p1]] += self.game.payoffs[actions]

                    # Determine who gets to observe this action. 
                    
                    # Reveal observations to update the belief state. This
                    # is where we can include more agents to increase the
                    # amount of observability
                    observer_ids = pair
                    observations.append((self.game, agent_ids, observer_ids, actions))

                # Update fitness
                fitness += payoff

                # All observers see who observed the action. 
                # for o in observations:
                    # Iterate over all of the observers
                for a in self.agents:
                    a.observe_k(observations, self.params['RA_K'], self.params['p_tremble'])
                    
                history.append({
                    'round': rounds,
                    'players': (self.agents[pair[0]].world_id, self.agents[pair[1]].world_id),
                    'actions': (observations[0][2][0], observations[1][2][0]),
                    'payoff': payoff,
                    'belief': (copy(self.agents[0].belief), copy(self.agents[1].belief))
                })

                if self.stop_condition(rounds): break
                
        return fitness, history
                
discount_stop_condition = lambda x: lambda n: not flip(x)
constant_stop_condition = lambda x: lambda n: n >= x

# adding multistep games and games that depend on the outcome of previosu games (e.g., ultimatum game)
# TODO: This currently expects a single game instance. TODO: Make this accept a list of games or a game generator function. 

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

    `RA_prior_precision`: This is how confident the agent is in its prior (its like a hyperprior). If it is 0, there is no uncertainty and the prior never changes. If this is non-zero, then `RA_prior` is just the mean of a distribution. The agent will learn the `RA_prior` as it interacts with more and more agents. 

    `p_tremble`: probability of noise in between forming a decision and making an action. 

    `RA_K`: is the number of theory-of-mind recursions to carry out. When RA_K is 0 the agent just tries to infer the type directly, when it is 1, you first infer what each agent knows and then infer what you know based on those agents and so on. 

    """
    return {
        'games': BinaryDictator(0, 1, 2), 
        'stop_condition': constant_stop_condition(10),
        'agent_types' : [
            SelfishAgent,
            ReciprocalAgent,
            AltruisticAgent
        ],
        'beta': 3,
        'moran_beta': .1,
        'RA_prior': .8,
        'RA_prior_precision': 0, # setting this to 0 turns off updating the prior
        'p_tremble': 0.0,
        'RA_K': 1,
    }

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
                                                      params['RA_prior_precision'],
                                                      params['beta']))
            fitness, history = w.run()
            for nround in range(len(history)):
                avg_beliefs = np.mean([history[nround]['belief'][0][w.agents[1].world_id][ReciprocalAgent],
                                       history[nround]['belief'][1][w.agents[0].world_id][ReciprocalAgent]])
                data.append({
                    'RA_prior': RA_prior,
                    'avg_beliefs': avg_beliefs,
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
    If a ReciprocalAgent and a Selfish agent are paired together. How quickly will the ReicprocalAgent detect it. Look at how fast this is learned as a function of the prior. 
    """
    
    print 'Running Protection Experiment'
    if os.path.isfile(path) and not overwrite: 
        print path, 'exists. Delete or set the overwrite flag.'
        return

    params = default_params()
    params['agent_types_world'] = agent_types =  [ReciprocalAgent, SelfishAgent]
    if ReciprocalAgent in agent_types:
        uniform = (1.0-RA_prior)/(len(agent_types)-1)
    else:
        uniform = 1.0/len(agent_types)
        
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
                 'RA_prior': params['RA_prior'],
                 'agent_types_world':[ReciprocalAgent,SelfishAgent],
                 'prior':namedArrayConstructor(agent_types)(
                     [uniform if agentType is not ReciprocalAgent else RA_prior for agentType in agent_types]
                 ),
                 'agent_types_model':[RecpirocalAgent,SelfishAgent],
                 'RA_prior_precision': params['RA_prior_precision'],
                 'beta': params['beta']
                },
                {'type': SelfishAgent, 'beta': params['beta']},
            ])
            
            fitness, history = w.run()
            for h in history:
                data.append({
                    'round': h['round'],
                    'RA_prior': RA_prior,
                    'belief': h['belief'][0][h['players'][1]],
                })

            data.append({
                'round': 0,
                'RA_prior': RA_prior,
                'belief': RA_prior,
            })

    df = pd.DataFrame(data)
    df.to_pickle(path)

def protection_plot(in_path = 'sims/protection.pkl', out_path='writing/evol_utility/figures/protection.pdf'):
    df = pd.read_pickle(in_path)

    sns.factorplot('round', 'belief', hue='RA_prior', data=df, ci=68)
    sns.despine()
    plt.ylim([0,1])
    plt.ylabel('P(Other is reciprocal | Interactions)'); plt.xlabel('Round #')
    plt.tight_layout()
    plt.savefig(out_path); plt.close()
    
def fitness_rounds_experiment(path = 'sims/fitness_rounds.pkl', overwrite = False):
    """
    Repetition supports cooperation. Look at how the number of rounds each dyad plays together and the average fitness of the difference agent types. 
    """
    
    print 'Running Fitness Rounds Experiment'
    if os.path.isfile(path) and not overwrite: 
        print path, 'exists. Delete or set the overwrite flag.'
        return


    params = default_params()
    params['N_agents'] = 50
    params['RA_K'] = 1
    params['RA_prior'] = .8
    N_runs = 10
    data = []

    for rounds in np.linspace(1, 8, 8, dtype=int):
        print rounds
        for r_id in range(N_runs):
            np.random.seed(r_id)

            params['stop_condition'] = constant_stop_condition(rounds)
            
            w = World(params, generate_random_genomes(params['N_agents'], params['agent_types'],params['agent_types'], params['RA_prior'], params['beta']))
            fitness, history = w.run()

            genome_fitness = Counter()
            genome_count = Counter()

            for a_id, a in enumerate(w.agents):
                genome_fitness[a.__class__.__name__] += fitness[a_id]
                genome_count[a.__class__.__name__] += 1

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

    
typesClassic = agent_types = [ReciprocalAgent,SelfishAgent]
typesAll = [ReciprocalAgent,AltruisticAgent,SelfishAgent]

agent_types = tuple(agent_types)
                                                    
if ReciprocalAgent in agent_types:
    uniform = (1.0-params['RA_prior'])/(len(agent_types)-1)
else:
    uniform = 1.0/len(agent_types)
    
prior = namedArrayConstructor(agent_types)(
    [uniform if agentType is not ReciprocalAgent else params['RA_prior'] for agentType in agent_types]
)




params['stop_condition'] = constant_stop_condition(10)
params['p_tremble'] = 0
params['RA_prior'] = 0.8
params['RA_prior_precision'] = 0

w = World(params, [
    {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'RA_prior_precision': params['RA_prior_precision'], 'beta': params['beta'],'prior' : prior, 'agent_types': agent_types},
    {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'RA_prior_precision': params['RA_prior_precision'], 'beta': params['beta'],'prior' : prior, 'agent_types': agent_types},
    {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'RA_prior_precision': params['RA_prior_precision'], 'beta': params['beta'],'prior' : prior, 'agent_types': agent_types},
    {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'RA_prior_precision': params['RA_prior_precision'], 'beta': params['beta'],'prior' : prior, 'agent_types': agent_types},
    
])


observations= [
    (w.game, [0, 1], range(len(w.agents)), 'keep'),
    (w.game, [0, 1], range(len(w.agents)), 'keep'),
    (w.game, [0, 1], range(len(w.agents)), 'keep'),
    (w.game, [0, 1], range(len(w.agents)), 'keep'),
    (w.game, [1, 2], range(len(w.agents)), 'give'),
    (w.game, [1, 2], range(len(w.agents)), 'give'),
    # (w.game, [0, 1], range(len(w.agents)), 'keep'),
    # (w.game, [1, 0], range(len(w.agents)), 'keep'),
    # (w.game, [0, 1], range(len(w.agents)), 'keep'),
    # (w.game, [1, 0], range(len(w.agents)), 'keep'),
    #(w.game, [1, 2], range(len(w.agents)), 'give'),
    #(w.game, [2, 1], range(len(w.agents)), 'give'),
    # (w.game, [1, 2], range(len(w.agents)), 'give'),

    
    # (w.game, [0, 1], range(len(w.agents)), 'keep'),
    #(w.game, [2, 3], range(len(w.agents)), 'give'),
    # (AllocationGame(), [2, 0, 1], range(len(w.agents)), 'give 1'),
]


K = 1
i = 3

print "prior: %s" % prior
w.agents[i].belief[0]
print "belief(type(0)=RA|Nothing) = ", w.agents[i].belief[0][ReciprocalAgent]
w.agents[i].observe_k(observations[:], 1)
print "belief(type(0)=RA|Obs) = ", w.agents[i].belief[0][ReciprocalAgent]
print "likelihood(0)",w.agents[i].likelihood[0][ReciprocalAgent]
print




# for i in range(len(w.agents)):
#     w.agents[i].observe_k(observations, K)
#     print i
#     pprint(w.agents[i].belief)
    # for k in w.agents[i].agents:
    #     print k,
    #     pprint( w.agents[i].agents[k].belief)



#forgiveness_experiment(overwrite=True)
#forgiveness_plot() 

# protection_experiment(overwrite=True)
# protection_plot()

# fitness_rounds_experiment(overwrite=True)
# fitness_rounds_plot()

