from __future__ import division
import scipy as sp
import numpy as np
import pandas as pd
import networkx as nx
import itertools
import random
from collections import Counter
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

    @staticmethod
    def decide_likelihood(deciding_agent, game, agent_ids, tremble = 0):
        """
        !!!
        decide likelihood of what? other player's actions?
        am I implicitly inferring agent "types" here?
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
        for a in game.actions:
            a_id = game.action_lookup[a]
            for payoff, agent_id in zip(game.payoffs[a], agent_ids):
                Us[a_id] += deciding_agent.utility(payoff, agent_id)
        
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


        
class ReciprocalAgent(Agent):
    def __init__(self, genome, world_id=None):
        super(ReciprocalAgent, self).__init__(genome, world_id)
        self.pop_prior = {
            ReciprocalAgent.__name__: self.genome['RA_prior'],
            SelfishAgent.__name__ : 1-self.genome['RA_prior']
        }
        self.belief = dict()
        self.agents = dict()
        self.likelihood = dict()
        
    def initialize_prior(self):
        # This needs to initialize the data structure that is used for the online update
        return self.pop_prior[ReciprocalAgent.__name__]
    
    def utility(self, payoff, agent_id):
        if agent_id == self.world_id:
            return payoff
        else:
            if agent_id not in self.belief:
                self.belief[agent_id] = self.initialize_prior()

            alpha = self.sample_alpha(self.belief[agent_id])
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

            game, agent_ids, observer_ids, action = observation
        
            # Do recursive theory of mind on all of the agents 
            # for pair in list(itertools.permutations(observer_ids,K+1)):
            for a_id in observer_ids:
                # Can't have a belief about what I think about what I think. Beliefs about others are first order beliefs.
                if a_id == self.world_id: continue

                # Initialize the level-2 agents
                if a_id not in self.agents:
                    self.agents[a_id] = ReciprocalAgent(self.genome, world_id = a_id)
                    for o_id in observer_ids:
                        self.agents[a_id].belief[o_id] = self.agents[a_id].initialize_prior()

                
        # This agent is the one that made the decision so you don't
        # learn anything about that agent
        for observation in observations:
            game, agent_ids, observer_ids, action = observation

            # Can't have a belief about what I think about what I think. Beliefs about others are first order beliefs.
            if agent_ids[0] == self.world_id: continue
            if self.world_id not in observer_ids: continue

            action_indx = game.action_lookup[action]

            selfish_agent = SelfishAgent(self.genome)
            selfish_likelihood = Agent.decide_likelihood(selfish_agent, game, [selfish_agent.world_id, self.world_id], tremble)[action_indx]

            reciprocal_agent = self.agents[agent_ids[0]]
            reciprocal_likelhood = Agent.decide_likelihood(reciprocal_agent, game, agent_ids, tremble)[action_indx]

            if agent_ids[0] not in self.likelihood:
                self.likelihood[agent_ids[0]] = {
                    SelfishAgent.__name__ : 1,
                    ReciprocalAgent.__name__: 1
                }


            self.likelihood[agent_ids[0]][SelfishAgent.__name__] *= selfish_likelihood
            self.likelihood[agent_ids[0]][ReciprocalAgent.__name__] *= reciprocal_likelhood


            
        # Update the priors after getting the likelihood estimate for each agent
        # TODO: Should the K-1 agents also have priors that get updated?

        # Update all beliefs
        # for a_id in self.likelihood:
            # game, agent_ids, observer_ids, action = observation

            # if agent_ids[0] == self.world_id: continue
            # if self.world_id not in observer_ids: continue
            # if K>0:
                # import ipdb; ipdb.set_trace()
            self.belief[agent_ids[0]] = self.pop_prior[ReciprocalAgent.__name__] * self.likelihood[agent_ids[0]][ReciprocalAgent.__name__] / (self.pop_prior[ReciprocalAgent.__name__] * self.likelihood[agent_ids[0]][ReciprocalAgent.__name__] + self.pop_prior[SelfishAgent.__name__] * self.likelihood[agent_ids[0]][SelfishAgent.__name__])

        # Observe the other person, when this code runs at K=0
        # nothing will happen because of the return at the top
        # of the function.
        for a_id in self.agents:
            self.agents[a_id].observe_k(observations, K-1, tremble)

            
        if K>0:
            self.update_prior()

    def update_prior(self):
        if self.genome['RA_prior_precision'] == 0: return
        
        D = list()
        for a, l in self.likelihood.iteritems():
            p = l[ReciprocalAgent.__name__] / (l[ReciprocalAgent.__name__] + l[SelfishAgent.__name__])

            w = abs(p - (1 - p))
            D.append([p*w, (1-p)*w])

        if not len(D): return # No observations

        D = np.array(D)
        
        prior = np.array([self.genome['RA_prior'], 1-self.genome['RA_prior']]) * self.genome['RA_prior_precision']

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

        
def generate_random_genomes(N, agent_types, RA_prior, RA_prior_precision, beta):
    genomes = []

    for _ in range(N):
        genomes.append({
            'type': np.random.choice(agent_types),
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
            ReciprocalAgent
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
    params['agent_types'] = [ReciprocalAgent]
    N_round = 10
    params['stop_condition'] = constant_stop_condition(N_round)
    data = []
    N_runs = 500
    for RA_prior in np.linspace(0.5, 0.95, 4):
        params['RA_prior'] = RA_prior
        print 'running prior', RA_prior

        for r_id in range(N_runs):
            np.random.seed(r_id) # Increment a new seed for each run
            w = World(params, generate_random_genomes(params['N_agents'], params['agent_types'], params['RA_prior'], params['RA_prior_precision'], params['beta']))
            fitness, history = w.run()
            for nround in range(len(history)):
                avg_beliefs = np.mean([history[nround]['belief'][0][w.agents[1].world_id], history[nround]['belief'][1][w.agents[0].world_id]])
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
    params['agent_types'] = [ReciprocalAgent, SelfishAgent]
    params['stop_condition'] = constant_stop_condition(10)
    data = []
    N_runs = 500
    for RA_prior in np.linspace(0.5, .95, 4):
        params['RA_prior'] = RA_prior
        print 'running prior', RA_prior

        for r_id in range(N_runs):
            np.random.seed(r_id)
            w = World(params, [
                {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'], 'RA_prior_precision': params['RA_prior_precision'],'beta': params['beta']},
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
            
            w = World(params, generate_random_genomes(params['N_agents'], params['agent_types'], params['RA_prior'], params['beta']))
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

params['stop_condition'] = constant_stop_condition(10)
params['p_tremble'] = 0
params['RA_prior'] = 0.8
params['RA_prior_precision'] = 10

w = World(params, [
    {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'RA_prior_precision': params['RA_prior_precision'], 'beta': params['beta']},
    {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'RA_prior_precision': params['RA_prior_precision'], 'beta': params['beta']},
    {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'RA_prior_precision': params['RA_prior_precision'], 'beta': params['beta']},
    {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'RA_prior_precision': params['RA_prior_precision'], 'beta': params['beta']},
    
])

observations= [
    (w.game, [0, 1], range(len(w.agents)), 'keep'),
    (w.game, [1, 0], range(len(w.agents)), 'give'),
    (w.game, [0, 1], range(len(w.agents)), 'keep'),
    (w.game, [1, 0], range(len(w.agents)), 'keep'),
    (w.game, [0, 1], range(len(w.agents)), 'keep'),
    (w.game, [1, 0], range(len(w.agents)), 'keep'),
    # (w.game, [0, 1], range(len(w.agents)), 'keep'),
    # (w.game, [1, 0], range(len(w.agents)), 'keep'),
    # (w.game, [0, 1], range(len(w.agents)), 'keep'),
    # (w.game, [1, 0], range(len(w.agents)), 'keep'),
    (w.game, [1, 2], range(len(w.agents)), 'give'),
    (w.game, [2, 1], range(len(w.agents)), 'give'),
    # (w.game, [1, 2], range(len(w.agents)), 'give'),

    
    # (w.game, [0, 1], range(len(w.agents)), 'keep'),
    (w.game, [2, 3], range(len(w.agents)), 'give'),
    # (AllocationGame(), [2, 0, 1], range(len(w.agents)), 'give 1'),
]


K = 1
i = 3
w.agents[i].observe_k(observations, 1)
print w.agents[i].belief
print w.agents[i].pop_prior['ReciprocalAgent']




# for i in range(len(w.agents)):
#     w.agents[i].observe_k(observations, K)
#     print i
#     pprint(w.agents[i].belief)
    # for k in w.agents[i].agents:
    #     print k,
    #     pprint( w.agents[i].agents[k].belief)



# forgiveness_experiment(overwrite=True)
# forgiveness_plot() 

# protection_experiment(overwrite=True)
# protection_plot()

# fitness_rounds_experiment(overwrite=True)
# fitness_rounds_plot()

