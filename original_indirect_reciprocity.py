from __future__ import division
import scipy as sp
import numpy as np
import pandas as pd
import networkx as nx
import itertools
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

from functools import partial
from utils import unpickled, pickled

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

class AgentType(type):
    def __str__(cls):
        return cls.__name__

    def __hash__(cls):
        return str(cls)
        
class Agent(object):
    __metaclass__ = AgentType
    def __init__(self, genome, world_id=None):
        self.genome = deepcopy(genome)
        self.beta = self.genome['beta']
        self.world_id = world_id
        self.belief = dict()


    def utility(self, payoff, agent):
        # TODO: Change in each of the agent classes to do the loop to
        # reflect the new decide_likelihood code.
        
        raise NotImplementedError

    def observe_k(self, observations, k, tremble = 0):
        pass

    @staticmethod
    def decide_likelihood(deciding_agent, game, agent_ids, tremble = 0):
        """This function is importantly used in two places. 
        1) It is used by the Agent when he makes his decision and returns a probability of taking each action.
        2) It is also used by the Agent when he makes an observation. This function then returns a likelihood of that decision. 

        `deciding_agent`: A reference to the agent who made the
        decision in this game. The first agent in the `agent_ids` is
        always this variable.

        `game`: an instance of the game being played.

        `agent_ids`: the world_ids of the agents playing the game. The
        order is important since the order of agent_ids corresponds to
        the order of the payoffs.

        `tremble`: trembling hand probability this should be 0 when deciding. It is only used in the likelihood.

        returns a probability vector representing what I think the
        deciding agent (possibly myself) would do in this game.

        """

        # Only have one action so just pick it
        if len(game.actions) == 1:
            # Returning a probability a vector
            return np.array([1])

        # BUG: META COMMENT ON BELOW. This is just saying the below
        # code is specific for the DictatorGames currently implemented
        # but it doesn't do any planning etc in multistage games. This
        # could be place that code will have to change to handle more
        # interesting games.

        # BUG: Get probability of the other players taking an action,
        # otherwise this only works for dictator game. For now just
        # assume its uniform since it doesn't matter for dictator
        # game. Can/Should use the belief distribution. May need to do
        # a logit response for simultaneous move games.

        # Utilities: Map between action_id and the utility for that action. 
        Us = np.zeros(len(game.actions))

        # Iterate over the actions in the game
        for a in game.actions:
            # Get the action id for that action
            a_id = game.action_lookup[a]

            # TODO: Assess the payoffs one-by-one for each of the
            # payoffs generated by that action. You should move this
            # loop into the utility function. So you can call
            # deciding_agent.utility(game.payoffs[a], agent_ids)
            # works.
            for payoff, agent_id in zip(game.payoffs[a], agent_ids):
                Us[a_id] += deciding_agent.utility(payoff, agent_id)

        # Return a linear combination of the softmax utility and a uniform distribution. 
        return (1-tremble) * softmax(Us, deciding_agent.beta) + tremble * np.ones(len(Us))/len(Us)

    def decide(self, game, agent_ids):
        # Tremble is always 0 for decisions since tremble happens in
        # the world, not the agent

        # Get the likelihood of each action. 
        ps = self.decide_likelihood(self, game, agent_ids, tremble = 0)

        # Samples from the distribution and selects the action_id
        action_id = np.squeeze(np.where(np.random.multinomial(1,ps)))

        # Return the label of the action
        return game.actions[action_id]


class SelfishAgent(Agent):
    def __init__(self, genome, world_id=None):
        super(SelfishAgent, self).__init__(genome, world_id)

    def utility(self, payoff, agent_id):
        # TODO: Update for new decide_likelihood
        if agent_id == self.world_id:
            return payoff
        else:
            return 0

class AltruisticAgent(Agent):
    def __init__(self, genome, world_id=None):
        super(AltruisticAgent, self).__init__(genome, world_id)

    def utility(self, payoff, agent_id):
        # TODO: Update for new decide_likelihood
        return payoff

        
class ReciprocalAgent(Agent):
    def __init__(self, genome, world_id=None):
        super(ReciprocalAgent, self).__init__(genome, world_id)
        # The prior probability that this agent believes the a new
        # agent will be of a certain type. The values should sum to 1.
        # FIXME: This will need to change if you use more than these
        # two agents.
        self.pop_prior = {
            ReciprocalAgent.__name__: self.genome['RA_prior'],
            SelfishAgent.__name__ : 1-self.genome['RA_prior']
        }

        # My belief about each agent in the world. The key is the
        # world_id of the agent and the value is a single number that
        # is the probability that the agent with that key is a ReciprocalAgent. 
        self.belief = dict()
        # Simulated models of other agents for likelihood comparisons etc. 
        self.models = dict()
        self.likelihood = dict()
        
    def utility(self, payoff, agent_id):
        # TODO: Update for new decide_likelihood
        if agent_id == self.world_id:
            return payoff
        else:
            # If I don't have any beliefs about that agent, use my prior. 
            if agent_id not in self.belief:
                self.belief[agent_id] = self.initialize_prior()

            alpha = self.sample_alpha(self.belief[agent_id])
            return alpha * payoff

    def sample_alpha(self, belief):
        # Decide what to do here? Should this be sampling and
        # then giving full weight? Or should it be weighting how much
        # we care? The weighted version is worse at punishing the bad
        # guys since it will still weight them a bit even when its
        # highly unlikely that they are reciprocal... Maybe this is
        # just a downside of being a nice person?

        # TODO: Make this abstract. What if you I want my alpha to be:
        # the probability that YOU would value me. So if I'm 100% sure
        # you are Reciprocalagent I will give you a 1. BUT this should
        # be true for the Altrustic type too so it can't just depend
        # on my belief but something richer than that.
        
        # return int(flip(belief))
        return belief

    def initialize_prior(self):
        # This needs to initialize the data structure that is used for the online update
        return self.pop_prior[ReciprocalAgent.__name__]

    def observe_k(self, observations, K, tremble = 0):
        """
        takes in observations = [(game, agent_ids, observer_ids, action), ...]
        
        everything in the observe list is observed at the same time. If you want to observe things in a sequence you need to call this function multiple times. 

        k = an integer. (function has special behavior for k =,<,> 0)
        
        Undefined for K<0. 
        K = 0 means that my model of you is one where you are not learning.
        K = 1 means that my model of you is one where you are learning but assuming everyone else is not learning. 
        K = 2 means that my model of you is one where you are learning, and assuming that everyone else is learning but that they don't think anyone is learning. 
        ....

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

        # This loop is for initializing the agents models of other agents
        for observation in observations:

            game, agent_ids, observer_ids, action = observation
        
            # Do recursive theory of mind on all of the agents 
            # for pair in list(itertools.permutations(observer_ids,K+1)):
            for observer_id in observer_ids:
                # Can't have a belief about what I think about what I
                #think. Beliefs about others are first order beliefs.
                #continue skips the remainder of this looping instance
                if observer_id == self.world_id: continue

                # Initialize the level-2 agents
                if observer_id not in self.models:
                    #assume the agent is reciprocal
                    self.models[observer_id] = ReciprocalAgent(self.genome, world_id = observer_id)
                    #for every observer, initialize their beliefs about the new agent to their prior
                    for o_id in observer_ids:
                        self.models[observer_id].belief[o_id] = self.models[observer_id].initialize_prior()

                
        for observation in observations:
            game, agent_ids, observer_ids, action = observation

            # Can't have a belief about what I think about what I think. Beliefs about others are first order beliefs.
            # so if i'm considering myself, skip to the next round of the loop. BUG: Remember the first agent_ids is the agent who acted
            if agent_ids[0] == self.world_id: continue
            
            #if I'm not one of the observers this round, skip to the next round. 
            if self.world_id not in observer_ids: continue

            action_indx = game.action_lookup[action]

            # TODO: The rest of this loop doesn't generalize to more than one agent. 
            
            selfish_agent = SelfishAgent(self.genome)
            selfish_likelihood = Agent.decide_likelihood(selfish_agent, game, [selfish_agent.world_id, self.world_id], tremble)[action_indx]

            # BUG: This needs to be generalized to more agent types.
            # Because self.agents *only* includes reciprocal agent
            # types!
            reciprocal_agent = self.models[agent_ids[0]]
            reciprocal_likelhood = Agent.decide_likelihood(reciprocal_agent, game, agent_ids, tremble)[action_indx]

            # TODO: Change to log-likelihood
            if agent_ids[0] not in self.likelihood:
                self.likelihood[agent_ids[0]] = {
                    SelfishAgent.__name__ : .5,
                    ReciprocalAgent.__name__: .5
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

            # Combine my likelihood with my prior to get my belief that agent_ids[0] is reciprocal. 
            self.belief[agent_ids[0]] = (self.pop_prior[ReciprocalAgent.__name__] * self.likelihood[agent_ids[0]][ReciprocalAgent.__name__]) / (self.pop_prior[ReciprocalAgent.__name__] * self.likelihood[agent_ids[0]][ReciprocalAgent.__name__] + self.pop_prior[SelfishAgent.__name__] * self.likelihood[agent_ids[0]][SelfishAgent.__name__])

        # Observe the other person, when this code runs at K=0
        # nothing will happen because of the return at the top
        # of the function. BUG: Should this only run on agents that are in the observer_ids or does it need to run on ALL agents that the agent has beliefs about?
        for a_id in self.models:
            self.models[a_id].observe_k(observations, K-1, tremble)

        # # Update the prior at the end but can comment it out and it will still work!
        # if K>0:
            # self.update_prior()

#    def update_prior(self):
#        """Do a hierarchical update on the prior by considering the agents
#        confidence in the types of agents and finds the MAP prior.
#        Will need to go from a Beta prior to a Dirichlet code to
#        generalize to multiple agent types.
#        """
#        
#        if self.genome['RA_prior_precision'] == 0: return
#        
#        D = list()
#        for a, l in self.likelihood.iteritems():
#            p = l[ReciprocalAgent.__name__] / (l[ReciprocalAgent.__name__] + l[SelfishAgent.__name__])
#            w = abs(p - (1 - p))
#            D.append([p*w, (1-p)*w])
#
#        if not len(D): return # No observations
#
#        D = np.array(D)
#        
#        prior = np.array([self.genome['RA_prior'], 1-self.genome['RA_prior']]) * self.genome['RA_prior_precision']
#
#        def ll(p):
#            # Beta Prior on Mean
#            logprior = sum((prior-1) * np.log([p[0], 1-p[0]]))
#
#            # Binomial likelihood
#            RA, S = D.sum(axis=0)
#            return -(
#                RA * np.log(p[0]) + S * np.log(1-p[0]) 
#                + logprior)
#        
#        out = sp.optimize.minimize(ll, D.sum(axis=0)[0], bounds = [(.001, .999)])
#        
#        self.pop_prior = {
#            ReciprocalAgent.__name__: out.x[0],
#            SelfishAgent.__name__ : 1-out.x[0]
#        }
        
def generate_random_genomes(N, agent_types, RA_prior, RA_prior_precision, beta):
    # Currently only agent_types is a list. So its the only thing
    # being randomized over. You can make the others into lists which
    # will cause those to be randomized over as well.
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
            # Initialize a new agent and add it to a list. 
            self.agents.append(
                genome['type'](genome, world_id)
            )
            self.id_to_agent[world_id] = self.agents[-1]
        
        self.game = params['games']
        self._stop_condition = params['stop_condition']
        self.stop_condition = partial(*params['stop_condition'])
        self.params = params
        self.tremble = params['p_tremble']
        self.last_run_results = {}

    def make_pickleable(self):
        return self
    def restore(self):
        return self
    
    def evolve(self, fitness, p=1, beta=1, mu=0.05):
        """
        This should implement something like the moran process. 
        Search for moran in this: http://www.pnas.org/content/110/7/2581.full.pdf for references
        Also: https://en.wikipedia.org/wiki/Moran_process

        `fitness`: the vector of fitnesses for these agents.

        `p`: proportion of agents who die and replaced by selection

        `beta`: selection strength. When 0 random replication, when Inf, only the top agent reproduces. This is the moran Beta

        `mu`: drift rate. Which corresponds to random replication (mutation). You can think of this like a trembling hand. Proportion of agents who die and are changed by drift.  
        """
        # FIXME: Need to increment the id's. Can't just make new
        # agents, otherwise new agents will be treated as old agents
        # if they share the same ID

        # FIXME: Can't overwrite the self.agents[] with a new agent
        # since everything is by agent_id.
        
        assert 0 # BROKEN (See above)

        # See who will be replaced by selection
        die = np.random.choice(range(len(self.agents)), int(p*len(self.agents)), replace=False)
        
        # See who will be replaced by drift. (these guys also die)
        die_random = np.random.choice(range(len(self.agents)), int(mu*len(self.agents)), replace=False)

        # Replace by selection
        for a in die:
            # Pick based on the softmax so this is where selection happens. 
            copy_id = sample_softmax(fitness, beta)
            self.agents[a] = self.agents[copy_id].__class__(self.agents[copy_id].genome)

        # Replace by drift. Select a new genome randomly from the
        # space of possible genomes. NOT from the population.
        new_genomes = generate_random_genomes(len(die_random), self.params['agent_types'], self.params['RA_prior'], self.params['beta'])
        for a, ng in zip(die_random, new_genomes):
            self.agents[a] = ng['type'](ng)

    def run(self):
        # take in a sampling function
        fitness = np.zeros(len(self.agents))
        history = []

        # Get all pairs
        pairs = list(itertools.combinations(range(len(self.agents)), 2))
        np.random.shuffle(pairs)
        seeds = []
        for pair in pairs:
            rounds = 0
            while True:

                np.random.seed(rounds)

                rounds += 1
                likelihoods = []
                observations = []
                payoff = np.zeros(len(self.agents))
                # Have both players play both roles in the dictator game
                player_orderings = [[pair[0], pair[1]], [pair[1], pair[0]]]
                seeds.append(np.random.get_state()[2])
                for p0, p1 in player_orderings:
                    
                    agents = [self.agents[p0], self.agents[p1]]
                    agent_ids = [self.agents[p0].world_id, self.agents[p1].world_id]
                    
                    # Intention -> Trembling Hand -> Action
                    intentions = agents[0].decide(self.game, agent_ids)
                    likelihoods.append(Agent.decide_likelihood(agents[0],self.game,agent_ids))
                    actions = copy(intentions)
                    
                    #does this assume that everyone has the same actions?
                    #does everyone tremble independently?
                    #are these joint actions?

                    # if flip(self.tremble):
                    #     actions = np.random.choice(self.game.actions)

                    actions = intentions

                    payoff[[p0, p1]] += self.game.payoffs[actions]

                    # Determine who gets to observe this action. 
                    
                    # Reveal observations to update the belief state. This
                    # is where we can include more agents to increase the
                    # amount of observability
                    observer_ids = pair
                    observations.append((self.game, agent_ids, observer_ids, actions))
                seeds.append(np.random.get_state()[2])

                
                # Update fitness
                fitness += payoff

                # All observers see who observed the action. 
                # for o in observations:
                    # Iterate over all of the observers
                seeds.append(np.random.get_state()[2])

                for a in self.agents:
                    a.observe_k(observations, self.params['RA_K'], self.params['p_tremble'])

                print pair
                print observations
                for agent in self.agents:
                    print agent.belief

                    
                seeds.append(np.random.get_state()[2])    
                history.append({
                    'round': rounds,
                    'players': (self.agents[pair[0]].world_id, self.agents[pair[1]].world_id),
                    'actions': (observations[0][3], observations[1][3]),
                    'pair':player_orderings,
                    'likelihoods':likelihoods,
                    'observations':observations,
                    'payoff': payoff,
                    'belief': (copy(self.agents[0].belief), copy(self.agents[1].belief))
                })

                if self.stop_condition(rounds): break

        self.last_run_results = {'fitness': fitness,'history': history,'seeds':seeds}
        
        return fitness, history

    def _make_pickleable(self):
        self.agents = [agent2dict(agent) for agent in self.agents]
        self.id_to_agent = {}
        self.stop_condition = None
        return self
    
    def _make_unpickleable(self):
        self.agents = map(dict2agent,self.agents)
        self.id_to_agent = {agent.world_id:agent for agent in self.agents}
        self.stop_condition = partial(*self._stop_condition)
        return self
    
    @staticmethod
    def unpickle(path):
        world = unpickled(path)
        return world._make_unpickleable()

    def pickle(self,path):
        self._make_pickleable()
        pickled(self,path)
        return self._make_unpickleable()

def agent2dict(agent):
    if isinstance(agent, ReciprocalAgent):
        for id in agent.models:
            agent.models[id] = agent2dict(agent.models[id])
    return vars(agent)

def dict2agent(agentDict):
    agent = agentDict['genome']['type'](agentDict['genome'],agentDict['world_id'])
    agent.__dict__.update(agentDict)
    if isinstance(agent, ReciprocalAgent):
        for id in agent.models:
            agent.models[id] = dict2agent(agent.models[id])
    return agent
        
                
def discount_stop_condition(x,n):
    return not flip(x)
def constant_stop_condition(x,n):
    return n >= x

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
        'stop_condition': [constant_stop_condition,10],
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
    params['stop_condition'] = [constant_stop_condition,N_round]
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
    params['stop_condition'] = [constant_stop_condition,10]
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
    
def fitness_rounds_experiment(pop_size = 4, path = 'sims/fitness_rounds.pkl', overwrite = False):
    """
    Repetition supports cooperation. Look at how the number of rounds each dyad plays together and the average fitness of the difference agent types. 
    """
    
    print 'Running Fitness Rounds Experiment'
    if os.path.isfile(path) and not overwrite: 
        print path, 'exists. Delete or set the overwrite flag.'
        return


    params = default_params()
    params['N_agents'] = pop_size
    params['RA_K'] = 0
    params['RA_prior'] = .8
    params['agent_types'] = [ReciprocalAgent,SelfishAgent]
    N_runs = 1
    data = []
    for rounds in [2]:
    # for rounds in np.linspace(1, 8, 8, dtype=int):
        print rounds
        for r_id in range(N_runs):
            np.random.seed(r_id)

            params['stop_condition'] = [constant_stop_condition,rounds]
            
            w = World(params, generate_random_genomes(params['N_agents'], params['agent_types'], params['RA_prior'], params['RA_prior_precision'], params['beta']))
            fitness, history = w.run()
            print fitness

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
    return w

def fitness_rounds_plot(in_path = 'sims/fitness_rounds.pkl', out_path='writing/evol_utility/figures/fitness_rounds.pdf'):

    df = pd.read_pickle(in_path)
    sns.factorplot('rounds', 'fitness', hue='genome', data=df,)
    sns.despine()
    plt.ylim([0,1.05])
    plt.ylabel('Fitness ratio'); plt.xlabel('# of repetitions')
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

def manual_experiments():

    params = default_params()

    params['stop_condition'] = [constant_stop_condition,10]
    params['p_tremble'] = 0
    params['RA_prior'] = 0.8
    params['RA_prior_precision'] = 0

    w = World(params, [
        {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'RA_prior_precision': params['RA_prior_precision'], 'beta': params['beta']},
        {'type': SelfishAgent, 'RA_prior': params['RA_prior'],'RA_prior_precision': params['RA_prior_precision'], 'beta': params['beta']},
        {'type': ReciprocalAgent, 'RA_prior': params['RA_prior'],'RA_prior_precision': params['RA_prior_precision'], 'beta': params['beta']}
    ])

    K = 0
    
    observations= [
        (w.game, [0, 1], [0, 1], 'give'),
        (w.game, [1, 0], [0, 1], 'keep'),
    ]

    for a in w.agents:
        a.observe_k(observations, K)
    for a in w.agents:
        print a.belief
    
    observations= [
        (w.game, [0, 2], [0, 2], 'give'),
        (w.game, [2, 0], [0, 2], 'give'),
    ]

    for a in w.agents:
        a.observe_k(observations, K)
    for a in w.agents:
        print a.belief

    
    # i = 2
    # w.agents[i].observe_k(observations, K)
    # print w.agents[i].belief

    # w.agents[i].observe_k(obs, K)
    # print w.agents[i].belief

    # print w.agents[i].pop_prior['ReciprocalAgent']




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
manual_experiments()
