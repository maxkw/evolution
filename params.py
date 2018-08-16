import numpy as np
from agents import IngroupAgent, RationalAgent
from utils import _issubclass
import seaborn as sns
#from frozendict import frozendict

sns.set_style('ticks')
sns.set_context('paper', font_scale=1.5)

def default_params(agent_types = None, RA_prior = None, games = None, N_agents= 10, tremble = 0, rounds = 10, **kwargs):
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

    `tremble`: probability of noise in between forming a decision and making an action. 

    `RA_K`: is the number of theory-of-mind recursions to carry out. When RA_K is 0 the agent just tries to infer the type directly, when it is 1, you first infer what each agent knows and then infer what you know based on those agents and so on. 

    """

    given_values = locals()
    given_values.update(kwargs)
    
    values =  {
        'N_agents' : N_agents,
        # 'games': games,
        'agent_types' : agent_types,
        'moran_beta' : .1,
        'tremble' : tremble,
        'agent_types_world': agent_types,
        'pop_size' : 100,
        's' : 1,
        'mu' : .01,
        'rounds' : rounds,
    }

    for key in kwargs:
        if key in values:
            values[key] = given_values[key]

    return values

def default_genome(agent_type = False, agent_types = None, prior = .5, **extra_args):

    if not agent_types:
        agent_types = default_params()["agent_types"]
    if not agent_type:
        agent_type = np.random.choice(agent_types)

    try:
        RA_prior = agent_type.genome["RA_prior"]
    except:
        pass

    try:
        agent_types = agent_type.genome['agent_types']
    except:
        pass
    if agent_types:
        agent_types = tuple(t if t != 'self' else agent_type for t in agent_types)
    genome = {
        'type': agent_type,
        'prior': prior,
        'prior_precision': 0,
        'beta': 5,
        #'prior': prior_generator(agent_type, agent_types, RA_prior = prior),
        "agent_types": agent_types,
        'RA_K':0,
        'tremble':0,
        'y':1,
        'p':1,
        'q':0,
    }

    for key in extra_args:
        if key in genome and key is not 'prior':
            genome[key] = extra_args[key]

    return genome
