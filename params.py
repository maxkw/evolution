import numpy as np
from indirect_reciprocity import SelfishAgent,ReciprocalAgent,NiceReciprocalAgent,AltruisticAgent,RationalAgent
from games import RepeatedPrisonersTournament
def default_params(agent_types = (SelfishAgent, ReciprocalAgent, AltruisticAgent),
                   RA_prior = .75, N_agents= 10, p_tremble = 0,rounds = 10, **kwargs):
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
    #agent_types =  [
    #    SelfishAgent,
    #    NiceReciprocalAgent,
    #    AltruisticAgent
    #]

    given_values = locals()
    given_values.update(kwargs)
    
    values =  {
        'N_agents':N_agents,
        'games': RepeatedPrisonersTournament(10),
        'agent_types' : agent_types,
        'beta': 3,
        'moran_beta': .1,
        'RA_prior': RA_prior,
        'p_tremble': 0.0,
        'agent_types_world': agent_types
    }

    for key in given_values:
        if key in values:
            values[key] = given_values[key]

    return values

def prior_generator(agent_types,RA_prior=False):
    """
    if RA_prior is False it generates a uniform prior over types
    if RA_prior is a dict from agent_type to a number it assigns those types
    the corresponding number
    if RA_prior is a number it divides that number uniformly among all rational types
    """
    
    agent_types = tuple(agent_types)
    type2index = dict(map(reversed,enumerate(agent_types)))
    size = len(agent_types)
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
print prior_generator((ReciprocalAgent,SelfishAgent),.75)[0]
assert prior_generator((ReciprocalAgent,SelfishAgent),.75)[0] == 0.75

def default_genome(agent_type = False, agent_types = None, RA_prior = .75, **extra_args):

    if not agent_types:
        agent_types = default_params["agent_types"]
    if not agent_type:
        agent_type = np.random.choice(agent_types)
    #print "args",agent_types,RA_prior
    #print "result",prior_generator(agent_types,RA_prior),
    genome = {
        'type': agent_type,
        'RA_prior': RA_prior,
        'prior_precision': 0,
        'beta': .3,
        'prior': prior_generator(agent_types,RA_prior),
        "agent_types":agent_types,
        'RA_K':2,
        'p_tremble':0
    }

    #print "keys\n\n\n",extra_args.keys()
    for key in extra_args:
        if key in genome:
            genome[key] = extra_args[key]

    return genome

def generate_random_genomes(N_agents, agent_types_world, **kwargs):
    return [default_genome(agent_type = np.random.choice(agent_types_world),**kwargs) for _ in range(N_agents)]

def generate_proportional_genomes(agent_proportions,**extra_args):
    if not agent_proportions:
        try:
            return generate_random_genomes(**extra_args)
        except:
            print "did you not provide proportions or N_agents+agent_types_world as parameters? bad move."
            raise

    agent_list = []
    pop_size = params['N_agents']
    for agent_type in sorted(agent_proportions.keys()):
        number = int(math.ceil(pop_size*agent_proportions[agent_type]))
        agent_list.extend([default_genome(agent_type,**extra_args) for _ in xrange(number)])
    return agent_list

