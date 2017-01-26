from __future__ import division
import pandas as pd
from experiment_utils import multi_call
import numpy as np
from indirect_reciprocity import World,default_params,generate_proportional_genomes
from indirect_reciprocity import ReciprocalAgent,SelfishAgent
from experiment_utils import is_sequency
from games import RepeatedPrisonersTournament
from collections import defaultdict


#print is_sequency(np.linspace(.1,.9,5))

###multi_call


@multi_call
def fitness_v_selfish(RA_K = [1], proportion = [round(n,5) for n in np.linspace(.1,.9,5)], N_agents = 50, visibility = "private", observability = .5, trial = range(10), RA_prior = .80, p_tremble = 0, agent_type = ReciprocalAgent, rounds = 10):
    
    condition = locals()
    condition['agent_types'] = [agent_type,SelfishAgent]
    condition['games'] = RepeatedPrisonersTournament(**condition)
    
    params = default_params(**condition)
    
    proportions = {agent_type:proportion,
                   SelfishAgent:1-proportion}

    world = World(params,generate_proportional_genomes(params,proportions))
    print world.agents
    fitness,history = world.run()
    
    ordered_types = [type(agent) for agent in world.agents]
    fitnesses = defaultdict(int)
    
    for a_type,fitness_score in zip(ordered_types,fitness):
        fitnesses[a_type] += fitness_score
        
    for a_type in params["agent_types_world"]:
        fitnesses[a_type] /= ordered_types.count(a_type)

    print fitnesses
    relative_avg_fitness = fitnesses[agent_type]/fitnesses[SelfishAgent]
    
    return relative_avg_fitness

fitness_v_selfish(N_agents = 1,trial =  range(100))

