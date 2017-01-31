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

    """
    Make a population of size 'N_agents' where 'proportion' percent of the population is of type 'agent_type' 
    and the remainder is of type 'SelfishAgent'. have them play RepeatedPrisonersTournament with each other
    return the ratio of average payoffs to agent_type vs average payoffs to SelfishAgent. 
    """
    
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

#fitness_v_selfish(N_agents = 1,trial =  range(100))

@multi_call
def first_impressions(RA_K=2,RA_prior=[.25,.5,.75],rational_type = ReciprocalAgent, agent_types = [[ReciprocalAgent,SelfishAgent]],N_cooperation=5):
    """
    An observer 'O' sees an agent 'A' cooperate 'N_cooperation' times with another agent 'B' before defecting once.
    What is O's belief that A is of type 'rational_type'?
    """
    params = default_params(RA_K =  RA_K, RA_prior = RA_prior, agent_types = agent_types)
    action_strings = ["C"*n+"D" for n in range(N_cooperation)]
    BD = BinaryDictator()

    def char_to_observation(action_char,order=[0,1]):
        action = "give" if action_char is "C" else "keep"
        return [(BD,order,[0,1,"O"],action)]
    
    def observations(actions_string):
        return map(char_to_observation,actions_string)

    record = []; append = record.append
    for OAB, OBA in [(RA_prior,RA_prior)]:
        for action_string in action_strings:
            observer = RationalAgent(default_genome(params,RationalAgent),'O')
            
            for i,observation in enumerate(observations(action_string)):                    
                observer.observe(observation)
            record.append(
                {
                    'belief':observer.belief_that(0,rational_type),
                    'action':action_string,
                    'cooperations':i,
                    'prior':RA_prior,
                })
    return record
