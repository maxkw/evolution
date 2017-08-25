from __future__ import division
from games import IndirectReciprocity
from world import World
from agents import WeAgent, AllD, ReciprocalAgent,AllC,shorthand_to_standing,leading_8_dict
from collections import defaultdict
from params import default_params,default_genome
from experiment_utils import experiment, plotter, multi_call, MultiArg
import numpy as np
import seaborn as sns
import pandas as pd
from utils import softmax_utility, softmax, normalized, memoized, logspace
from functools import partial
from evolve import limit_param_plot, limit_param_plot, ssd_v_param
from steady_state import all_partitions, steady_state, mm_to_limit_mcp, mcp_to_invasion
from itertools import permutations,product,imap,chain,repeat, izip,starmap,combinations
from math import factorial,ceil

from games import RepeatedPrisonersTournament
from experiments import matchup, matchup_matrix_per_round


def image_contest():
    pop_size = 20
    rounds = 50
    

    unique_interactions =  factorial(pop_size)/factorial(pop_size-2)
    assert unique_interactions >= rounds

    opponent_types = (
        AllD,
        AllC,
    )
    W = WeAgent
    S = shorthand_to_standing("ggggbbbbynyn")

    shorthand_name_pairs = (
        ("ggggbbbb", "Scoring"),
        ("ggggbgbb", "Standing"),
        ("gbgbbgbb", "Judging"),
        ("gbgbbbbb", "Shunning"))
    default_action = "ynyn"

    named_imagers = []
    for s,n in shorthand_name_pairs:
        t = shorthand_to_standing(s+default_action)
        t._nickname = n
        named_imagers.append(t)
    #named_imagers = tuple(named_imagers)

    standing_types = tuple(s_type for name,s_type in sorted(leading_8_dict().iteritems(),key = lambda x:x[0]))
    
   
    #+standing_types#[0:1]

    WeRange = [WeAgent(agent_types = ('self',)+opponent_types, beta = b, RA_prior = p)
                for b,p in product([3,5,10],[.25,.5,.75])]

    player_types = [WeAgent(agent_types = ('self',)+opponent_types, beta = 10, RA_prior = .5),
                    #S
    ]+named_imagers

    #player_types = WeRange#+named_imagers

    for s in [1]:#range(len(player_types)+1):
        pt = player_types
        for t in [#0,
                  .05
        ]:
            limit_param_plot(
                'rounds',
                experiment = compare_ssd_v_param,
                player_types = pt,
                opponent_types = opponent_types,
                tremble = .05,
                rounds = rounds,
                benefit = 10,
                s = s,
                pop_size = pop_size
            )
            break

# def sim_complete_analysis(simulator, types,  pop_size, s, **kwargs):
#     """
#     calculates the steady state distribution over population compositions
#     """
#     type_count = len(types)
#     transition = sim_pop_transition_matrix(simulator, types, pop_size, s, **kwargs)
#     ssd = steady_state(transition)

#     pop_sum = np.zeros(type_count)
#     for p, partition in zip(ssd, sorted(set(all_partitions(pop_size, type_count)))):
#         pop_sum += np.array(partition) * p

#     return pop_sum / pop_size

# def sim_complete_analysis_per_round(types, pop_size, s, rounds, **kwargs):
#     trials = 100
#     type_count = len(types)
#     partitions = set(all_partitions(pop_size,type_count))
    
#     part_to_id = dict(map(reversed,enumerate(sorted(partitions))))
#     rounds_list = range(1,rounds+1)

#     pool = Pool(8)
#     def part_to_argdict(part):
#         prts,typs = zip(*((p,t) for p,t in zip(part,types) if p is not 0))
#         return dict(counts = prts, player_types = typs, rounds = rounds, trials = trials, **kwargs)

#     partition_to_round_to_fitness = zip(partitions, pool.map(indirect_simulator_from_dict, map(part_to_argdict,partitions)))
    
#     partition_to_fitness_per_round = {}
#     for p,rtf in partition_to_round_to_fitness:
#         #partition_to_fitness_per_round[p]
#         round_to_fitness = {}
#         non_zero_types = np.array(p)!=0
#         for r,f in enumerate(rtf,1):
#             fitness = np.zeros_like(types)
#             fitness[non_zero_types] = softmax(f,s)
#             round_to_fitness[r] = fitness
#         partition_to_fitness_per_round[p] = round_to_fitness
  
#     expected_pop_per_round = {}
#     part_to_fitness = {}
#     for r in rounds_list:
#         for partition in partitions:
#             try:

#                 part_to_fitness[partition] = partition_to_fitness_per_round[partition][r]
#             except:
#                 #print partition
#                 #print r
#                 #print partition_to_fitness_per_round[partition]
#                 raise
#         transition_matrix = fitnesses_to_transition_matrix(part_to_fitness,types,pop_size)
#         ssd = steady_state(transition_matrix)

#         pop_sum = np.zeros(type_count)
#         for p, partition in zip(ssd, sorted(partitions)):
#             pop_sum += np.array(partition) * p

#         expected_pop_per_round[r] = pop_sum / pop_size
#     return expected_pop_per_round



def test_sim_limit_analysis():
    
    pop_size = 10

    #rounds/(pop-1) = avg decisions per agent
    #rounds/((pop-1)*2) = avg interactions per agent

    # Average interactions per agent
    aipa = 2.5

    #rounds = int(ceil(aipa*(pop_size-1)*2))
    rounds = 50

    opponents = (AllC,AllD)
    types = (WeAgent(RA_prior = .5, beta = 10, agent_types = ('self',)+opponents),)+opponents
    params = dict(param = 'rounds',
                  player_types = types,#tuple(sorted(types)),
                  tremble = 0,
                  benefit = 10,
                  s = 1,
                  games = IndirectReciprocity,
                  pop_size = pop_size,
                  extension = ".png",
    )

    ana_params = dict(experiment = ssd_v_param, rounds = rounds, direct = True,  **params)
    sim_params = dict(experiment = ssd_v_param, rounds = rounds, direct = False,  **params)

    limit_param_plot(**sim_params)
    #limit_param_plot(**ana_params)
    

def test_rmcp_creation():
    rounds = 50
    pop_size = 10
    opponents = (AllD,AllC)
    types = opponents#+(WeAgent(RA_prior = .5, beta = 10, agent_types = ('self',)+opponents),)#+opponents
    params = dict(param = 'rounds',
                  player_types = tuple(sorted(types)),
                  tremble = .05,
                  benefit = 10,
                  rounds = 50,
                  games = IndirectReciprocity,
                  s = 1,
                  pop_size = pop_size,
                  trials = 1
    )



    """
    make mcp from payoffs
    """

    ana_rmcp = ana_to_rmcp(**params)


    """
    make mcp from simulation results
    """

    sim_rmcp = sim_to_limit_rmcp(**params)

    print zip(ana_rmcp[0],sim_rmcp[0])

    


if __name__ == "__main__":

    #for i in range(50):
        
    #    print i,len(set(all_partitions(i,3)))
    #image_contest()
    #sim_invasion_matrix(0,"ABCD",5,0,10)
    #test_sanity()
    test_sim_limit_analysis()
    #test_rmcp_creation()
    
     #test_matchup_to_avg()

