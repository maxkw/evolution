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
from evolve import limit_param_plot, limit_param_plot,limit_v_param
from steady_state import all_partitions, steady_state, payoff_to_mcp_matrix, mcp_to_invasion
from itertools import permutations,product,imap,chain,repeat, izip,starmap,combinations
from math import factorial,ceil
from multiprocessing import Pool
from games import RepeatedPrisonersTournament
from experiments import matchup, matchup_matrix_per_round


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

@experiment(unpack = 'record', memoize = False, verbose = 3)
def sim_ssd_v_param(param, player_types, **kwargs):

    # Test to make sure each agent interacts with a new agent each
    # time. Otherwise its not true 'indirect' reciprocity.
    unique_interactions = kwargs['pop_size'] * (kwargs['pop_size'] - 1)
    if unique_interactions <= kwargs['rounds']:
        raise Exception("There are more rounds than unique interactions. Raise pop_size or lower rounds.")

    record = []

    if param == "rounds":
        expected_pop_per_round = sim_limit_analysis(player_types = player_types, **kwargs)
        for r, pop in enumerate(expected_pop_per_round):
            for t, p in zip(player_types, pop):
                record.append({
                    'rounds': r,
                    'type': t.short_name('agent_types'),
                    'proportion': p
                })

        return record

    if param == "RA_prior":
        Xs = np.linspace(0,1,21)[1:-1]
    elif param == "beta":
        Xs = logspace(.5,6,11)
    elif param == 'pop_size':
        Xs = np.unique(np.geomspace(2, 2**10, 200, dtype=int))
    elif param == 's':
        Xs = logspace(start = .001, stop = 15, samples = 100)
    elif param == 'benefit':
        Xs = [1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
    else:
        print param
        raise Exception("Unknown param provided")

    for x in Xs:
        for t, p in zip(player_types, sim_limit_analysis(types = player_types, **dict(kwargs,**{param:x}))[-1]):
            record.append({
                param: x,
                "type": t.short_name("agent_types"),
                "proportion": p
            })
    return record

@memoized
def indirect_simulator(player_types, rounds, *args, **kwargs):
    matchup_data = matchup(per_round = True, player_types = player_types, rounds = rounds, *args, **kwargs)
    fitness_per_round = avg_payoff_per_type_from_matchup(matchup_data)
    return fitness_per_round

def indirect_simulator_from_dict(d):
    return indirect_simulator(**d)

def compare_sim_ssd_v_param(param, player_types, opponent_types, **kwargs):
    dfs = []
    for player_type in player_types:
        df = sim_ssd_v_param(param = param, player_types = (player_type,)+opponent_types, **kwargs)
        dfs.append(df[df['type']==player_type.short_name("agent_types")])
    return pd.concat(dfs, ignore_index = True)


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
                experiment = compare_sim_ssd_v_param,
                player_types = pt,
                opponent_types = opponent_types,
                tremble = .05,
                rounds = rounds,
                benefit = 10,
                s = s,
                pop_size = pop_size
            )
            break


def sim_limit_analysis(player_types, pop_size, s, rounds, trials = 100, **kwargs):
    rmcp = sim_to_rmcp(player_types, pop_size, rounds, trials, **kwargs)

    ssds = []
    for r, mcp in enumerate(rmcp):
        ssds.append(mcp_to_ssd(mcp, s))
        
    return ssds

def sim_to_rmcp(player_types, pop_size,rounds, trials, **kwargs):
    pool = Pool(8)

    #produce all elements along the edges of the population simplex
    #does not include the homogeneous populations at the vertices
    #ordered populations, going from (1,pop_size-1) to (pop_size-1,1)
    liminal_pops = [(i, pop_size-i) for i in range(1,pop_size)]

    #all the pairings of two player_types, note these are combinations
    unique_matchups = list(combinations(player_types,2))

    matchups, counts = zip(*list(product(unique_matchups, liminal_pops)))

    def part_to_argdict(matchup, count):
        return dict(player_types = zip(matchup,count), rounds = rounds, trials = trials, **kwargs)

    payoffs = pool.map(indirect_simulator_from_dict, imap(part_to_argdict, matchups, counts))

    #make a mapping from matchup to list of lists of payoffs
    #the first level is ordered by partitions
    #the second layer is ordered by rounds

    type_indices_matchups = list(combinations(range(len(player_types)), 2))
    rmcp = np.zeros((rounds, len(unique_matchups), len(liminal_pops), 2))

    # Unpack the data into a giant matrix
    for (m, c), p in zip(product(range(len(unique_matchups)), range(len(liminal_pops))), payoffs):
        rmcp[:, m, c, :] = np.array(p)

    rmcp /= (pop_size-1)
    return rmcp
  
def avg_payoff_per_type_from_matchup(matchup_data):
    running_fitness = 0
    fitness_per_round = []

    for r, r_d in matchup_data.groupby('round'):
        fitness = []
        for i, (t, t_d) in enumerate(r_d.groupby('type')):
            fitness.append(t_d['fitness'].mean())

        running_fitness += np.array(fitness)
        fitness_per_round.append(np.array(running_fitness)/r)

    return fitness_per_round[1:]

def ana_to_rmcp(player_types,pop_size,rounds,trials, **kwargs):
    payoffs = matchup_matrix_per_round(player_types = player_types, max_rounds = rounds, trials = trials, **kwargs)
    rmcp = np.array([payoff_to_mcp_matrix(payoff,pop_size) for r,payoff in payoffs])
    return rmcp

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
                  player_types = tuple(sorted(types)),
                  tremble = 0,
                  benefit = 10,
                  s = 1,
                  pop_size = pop_size,
    )

    sim_params = dict(experiment = sim_ssd_v_param, rounds = rounds, file_name = "sim",  **params)
    classic_params = dict(experiment = limit_v_param, max_rounds = rounds, analysis = "limit", file_name = "heatmap", **params)
    limit_param_plot(analysis = 'limit', **sim_params)
    #limit_param_plot(analysis = 'complete', **sim_params)
    limit_param_plot(**classic_params)


    # c = sim_complete_analysis_per_round(**params)
    #l = sim_limit_analysis(**params)

    #for r in xrange(1,50):
        # print c[r],l[r]
    #    print l[r]


def mmpr_to_cumulative_payoff(pop, mmpr):
    cumulatives = []
    I = np.identity(len(pop))
    counts = np.array(pop)
    pop_size = sum(pop)

    for r, payoffs in mmpr:
        pay = [np.dot(counts-I[t], payoffs[t])/(pop_size-1) for t in [0,1]]
        cumulatives.append(pay)
        
    return cumulatives

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

    sim_rmcp = sim_to_rmcp(**params)

    print zip(ana_rmcp[0],sim_rmcp[0])


def test_matchup_to_avg():
    rounds = 50
    pop_size = 10
    opponents = (AllD,)
    types = opponents+(WeAgent(RA_prior = .5, beta = 10, agent_types = ('self',)+opponents),)#+opponents
    #types = opponents+(AllC, )
    types = tuple(sorted(types))
    params = dict(param = 'rounds',
                  tremble = .0,
                  benefit = 10,
                  s = 1,
                  pop_size = pop_size,
                  trials = [1]
    )

    pop = (5,5)
    m_d = matchup(player_types = zip(types,pop), per_round = True, rounds = rounds, **params)
    print "run once"
    m = avg_payoff_per_type_from_matchup(m_d)
    mm = mmpr_to_cumulative_payoff(pop, matchup_matrix_per_round(player_types = types, max_rounds = rounds, **params))
    assert len(mm) == len(m)

    for r, (mat,vec) in enumerate(zip(mm,m)):
        print r
        print "ana", mat
        print "sim",vec/(pop_size-1)
    


if __name__ == "__main__":

    #for i in range(50):
        
    #    print i,len(set(all_partitions(i,3)))
    #image_contest()
    #sim_invasion_matrix(0,"ABCD",5,0,10)
    #test_sanity()
    test_sim_limit_analysis()
    #test_rmcp_creation()
    
     #test_matchup_to_avg()

