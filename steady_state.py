from __future__ import division
from collections import Counter, defaultdict
from itertools import product, permutations, combinations, izip
from utils import normalized, softmax, excluding_keys
from math import factorial
import numpy as np
from experiment_utils import multi_call, experiment, plotter, MultiArg
from observability_experiments import indirect_simulator
from functools import partial
from utils import memoized
from multiprocessing import Pool
from experiments import matchup,matchup_matrix_per_round
from complete import all_partitions


def steady_state(matrix):
    for i,c in enumerate(matrix.T):
        np.testing.assert_approx_equal(np.sum(c),1)
        try:
            assert all(c>=0)
        except:
            print c
            print matrix
            raise
    vals,vecs = np.linalg.eig(matrix)

    def almost_1(n):
        return np.isclose(n,1,.001) or (np.isclose(n.real,1,.001) and np.isclose(n.imag,0,.001))

    def negative_vec(vec):
        return all([i<0 or np.isclose(i,0) for i in vec])

    steady_states =[]
    for val,vec in sorted(zip(vals,vecs.T),key = lambda a:a[0]):
        
        if np.isclose(val,1):
            
            if negative_vec(vec):
                steady_states.append((val,np.absolute(vec)))
            elif all(np.logical_or(np.isclose(vec, 0), vec >= 0)):
                steady_states.append((val,vec))
            # for each element must be either greater OR close to 0

    try:
        [steady_states] = steady_states
        steady_states = steady_states[1]
        
    except Exception as e:
        print Warning("Multiple Steady States")
        #print "matrix"
        #for m in matrix:
        #    print m
        #print "ssds"
        #for i in steady_states:
        #    print i
        #print ""
        return np.array(normalized([n.real for n in steady_states[0][1]]))
        raise e

    return np.array(normalized([n.real for n in steady_states]))

def mm_to_limit_mcp(payoff,pop_size):
    """
    this takes a TxT matrix that gives the payoff to t1 when facing t2 into a
    matchup->composition->payoff matrix, which goes from the
    index of an ordered pair of type indices AND
    the index of a population composition,
    to a vector of payoffs indexed by type
    """

    type_count = len(payoff)
    liminal_pops = [np.array((i, pop_size-i)) for i in range(1,pop_size)]
    type_indices_matchups = list(combinations(range(type_count), 2))

    I = np.identity(type_count)

    mcp_lists= []
    for types in type_indices_matchups:
        payoffs = []
        type_indices = np.array([True if i in tuple(types) else False for i in range(type_count)])
        for counts in liminal_pops:
            pay = np.array([np.dot(counts-I[t][type_indices],payoff[t][type_indices])/(pop_size-1) for t in types])
            payoffs.append(pay)
            
        mcp_lists.append(payoffs)
        
    mcp_matrix = np.array(mcp_lists)
    return mcp_matrix

def avg_payoff_per_type_from_sim(sim_data):
    running_fitness = 0
    running_interactions = 0
    fitness_per_round = []
    pop_size = max(sim_data['id'].unique())+1

    for r, r_d in sim_data.groupby('round'):
        # Skip the zeroth round since there is no action in that round.
        if r == 0: continue
        
        fitness = []
        interactions = []
        for i, (t, t_d) in enumerate(r_d.groupby('type')):
            fitness.append(t_d['fitness'].mean())
            interactions.append(t_d['interactions'].mean())

        running_fitness += np.array(fitness)
        running_interactions += np.array(interactions)

        # # Depending on whether an interaction is a round or entire play through a Circular is a round will depend on how these should be normalized. It should always normalize by the total number of interactions. 

        # fitness_per_round.append(np.array(running_fitness)/(r*(pop_size-1)))
        #fitness_per_round.append(np.array(running_fitness / r))
        fitness_per_round.append(running_fitness/running_interactions)

    return fitness_per_round

@memoized
def indirect_simulator(player_types, *args, **kwargs):
    print player_types;assert 0
    sim_data = matchup(per_round = True, player_types = player_types, *args, **kwargs)
    fitness_per_round = avg_payoff_per_type_from_sim(sim_data)
    return fitness_per_round

def indirect_simulator_from_dict(d):
    return indirect_simulator(**d)

@memoized
def sim_to_limit_rmcp(player_types, pop_size, rounds, parallelized = True, **kwargs):
    pool = Pool(8)

    assert player_types == sorted(player_types)

    # produce all elements along the edges of the population simplex
    # does not include the homogeneous populations at the vertices
    # ordered populations, going from (1,pop_size-1) to (pop_size-1,1)
    populations = [(i, pop_size-i) for i in range(1, pop_size)]

    # all the pairings of two player_types, note these are combinations
    matchups = list(combinations(player_types, 2))

    matchup_pop_dicts = [dict(player_types = zip(*pop_pair), rounds = rounds, **kwargs) for pop_pair in product(matchups, populations)]

    # make a mapping from matchup to list of lists of payoffs
    # the first level is ordered by partitions
    # the second layer is ordered by rounds
    # payoffs = map(indirect_simulator_from_dict, matchup_pop_dicts)
    if parallelized == True:
        payoffs = pool.map(indirect_simulator_from_dict, matchup_pop_dicts)
    elif parallelized == False:
        payoffs = map(indirect_simulator_from_dict, matchup_pop_dicts)

    # Unpack the data into a giant matrix
    rmcp = np.zeros((rounds, len(matchups), len(populations), 2))
    for ((m,matchup), c), p in zip(product(enumerate(matchups), range(len(populations))), payoffs):
        rmcp[:, m, c, :] = np.array(p)

    return rmcp

@memoized
def ana_to_limit_rmcp(player_types, pop_size, rounds, **kwargs):
    payoffs = matchup_matrix_per_round(player_types = player_types, max_rounds = rounds, **kwargs)
    rmcp = np.array([mm_to_limit_mcp(payoff,pop_size) for r,payoff in payoffs])
    return rmcp

def mcp_to_ssd(mcp,s):
    transition = mcp_to_invasion(np.exp(s*mcp))
    return steady_state(transition)

def mcp_to_invasion(mcp, type_count):
    """
    type_indices_matchups, a list of the matchups where instead of the type, it's index is in it's position
    mcp_matrix is a matchup x count x payoff, matrix

    a matchup is the combination of agent types involved,
    a count is the ordered population composition
    a payoff is the vector of payoffs to each of the participating agents for that matchup and population composition
    """
    

    type_indices_matchups = list(combinations(range(type_count), 2))
    transition = np.zeros((type_count,)*2)

    for matchup, payoff_by_parts in izip(type_indices_matchups, mcp):
        a,b = matchup
        

        pbp = payoff_by_parts

        ratios = np.divide(pbp[:,1],pbp[:,0])

        ab = ratios
        ba = list(reversed(np.reciprocal(ratios)))

        trans_fn = lambda seq: 1/((type_count-1)*(1+np.sum(np.cumprod(seq))))
        transition[a,b] = trans_fn(ab)
        transition[b,a] = trans_fn(ba)

    for i in range(type_count):
        transition[i,i] = 1-np.sum(transition[:,i])
        try:
            np.testing.assert_approx_equal(np.sum(transition[:,i]),1)
        except:
            print transition[:,i]
            print np.sum(transition[:,i])
            raise

    return transition

def limit_analysis(player_types, s, direct = False, **kwargs):
    if direct:
        rmcp = ana_to_limit_rmcp(player_types, **kwargs)
    else:
        #rmcp = sim_to_limit_rmcp(player_types, **kwargs)
        rmcp = sim_to_rmcp(player_types,analysis_type = 'limit', **kwargs)

    rmcp = np.exp(s * rmcp)
    ssds = []
    for mcp in rmcp:
        ssds.append(steady_state(mcp_to_invasion(mcp, len(player_types))))

    return ssds


####
# Testing
####

def test_complete_limit():
    matrix = np.array([[0, .1], [-.1, 2]])
    s = 1; N = 100; mu = 0.0001
    np.testing.assert_allclose(
        complete_analysis(matrix, N, s, mu=mu),
        limit_analysis(matrix, N, s),
        rtol = 0.001, atol=0.001)

@memoized
def evo_analysis(player_types, analysis_type = 'limit', *args, **kwargs):
    type_to_index = dict(map(reversed, enumerate(sorted(player_types))))
    original_order = np.array([type_to_index[t] for t in player_types])
    player_types = sorted(player_types)

    if analysis_type == 'complete':
        ssds = complete_analysis(player_types = player_types, *args,**kwargs)
    elif analysis_type == 'limit':
        ssds = limit_analysis(player_types = player_types, *args,**kwargs)

    return np.array(ssds)[:,original_order]


def matchups_and_populations(player_types, pop_size, analysis_type):
    """
    adding a matchups/populations pair to this function makes 'sim_to_rmcp' work automagically
    matchups are combinations of player_types,
    populations are permutations of integers that add up to pop_size,
    the number of these summands must be equal to the number of player_types in a combination
    """

    type_count = len(player_types)
    if analysis_type == 'limit':
        # all the pairings of two player_types, note these are combinations
        matchups = list(combinations(player_types, 2))
       
        # produce all elements along the edges of the population simplex
        # does not include the homogeneous populations at the vertices
        # ordered populations, going from (1,pop_size-1) to (pop_size-1,1)
        # note that the case (0,n) and (n,0) are not considered in the limit
        populations = [(i, pop_size-i) for i in range(1, pop_size)]

    if analysis_type == 'complete':
        # in the complete analysis there is only a single matchup, which is everyone
        matchups = [player_types]

        # the populations for the complete case are literally every partitioning
        # of 'pop_size' into 'type_count' numbers that add up to it including 0s
        #
        # NOTE:
        # it seems 'all_partitions' returns some repeats, so we use 'set' to fix it
        # we use 'sorted' to guarantee an ordering from set
        populations = list(sorted(set(all_partitions(pop_size,type_count))))
    return matchups, populations


@memoized
def simulation(player_types, *args, **kwargs):
    
    original_type_count = len(player_types)
    counted_players = [p[1] != 0 for p in player_types]
    new_player_types = [p for p in player_types if p[1] != 0]
    #print new_player_types
    sim_data = matchup(per_round = True, player_types = new_player_types, *args, **kwargs)
    raw_fitness_per_round = avg_payoff_per_type_from_sim(sim_data)

    expanded_fitness_per_round = []
    for fitness in raw_fitness_per_round:
        new_fitness = np.zeros(original_type_count)
        new_fitness[counted_players] = fitness
        expanded_fitness_per_round.append(new_fitness)
    return expanded_fitness_per_round


def simulation_from_dict(d):
    return simulation(**d)

@memoized
def sim_to_rmcp(player_types, pop_size, analysis_type = 'limit', parallelized = True, **kwargs):
    assert player_types == sorted(player_types)
    matchups, populations = matchups_and_populations(player_types, pop_size, analysis_type)
    matchup_pop_dicts = [dict(player_types = zip(*pop_pair), **kwargs) for pop_pair in product(matchups, populations)]

    # make a mapping from matchup to list of lists of payoffs
    # the first level is ordered by partitions
    # the second layer is ordered by rounds
    # payoffs = map(indirect_simulator_from_dict, matchup_pop_dicts)
    if parallelized == True:
        pool = Pool(8)
        payoffs = pool.map(simulation_from_dict, matchup_pop_dicts)
    elif parallelized == False:
        payoffs = map(simulation_from_dict, matchup_pop_dicts)


    assert not (analysis_type == 'limit') or (len(payoffs[0][0]) == 2)

    # Unpack the data into a giant matrix
    true_rounds = max(map(len,payoffs))
    rmcp = np.zeros((true_rounds, len(matchups), len(populations), len(payoffs[0][0])))
    for ((m,matchup), c), p in zip(product(enumerate(matchups), range(len(populations))), payoffs):
        rmcp[:, m, c, :] = np.array(p)

    return rmcp


def cp_to_transition(cp, partitions, pop_size,  mu = None, **kwargs):
    if mu == None:
        mu = .00000001
    testing = np.array(cp)
    testing.flatten()
    try:
        assert (testing>=0).all()
    except AssertionError:
        print "cp has negative"
        print testing
        print cp
        raise
    type_count = len(cp[0])
    I = np.identity(type_count)
    part_to_id = dict(map(reversed,enumerate(partitions)))
    partition_count = len(part_to_id)
    transition = np.zeros((partition_count,)*2)

    birth_death_pairs = list(permutations(xrange(type_count),2))
    for i,(payoff, pop) in enumerate(zip(cp, partitions)):
        node = np.array(pop)
        for b,d in birth_death_pairs:
            if pop[d] != 0:
                neighbor = pop + I[b] - I[d]
                death_odds = pop[d] / pop_size
                birth_odds = payoff[b] * (1-mu) + mu * (1 / type_count)
                transition[part_to_id[tuple(neighbor)],i] = death_odds * birth_odds

    for i in xrange(partition_count):
        rest = sum(transition[:,i])
        if rest>1:
            print transition[:,i]
            raise Warning('sum of outgoing weights is more than 1')
        transition[i,i] = 1-rest

    return transition


def ssd_to_expected_pop(ssd, partitions):
    type_count = len(partitions[0])
    pop_size = sum(partitions[0])
    pop_sum = np.zeros(type_count)

    for p,partition in zip(ssd, partitions):
        pop_sum += np.array(partition) * p

    return pop_sum / pop_size


def duels_to_rcp(duels, partitions, **kwargs):
    rcp = []
    type_count = len(duels[0][1][0])
    I = np.identity(type_count)
    for r, duel in duels:
        pop_to_payoff = []
        for pop in partitions:
            payoff = [np.dot(pop-I[t],duel[t]) for t in range(type_count)]
            #payoff = softmax(payoff,s)
            #payoff = [f if p!=0 else 0 for p,f in zip(pop,payoff)]
            pop_to_payoff.append(payoff)
        rcp.append(pop_to_payoff)
    return np.array(rcp)

def old_complete(payoff, pop_size, s, **kwargs):
    """
    calculates the steady state distribution over population compositions
    """
    type_count = len(payoff)
    transition = pop_transition_matrix(payoff, pop_size, s, **kwargs)
    ssd = steady_state(transition)

    pop_sum = np.zeros(type_count)
    for p,partition in zip(ssd, sorted(set(all_partitions(pop_size, type_count)))):
        pop_sum += np.array(partition) * p

    return pop_sum / pop_size

def complete_payoffs(player_types, rounds, pop_size, **kwargs):
    return matchup_matrix_per_round(player_types = player_types, max_rounds = rounds, **kwargs)

def complete_softmax(rcp,populations,s):
    ###
    ### convert payoffs into fitnesses
    ###
    # first make a mask with the same dimensions as the rcp
    # it is 'True' wherever 'populations' is 0
    rounds = len(rcp)
    non_players = np.array((np.array(populations) == 0,)*rounds)
    # exponentiate the rcp
    expd = np.exp(s*rcp)
    expd[non_players] = 0
    sumd = np.sum(expd,axis = 2)
    ret = expd/sumd[:,:,None]



    # zero out the non-player entries
    # this is necessary, because non-players have fitness = 0

    return ret

def complete_analysis(player_types, s, direct = False, mu = None, **kwargs):

    pop_size = kwargs['pop_size']
    _, populations = matchups_and_populations(player_types, pop_size, analysis_type = 'complete')

    if direct:
        duels = complete_payoffs(player_types, **kwargs)
        rcp = duels_to_rcp(duels, populations, **kwargs)

    else:
        rmcp = sim_to_rmcp(player_types, analysis_type = 'complete', **kwargs)
        rcp = np.array([mcp[0] for mcp in rmcp])


    softmax_rcp = complete_softmax(rcp,populations,s)


    transitions = [cp_to_transition(cp, populations, mu = mu, **kwargs) for cp in softmax_rcp]

    expected_pops = []
    for transition in transitions:
        ssd = steady_state(transition)
        expected_pops.append(ssd_to_expected_pop(ssd, populations))

    return expected_pops

if __name__ == "__main__":
    pass
