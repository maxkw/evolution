from collections import Counter, defaultdict
from itertools import product, permutations, combinations, filterfalse
from utils import normalized, softmax, excluding_keys
from math import factorial
import numpy as np
from functools import partial
from utils import memory, softmax
from multiprocessing import Pool
from experiments import matchup, matchup_matrix_per_round
from copy import copy
from joblib import Parallel, delayed
import params
from tqdm import tqdm
from iteround import saferound


###
# Limit
###


def mm_to_limit_mcp(payoff, pop_size):
    """
    this takes a TxT matrix that gives the payoff to t1 when facing t2 into a
    matchup->composition->payoff matrix, which goes from the
    index of an ordered pair of type indices AND
    the index of a population composition,
    to a vector of payoffs indexed by type
    """

    type_count = len(payoff)

    type_indices_matchups, liminal_pops = matchups_and_populations(
        range(type_count), pop_size, "limit"
    )

    mcp_lists = []
    for types in type_indices_matchups:
        payoffs = []
        type_indices = np.isin(np.arange(type_count), types)

        for counts in liminal_pops:
            pay = list()

            for t in types:
                pay.append(
                    np.dot(
                        counts - np.identity(type_count)[t][type_indices],
                        payoff[t][type_indices],
                    )
                    / (pop_size - 1)
                )

            payoffs.append(np.array(pay))
        mcp_lists.append(payoffs)
    mcp_matrix = np.array(mcp_lists)
    
    return mcp_matrix


# @memory.cache
def ana_to_limit_rmcp(player_types, pop_size, rounds, **kwargs):
    payoffs = matchup_matrix_per_round(
        player_types=player_types, rounds=rounds, **kwargs
    )
    
    rmcp = np.array([mm_to_limit_mcp(payoff, pop_size) for r, payoff in payoffs])
    return rmcp, payoffs


def mcp_to_invasion(mcp, type_count):
    """
    type_indices_matchups, a list of the matchups where instead of the type, it's index is in it's position
    mcp_matrix is a matchup x count x payoff, matrix

    a matchup is the combination of agent types involved,
    a count is the ordered population composition
    a payoff is the vector of payoffs to each of the participating agents for that matchup and population composition
    """

    type_indices_matchups = list(combinations(list(range(type_count)), 2))
    transition = np.zeros((type_count,) * 2)

    for matchup, payoff_by_parts in zip(type_indices_matchups, mcp):
        a, b = matchup

        pbp = payoff_by_parts

        ratios = np.divide(pbp[:, 1], pbp[:, 0])

        ab = ratios
        ba = list(reversed(np.reciprocal(ratios)))

        trans_fn = lambda seq: 1 / ((type_count - 1) * (1 + np.sum(np.cumprod(seq))))
        
        transition[a, b] = trans_fn(ab)
        transition[b, a] = trans_fn(ba)

    for i in range(type_count):
        transition[i, i] = 1 - np.sum(transition[:, i])
        try:
            np.testing.assert_approx_equal(np.sum(transition[:, i]), 1)
        except:
            print("Outgoing transitions from %s don't add up to 1" % i)
            print(transition[:, i])
            print(np.sum(transition[:, i]))
            raise

    return transition


def limit_analysis(player_types, s, direct=False, **kwargs):
    if direct:
        rmcp, payoffs = ana_to_limit_rmcp(player_types, **kwargs)
    else:
        # assert kwargs['game'].N_players == 2
        rmcp = sim_to_mcp(player_types, analysis_type="limit", **kwargs)
        payoffs = None

    ssds = []
    e_rmcp = np.exp(s * rmcp)

    if len(e_rmcp.shape) == 4:
        # This is for the case that it is calculated per_round so need to
        # compute steady state for each round independently.
        for mcp in e_rmcp:
            ssd = steady_state(mcp_to_invasion(mcp, len(player_types)))
            ssds.append(ssd)

    else:
        ssds.append(steady_state(mcp_to_invasion(e_rmcp, len(player_types))))

    return np.array(ssds), payoffs


###
# Complete
###


def cp_to_transition(cp, populations, pop_size, mu=None, **kwargs):
    if mu == None:
        mu = 0.001

    testing = np.array(cp)
    testing.flatten()
    assert (testing >= 0).all()

    type_count = len(cp[0])
    I = np.identity(type_count).astype(int)
    part_to_id = dict(list(map(reversed, enumerate(populations))))
    partition_count = len(part_to_id)
    transition = np.zeros((partition_count,) * 2)

    birth_death_pairs = list(permutations(range(type_count), 2))
    for i, (payoff, pop) in enumerate(zip(cp, populations)):
        node = np.array(pop)
        for b, d in birth_death_pairs:
            if pop[d] != 0:
                neighbor = pop + I[b] - I[d]
                death_odds = pop[d] / pop_size
                birth_odds = payoff[b] * (1 - mu) + mu * (1 / type_count)
                transition[part_to_id[tuple(neighbor)], i] = death_odds * birth_odds

    for i in range(partition_count):
        rest = sum(transition[:, i])
        if rest > 1:
            print(transition[:, i])
            raise Warning("sum of outgoing weights is more than 1")
        transition[i, i] = 1 - rest

    return transition


# def complete_payoffs(player_types, rounds, pop_size, **kwargs):
#     return matchup_matrix_per_round(player_types=player_types, rounds=rounds, **kwargs)


def duels_to_rcp(duels, partitions, **kwargs):
    rcp = []
    pop_size = sum(partitions[0])
    type_count = len(duels[0][1][0])
    I = np.identity(type_count)
    for r, duel in duels:
        pop_to_payoff = []
        for pop in partitions:
            payoff = [
                np.dot(pop - I[t], duel[t]) / (pop_size - 1) for t in range(type_count)
            ]
            # payoff = softmax(payoff,s)
            # payoff = [f if p!=0 else 0 for p,f in zip(pop,payoff)]
            pop_to_payoff.append(payoff)
        rcp.append(pop_to_payoff)
    return np.array(rcp)


def complete_analysis(player_types, s, direct=False, mu=None, **kwargs):
    pop_size = kwargs["pop_size"]
    _, populations = matchups_and_populations(
        player_types, pop_size, analysis_type="complete"
    )

    if direct:
        assert 0
        # Broken since removed `rounds` as part of the analysis pipeline
        # duels = complete_payoffs(player_types, **kwargs)
        # rcp = duels_to_rcp(duels, populations, **kwargs)

    else:
        mcp = sim_to_mcp(player_types, analysis_type="complete", **kwargs)
        # There are no `matchups` for complete analysis since all
        # agent types are in play
        cp = mcp[0]

    softmax_cp = np.zeros_like(cp)

    # For populations that don't have players of a certain type their cp will be NaN so we need to mask those out
    active_players = np.isnan(cp) == False
    for p in range(len(populations)):
        softmax_cp[p][active_players[p]] = softmax(cp[p][active_players[p]], s)

    transition = cp_to_transition(softmax_cp, populations, mu=mu, **kwargs)

    # Steady state prevalence of each node
    ssd = steady_state(transition)
    
    total_payoff = 0
    # Get the expected payoff for each population
    for i in range(len(populations)):
        for j in range(len(player_types)):
            if active_players[i][j]:
                total_payoff += ssd[i] * cp[i][j] * populations[i][j]
    
    total_payoff = total_payoff / pop_size 

    # Expected population composition
    pop_sum = np.zeros(len(player_types))
    for p, pop in zip(ssd, populations):
        pop_sum += p * np.array(pop)

    expected_pop = pop_sum / pop_size
    
    return [expected_pop], total_payoff


###
# Common code
##


def steady_state(matrix):
    for i, c in enumerate(matrix.T):
        try:
            np.testing.assert_approx_equal(np.sum(c), 1)
            assert all(c >= 0)
        except:
            print("has some negative?", c)
            print(matrix)
            raise

    vals, vecs = np.linalg.eig(matrix)

    def negative_vec(vec):
        return all([i < 0 or np.isclose(i, 0) for i in vec])

    steady_states = []
    for val, vec in sorted(zip(vals, vecs.T), key=lambda a: a[0]):

        if np.isclose(val, 1):

            if negative_vec(vec):
                steady_states.append((val, np.absolute(vec)))
            # for each element must be either greater OR close to 0
            elif all(np.logical_or(np.isclose(vec, 0), vec >= 0)):
                steady_states.append((val, vec))

    try:
        [steady_states] = steady_states
        steady_states = steady_states[1]

    except Exception as e:
        print(Warning("Multiple Steady States"))
        return steady_states[0][1]
        raise e

    return np.array(normalized([n.real for n in steady_states]))


def avg_payoff_per_type_from_sim(sim_data, player_types, cog_cost, game=None, **kwargs):
    # For now, do not implement cognitive costs. Its not clear how
    # they should be applied. Should they be done on a per-interaction
    # per-type basis?
    assert cog_cost == 0

    types, counts = list(zip(*player_types))
    type_to_index = dict(list(map(reversed, enumerate(types))))
    pop_size = sum(counts)

    running_fitness = np.zeros(len(types))
    running_interactions = np.zeros(len(types))
    means = sim_data.groupby("type")[["interactions", "fitness"]].mean()

    for t, c in player_types:
        if c == 0:
            running_fitness[type_to_index[t]] = np.nan
            running_interactions[type_to_index[t]] = np.nan

    for t, t_id in type_to_index.items():
        # Skip types that weren't participating
        if np.isnan(running_fitness[t_id]):
            continue

        t = str(t)

        running_interactions[t_id] = means.loc[t, "interactions"]
        running_fitness[t_id] = (means.loc[t, "fitness"])

    return running_fitness / running_interactions


def simulation(player_types, cog_cost=0, *args, **kwargs):
    # types, _ = list(zip(*player_types))
    active_players = [p for p in player_types if p[1] != 0]

    # # Only one active player. No reason to run any sims.
    # if len(active_players) == 1:
    #     fitness = list()
    #     for p, c in player_types:
    #         if c == 0:
    #             fitness.append(np.nan)
    #         else:
    #             fitness.append(1)
    #     return np.array(fitness)

    sim_data = matchup(player_types=active_players, *args, **kwargs)

    fitness_per_interaction = avg_payoff_per_type_from_sim(
        **dict(kwargs, **dict(sim_data=sim_data, cog_cost=cog_cost)),
        player_types=player_types,
    )

    return fitness_per_interaction


def matchups_and_populations(player_types, pop_size, analysis_type):
    """
    adding a matchups/populations pair to this function makes 'sim_to_mcp' work automagically
    matchups are combinations of player_types,
    populations are permutations of integers that add up to pop_size,
    the number of these summands must be equal to the number of player_types in a combination
    """

    type_count = len(player_types)
    if analysis_type == "limit":
        # all the pairings of two player_types, note these are combinations
        matchups = list(combinations(player_types, 2))

        # produce all elements along the edges of the population simplex
        # does not include the homogeneous populations at the vertices
        # ordered populations, going from (1,pop_size-1) to (pop_size-1,1)
        # note that the case (0,n) and (n,0) are not considered in the limit
        populations = [(i, pop_size - i) for i in range(1, pop_size)]

    if analysis_type == "complete":
        # in the complete analysis there is only a single matchup, which is everyone
        matchups = [player_types]
        populations = agent_pop_simplex(pop_size, type_count)
    return matchups, populations


def agent_pop_simplex(num_agents, types):
    """Returns a list of partitions of `num_agents` divided into `types`
    bins. This returns all nodes of the simplex with edge length
    `num_agents` and dimension `types-1`

    e.g., agent_pop_simplex(3, 3)=[[3,0,0], [2,1,0],
    [1,2,0],...,[0,0,3]]"""

    partitions = list()

    def _agent_pop_simplex(allocated_list, type_index, num_agents_left):
        if num_agents_left == 0:
            partitions.append(tuple(allocated_list))
            return

        for num_to_allocate in range(num_agents_left + 1):
            if type_index < types:
                new_list = list(allocated_list)
                new_list[type_index] = num_to_allocate
                _agent_pop_simplex(
                    new_list, type_index + 1, num_agents_left - num_to_allocate
                )

    _agent_pop_simplex([0] * types, 0, num_agents)
    return partitions


def sim_to_mcp(player_types, pop_size, analysis_type="limit", **kwargs):
    matchups, populations = matchups_and_populations(
        player_types, pop_size, analysis_type
    )
    matchup_pop_dicts = [
        dict(player_types=list(zip(*pop_pair)), **kwargs)
        for pop_pair in product(matchups, populations)
    ]

    # Need to turn off memoization here OR group them into a
    # single file since this function will make way too many files
    # (one for each parameter). Instead need to cache the output of
    # THIS function.
    # payoffs = Parallel(n_jobs=params.n_jobs)(delayed(simulation)(**pop_dict) for pop_dict in tqdm(matchup_pop_dicts, disable=params.disable_tqdm))
    payoffs = Parallel(n_jobs=params.n_jobs)(
        delayed(simulation)(**pop_dict) for pop_dict in matchup_pop_dicts
    )

    assert not (analysis_type == "limit") or (len(payoffs[0]) == 2)

    # Unpack the data into a giant matrix
    matchup_list = list(product(enumerate(matchups), list(range(len(populations)))))
    mcp = np.zeros((len(matchups), len(populations), len(payoffs[0])))
    for ((m, matchup), c), p in zip(matchup_list, payoffs):
        mcp[m, c, :] = p

    return mcp


# @memoize
def evo_analysis(player_types, analysis_type="limit", direct=True, *args, **kwargs):
    # Canonical ordering so that the cache will hit
    # player_types = sorted(player_types, key=lambda x: x.__name__)

    # Sorting for so that the cache will still hit for random orderings
    type_to_index = dict(list(map(reversed, enumerate(sorted(player_types)))))
    original_order = np.array([type_to_index[t] for t in player_types])

    # If playing the direct-reciprocity game then use the direct
    # method where we don't have to compute the payoffs for each
    # population composition.
    direct_games = ["direct", "direct_seq"]
    if kwargs["game"] in direct_games and direct:
        direct = True
    else:
        direct = False

    if analysis_type == "complete":
        ssds, payoffs = complete_analysis(
            player_types=player_types, direct=direct, *args, **kwargs
        )
    elif analysis_type == "limit":
        ssds, payoffs = limit_analysis(player_types=player_types, direct=direct, *args, **kwargs)

    return np.array(ssds), payoffs
