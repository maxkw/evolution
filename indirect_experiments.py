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
from utils import softmax_utility,softmax,normalized,memoized
from functools import partial
from evolve import limit_param_plot, limit_param_plot
from steady_state import all_partitions,steady_state
from itertools import permutations,product,imap,chain,repeat, izip,starmap,combinations
from math import factorial
from multiprocessing import Pool
from games import RepeatedPrisonersTournament

def logspace(start = .001,stop = 1, samples=10):
    mult = (np.log(stop)-np.log(start))/np.log(10)
    plus = np.log(start)/np.log(10)
    return np.array([0]+list(np.power(10,np.linspace(0,1,samples)*mult+plus)))

@multi_call(unordered = ['agent_types'], verbose = 3)
def indirect_game_ratios(WA_ratio = .5, pop_size = 30, *args, **kwargs):

    n = pop_size
    c = int(pop_size*WA_ratio)
    #rounds = proportion_of_matches
    type_count= dict([(WeAgent,c), (AllD,n-c)])
    return indirect_game(type_count,*args,**kwargs)

@multi_call(unordered = ['agent_types'], verbose = 0)
@experiment(unpack = 'record', trials = 100,verbose =3)
def indirect_game(type_count_pairs, observability = 1, rounds = 50, tremble= 0, **kwargs):
    types = []
    type_to_count = dict(type_count_pairs)
    for t,count in type_to_count.iteritems():
        types += [t]*count

    params = default_params(**kwargs)
    params.update(kwargs)
    params['games'] = g = RepeatedPrisonersTournament(rounds = rounds, observability = observability, tremble = tremble,**kwargs)#IndirectReciprocity(rounds, observability, tremble = tremble, **kwargs)
    genomes = [default_genome(agent_type = t, **params) for t in types]

    world = World(params,genomes)

    fitness, history = world.run()

    fitness_per_type = defaultdict(int)
    
    records =[]
    fitness_per_type = defaultdict(int)
    if not True:#kwargs.get('per_round', False):
        for t,f in zip(types,fitness):
            fitness_per_type[t]+=f

        for t in type_to_count:
            try:
                fitness_per_type[t] /= type_to_count[t]
            except ZeroDivisionError:
                assert fitness_per_type[t] == 0
    
        for t,f in fitness_per_type.iteritems():
            records.append({"type":t,
                            "fitness":f})
    else:
        avg_fitness_per_type = defaultdict(int)
        for event in history:
            fitness = event['payoff']
            r = event['round']
            
            for t,f in zip(types,fitness):
                fitness_per_type[t]=f
            
            for t in type_to_count:
                try:
                    avg_fitness_per_type[t] = fitness_per_type[t] / type_to_count[t]
                except ZeroDivisionError:
                    assert fitness_per_type[t] == 0
                    
            for t,f in avg_fitness_per_type.iteritems():
                records.append({
                    "round":r,
                    "type":t,
                    "fitness":f})

    #fitness_ratio = fitness_per_type[WeAgent]/fitness_per_type[AllD]
    #return [{"fitness_ratio":fitness_ratio}]
    return records
@memoized
def indirect_simulator(types, per_round = False, *args,**kwargs):
    data = indirect_game(*args,**kwargs)
    if not per_round:
        fitness = defaultdict(int)
        for r,r_d in data.groupby('round'):
            for t,d in r_d.groupby('type'):
                fitness[t] += d.mean()['fitness']
        return [fitness[t] for t in types]
    else:
        fitness_per_round = {}
        fitness = defaultdict(int)
        #print  types
        for r,r_d in data.groupby('round'):
            for t,d in r_d.groupby('type'):
                fitness[t] = d.mean()['fitness']
            #print fitness
            fitness_per_round[r] = [fitness[t] for t in types]
        return fitness_per_round

def indirect_simulator_from_dict(d):
    return indirect_simulator(**d)

@plotter(indirect_game_ratios, plot_exclusive_args = ['data'])
def relative_fitness_vs_proportion(data = [], *args, **kwargs):
    print data[data['observability']==.5]

    #records = []
    #for (WA_ratio,observability), df in data.groupby(['WA_ratio','observability']):
    #    fitness_ratio = df[df['type'] == WeAgent].mean()['fitness']/df[df['type'] == AllD].mean()['fitness']
    #    records.append({"WA_ratio": WA_ratio,
    #                    "observability": observability,
    #                    "fitness_ratio": fitness_ratio})
    #data = pd.DataFrame(records)
    data = data[data['type'] == WeAgent]
    g = sns.factorplot(data = data, x = 'WA_ratio', y = 'fitness', hue = 'observability', kind = 'point')

    g.set(ylim = (0,1))



    

def sim_complete_analysis(simulator, types,  pop_size, s, **kwargs):
    """
    calculates the steady state distribution over population compositions
    """
    type_count = len(types)
    transition = sim_pop_transition_matrix(simulator, types, pop_size, s, **kwargs)
    ssd = steady_state(transition)

    pop_sum = np.zeros(type_count)
    for p, partition in zip(ssd, sorted(set(all_partitions(pop_size, type_count)))):
        pop_sum += np.array(partition) * p

    return pop_sum / pop_size


def sim_pop_transition_matrix(simulator, types, pop_size, s, mu = .0001, **kwargs):
    """
    returns a matrix that returns the probability of transitioning from one population composition
    to another

    the index of a population is it's position as given by the 'all_partitions()' function
    """
    type_count = len(types)
    I = np.identity(type_count)
    part_to_id = dict(map(reversed,enumerate(sorted(set(all_partitions(pop_size,type_count))))))
    partition_count = len(part_to_id)
    transition = np.zeros((partition_count,)*2)
    
    for pop, i in part_to_id.iteritems():
        fitnesses = softmax(simulator(type_count_pairs = tuple(zip(types, pop))), s)
        
        for t in range(type_count):
            if pop[t] == 0:
                fitnesses[t] = 0
                
        fitnesses = normalized(fitnesses)
        
        for b, d in permutations(xrange(type_count), 2):
            if pop[d] != 0:
                neighbor = pop+I[b] - I[d]
                death_odds = pop[d] / pop_size
                birth_odds = fitnesses[b] * (1-mu) + mu * (1 / type_count)

                transition[part_to_id[tuple(neighbor)], i] = death_odds * birth_odds
                

    for i in xrange(partition_count):
        transition[i,i] = 1 - sum(transition[:,i])

    return transition

def sim_complete_analysis_per_round(types, pop_size, s, rounds, **kwargs):
    type_count = len(types)
    partitions = set(all_partitions(pop_size,type_count))
    
    part_to_id = dict(map(reversed,enumerate(sorted(partitions))))
    rounds_list = range(1,rounds+1)

    #partition_to_fitness_per_round = {}#p: {r:softmax(f,s) for r,f in simulator(type_count_pairs = tuple(zip(types,p))).iteritems()}
    #for p in partitions:
    #    partition_to_fitness_per_round[p] = fpr = {r:softmax(f,s)
    #                                         for r,f in simulator(type_count_pairs = tuple(zip(types,p))).iteritems()}
    pool = Pool(8)
    def part_to_argdict(part):
        return dict(type_count_pairs = tuple(zip(types,part)), types = types, rounds = rounds, trials = 200, per_round = True, **kwargs)
    partition_to_round_to_fitness = zip(partitions, pool.map(indirect_simulator_from_dict, map(part_to_argdict,partitions)))
    
    partition_to_fitness_per_round = {}
    for p,rtf in partition_to_round_to_fitness:
        partition_to_fitness_per_round[p] = {r:softmax(f,s) for r,f in rtf.iteritems()}
  
    expected_pop_per_round = {}
    part_to_fitness = {}
    for r in rounds_list:
        for partition in partitions:
            try:

                part_to_fitness[partition] = partition_to_fitness_per_round[partition][r]
            except:
                #print partition
                #print r
                #print partition_to_fitness_per_round[partition]
                raise
        transition_matrix = fitnesses_to_transition_matrix(part_to_fitness,types,pop_size)
        ssd = steady_state(transition_matrix)

        pop_sum = np.zeros(type_count)
        for p, partition in zip(ssd, sorted(partitions)):
            pop_sum += np.array(partition) * p

        expected_pop_per_round[r] = pop_sum / pop_size
    return expected_pop_per_round

def fitnesses_to_transition_matrix(pop_to_payoff, types, pop_size,mu = .001):
    """
    returns a matrix that returns the probability of transitioning from one population composition
    to another

    the index of a population is it's position as given by the 'all_partitions()' function
    """
    type_count = len(types)
    I = np.identity(type_count)

    #print pop_to_payoff
    part_to_id = dict(map(reversed,enumerate(sorted(pop_to_payoff.keys()))))
    partition_count = len(part_to_id)
    transition = np.zeros((partition_count,)*2)
    
    for pop, i in part_to_id.iteritems():
        fitnesses = pop_to_payoff[pop]
        #for t in range(type_count):
        #     if pop[t] == 0:
        #         fitnesses[t] = 0
        #fitnesses = normalized(fitnesses)
        for b, d in permutations(xrange(type_count), 2):
            if pop[d] != 0:
                neighbor = pop+I[b] - I[d]
                death_odds = pop[d] / pop_size
                birth_odds = fitnesses[b] * (1-mu) + mu * (1 / type_count)
                transition[part_to_id[tuple(int(i) for i in neighbor)], i] = death_odds * birth_odds
                

    for i in xrange(partition_count):
        transition[i,i] = 1 - sum(transition[:,i])

    return transition


def indirect_analysis(types, pop_size = 100, s = 1, **kwargs):
    sim_conditions = dict(types = types, **kwargs)
    simulator = partial(indirect_simulator, **sim_conditions)
    return sim_complete_analysis(simulator, types, pop_size = pop_size, s = s, **kwargs)

def indirect_analysis_per_round(types, rounds,  pop_size = 100, s = 1, **kwargs):
    #sim_conditions = dict(types = types, rounds = rounds, **kwargs)
    #simulator = partial(indirect_simulator, per_round = True, **sim_conditions)
    return sim_complete_analysis_per_round(types, rounds = rounds, pop_size = pop_size, s = s, **kwargs)

def test_indirect_analysis():
    from agents import WeAgent,AllD,AllC
    #print 
    types = (WeAgent(agent_types = ('self',AllD), beta = 3, RA_prior = .5),AllD)
    return indirect_analysis_per_round(types = types, pop_size = 10, s = 1, observability = 1, rounds = 10, tremble = 0, trials = 10)

@experiment(unpack = 'record', memoize = False, verbose = 3)
def sim_ssd_v_param_dispatch(param, player_types, analysis = 'limit', **kwargs):
    
    if param == "RA_prior":
        Xs = np.linspace(0,1,21)[1:-1]
    elif param == "beta":
        Xs = logspace(.5,6,11)
    elif param == 'pop_size':
        Xs = np.unique(np.geomspace(2, 2**10, 200, dtype=int))
    elif param == 's':
        Xs = logspace(start = .001, stop = 15, samples = 100)
    elif param == 'bc':
        Xs = [1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
    elif param == "rounds":
        if analysis == 'limit':
            return sim_ssd_limit_v_rounds(player_types,**kwargs)
        else:
            return sim_ssd_v_rounds(player_types,**kwargs)
    else:
        print param
        raise Exception("Unknown param provided")

    record = []
    for x in Xs:
        for t,p in zip(player_types, indirect_analysis(types = player_types, **dict(kwargs,**{param:x}))):
            record.append({
                param:x,
                "type":t.short_name("agent_types"),
                "proportion":p
            })
    return record

def sim_ssd_v_param(param, player_types, analysis = 'limit', **kwargs):
    if param == 'rounds':
        if analysis == 'limit':
            return sim_ssd_limit_v_rounds(player_types,**kwargs)
        else:
            return sim_ssd_v_rounds(player_types,**kwargs)
    else:
        return sim_ssd_v_param_dispatch(param,player_types, analysis,**kwargs)



@experiment(unpack = 'record', memoize = False, verbose = 3)
def sim_ssd_v_rounds(player_types, rounds, **kwargs):
    record = []
    rounds_to_expected_pop = indirect_analysis_per_round(types = player_types, rounds = rounds, **kwargs)
    for rounds in rounds_to_expected_pop:
        for t,p in zip(player_types,rounds_to_expected_pop[rounds]):
            record.append({
                'rounds':rounds,
                'type': t.short_name('agent_types'),
                'proportion':p
            })

    return record

@experiment(unpack = 'record', memoize = False, verbose = 3)
def sim_ssd_limit_v_rounds(player_types, rounds, **kwargs):
    record = []
    rounds_to_expected_pop = sim_limit_analysis(types = player_types, rounds = rounds, **kwargs)
    for r in rounds_to_expected_pop:
        for t,p in zip(player_types,rounds_to_expected_pop[r]):
            record.append({
                'rounds':r,
                'type': t.short_name('agent_types'),
                'proportion':p
            })

    return record

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


def sim_limit_analysis(types,pop_size,s,rounds,**kwargs):
    type_count = len(types)
    type_indices = range(type_count)
    rounds_list = range(1,rounds+1)

    #produce all elements along the edges of the population simplex
    #does not include the homogeneous populations at the vertices
    size = int(pop_size/2)# + pop_size%2
    liminal_pops = set(imap(tuple,chain(*imap(permutations,imap(chain, izip(xrange(1,size+1),xrange(pop_size-1,size,-1)),(repeat(0,type_count-2) for _ in xrange(pop_size+1)))))))



    #ordered populations, going from (1,pop_size-1) to (pop_size-1,1)
    liminal_pops = zip(xrange(1,pop_size),xrange(pop_size-1,0,-1))

    #all the pairings of two types, note these are combinations
    matchups = list(combinations(types,2))
    matchups_indices = list(combinations(type_indices,2))

    #pairings of the type ((matchup, ((type, pop), ...)), ...)
    #the order of these is implicitly given by liminal_pops
    #these are fed to the simulator
    types_pops_pairs = list(imap(lambda x:(x[0],tuple(zip(*x))),product(matchups,liminal_pops)))
    print types_pops_pairs

    #extract a list of matchups, these have repeats
    matchup_list,type_count_pairs = zip(*types_pops_pairs)


    def part_to_argdict(thing):
        
        types,type_count_pairs = thing
        return dict(type_count_pairs = type_count_pairs, types = types, rounds = rounds, trials = 200, per_round = True, **kwargs)
    pool = Pool(8)

    def dict_to_tuples(d):
        return (i[1] for i in sorted(d.iteritems(), key = lambda x: x[0]))

    #(matchup,(payoffs), ...)))
    matchup_rtf_pairs = izip(matchup_list, imap(dict_to_tuples,pool.map(indirect_simulator_from_dict, imap(part_to_argdict,types_pops_pairs))))

    #make a mapping from matchup to list of lists of payoffs
    #the first level is ordered by partitions
    #the second layer is ordered by rounds
    matchup_to_payoffs = defaultdict(list)
    for matchup,rtf in matchup_rtf_pairs:
        matchup_to_payoffs[matchup].append(rtf)

    #matchup now maps to a list ordered by rounds, within which lists are ordered by population
    for matchup, rtf in matchup_to_payoffs.iteritems():
        matchup_to_payoffs[matchup] = zip(*rtf)

    payoffs = rounds_matchup_partition_payoff = zip(*(matchup_to_payoffs[m] for m in matchups))

    try:
        assert len(payoffs) == rounds
    except:
        if len(payoffs) - rounds:
            payoffs = payoffs[1:]
        else:
            raise

    #payoffs are already ordered by partitions in liminal_pops
    rmpp = round_to_matchup_to_payoffs = {}

    #i want (round:matchup,pop,payoff)

    ssds = []
    for r,matchup_to_part_to_payoff in izip(rounds_list,payoffs):
        transition = np.zeros((type_count,)*2)
        for matchup, payoff_by_parts in izip(matchups_indices, matchup_to_part_to_payoff):
            a,b = matchup

            accum_ab = 1
            accum_ba = 1
            payoff_by_parts = [softmax(p,s) for p in payoff_by_parts]
            for p_ab, p_ba in izip(payoff_by_parts,reversed(payoff_by_parts)):

                accum_ab *= p_ab[1]/p_ab[0]
                accum_ab += 1

                accum_ba *= p_ba[0]/p_ba[1]
                accum_ba += 1

            transition[a,b] = 1/accum_ab
            transition[b,a] = 1/accum_ba

        for i in xrange(type_count):
            transition[i,i] = 1-np.sum(transition[:,i])
            try:
                np.testing.assert_approx_equal(np.sum(transition[:,i]),1)
            except:
                print transition[:,i]
                print np.sum(transition[:,i])
                raise
        ssds.append(steady_state(transition))

    return dict(enumerate(ssds))



    #partition_to_fitness_per_round = {}
    #for p,rtf in partition_to_round_to_fitness:
    ##    partition_to_fitness_per_round[p] = {r:softmax(f,s) for r,f in rtf.iteritems()}
   # 
    #for r in rounds_list:
    #    for a,b in combinations(xrange(type_count),2):
    #        pass
    #    pass


def test_sim_limit_analysis():
    types = (AllC,AllD)
    #sim_invasion_matrix(None,types,5,1,10)

    pop_size = 10
    params = dict(#param = 'rounds',
                  #experiment = sim_ssd_v_param,
                  types = types,
                  tremble = .05,
                  rounds = 50,
                  benefit = 3,
                  s = 1,
                  pop_size = pop_size)
    
    #limit_param_plot(analysis = 'limit', **params)
    #limit_param_plot(analysis = 'complete', **params)

    c = sim_complete_analysis_per_round(**params)
    l = sim_limit_analysis(**params)

    for r in xrange(1,50):
        print c[r],l[r]


if __name__ == "__main__":

    #for i in range(50):
        
    #    print i,len(set(all_partitions(i,3)))
    #image_contest()
    #sim_invasion_matrix(0,"ABCD",5,0,10)
    test_sim_limit_analysis()
    #print softmax([0,1,2],1)
    #print softmax([1,2],1)
