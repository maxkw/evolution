from __future__ import division
import pandas as pd
import seaborn as sns
from experiment_utils import multi_call,experiment,plotter,MultiArg,cplotter, memoize, apply_to_args
import numpy as np
from params import default_params,generate_proportional_genomes,default_genome
from indirect_reciprocity import World,ReciprocalAgent,SelfishAgent,AltruisticAgent,NiceReciprocalAgent,RationalAgent,gTFT,AllC,AllD,Pavlov, RandomAgent
from games import RepeatedPrisonersTournament,BinaryDictator,Repeated,PrivatelyObserved,Symmetric
from collections import defaultdict
from itertools import combinations_with_replacement, combinations
from itertools import permutations
from itertools import product,islice,cycle
import matplotlib.pyplot as plt
from numpy import array
from copy import copy,deepcopy
from utils import softmax_utility,issubclass
import operator
from fractions import gcd as binary_gcd
from fractions import Fraction

priors_for_RAvRA = map(tuple,map(sorted,combinations(np.linspace(.75,.25,3),2)))
diagonal_priors = [(n,n) for n in np.linspace(.75,.25,3)]

letter_to_id = dict(map(reversed,enumerate("ABCDEFGHIJK")))
letter_to_action = {"C":'give',"D":'keep'}

def gcd(*numbers):
    """Return the greatest common divisor of the given integers"""
    return reduce(binary_gcd, numbers)


def lcm(*numbers):
    """Return lowest common multiple."""
    def lcm(a, b):
        return (a * b) / gcd(a, b)
    return reduce(lcm, numbers, 1)

def justcaps(t):
    return filter(str.isupper,t.__name__)

#@memoize
@multi_call(unordered = ['agent_types'], twinned = ['player_types','priors','Ks'], verbose=3)
@experiment(unpack = 'dict', trials = 100, verbose = 3)
def binary_matchup(player_types = (NiceReciprocalAgent,NiceReciprocalAgent), priors = (.75, .75), Ks=(1,1), **kwargs):
    condition = dict(locals(),**kwargs)
    params = default_params(**condition)
    genomes = [default_genome(agent_type = t, RA_prior=p, RA_K = k, **condition) for t,p,k in zip(player_types,priors,Ks)]
    world = World(params,genomes)
    fitness,history = world.run()
    return {'fitness':fitness,
            'history':history,
            'p1_fitness':fitness[0]}

@multi_call(unordered = ['player_types','agent_types'], verbose=3)
@experiment(unpack = 'record', trials = 100, verbose = 3)
def matchup(player_types, **kwargs):
    #print "HIIIIIIIIIIIIIIIIIII\n\n\n"
    #assert False
    #assert len(player_types)==2
    condition = dict(locals(),**kwargs)
    params = default_params(**condition)
    genomes = [default_genome(agent_type = t, **condition) for t in player_types]
    world = World(params,genomes)
    fitness,history = world.run()

    beliefs = []
    for agent in world.agents:
        try:
            beliefs.append(agent.belief)
        except:
            beliefs.append(None)

    record = []
    if not kwargs.get('per_round',False):
        for t,f,b in zip(player_types,fitness,beliefs):
            record.append({"type":t,"fitness":f,'belief':b})
    else:
        for event in history:
            payoffs = event['payoff']
            ids = [agent.world_id for agent in event['players']]
            r = event['round']
            for t,a_id,p in zip(player_types,ids,payoffs):
                record.append({'type' : t,
                               'id' : a_id,
                               'round' : r,
                               'fitness' : p})
    return record

def matchup_grid(player_types,**kwargs):
    player_types = MultiArg(combinations_with_replacement(player_types,2))
    return matchup(player_types,**kwargs)

def matchup_matrix(player_types,**kwargs):
    condition = dict(locals(),**kwargs)
    params = default_params(**condition)

    data = matchup_grid(player_types,**kwargs)
    index = dict(map(reversed,enumerate(player_types)))
    payoffs = np.zeros((len(player_types),)*2)
    for combination in data['player_types'].unique():
        for matchup in permutations(combination):
            player,opponent = matchup
            p,o = tuple(index[t] for t in matchup)
            trials = data[(data['player_types']==combination) & (data['type']==player)]
            payoffs[p,o] = trials.mean()['fitness']

    # If the game has the rounds field defined (i.e., it is a repeated
    # game). Then divide by the number of rounds. Otherwise return the
    # payoffs unmodified.
    try:
        payoffs /= params['games'].rounds
    except:
        print Warning("matchup_matrix didn't find a round parameter in the game")
     
    return payoffs

def matchup_data_to_matrix(data):
    index = dict(map(reversed,enumerate(player_types)))
    payoffs = np.zeros((len(player_types),)*2)
    for combination in data['player_types'].unique():
        for matchup in permutations(combination):
            player,opponent = matchup
            p,o = tuple(index[t] for t in matchup)
            trials = data[(data['player_types']==combination) & (data['type']==player)]
            payoffs[p,o] = trials.mean()['fitness']

def matchup_matrix_per_round(player_types, max_rounds, **kwargs):
    condition = dict(locals(),**kwargs)
    params = default_params(**condition)

    all_data = matchup_grid(player_types, per_round = True, rounds = max_rounds, **kwargs)
    index = dict(map(reversed,enumerate(player_types)))
    payoffs_list = []
    payoffs = np.zeros((len(player_types),)*2)
    # FIXME: Why is this starting from 1?
    for r in range(1, max_rounds + 1):
        data = all_data[all_data['round']==r]
        for combination in data['player_types'].unique():
            for matchup in set(permutations(combination)):
                player,opponent = matchup
                p,o = tuple(index[t] for t in matchup)
                trials = data[(data['player_types']==combination) & (data['type']==player)]
                # import pdb; pdb.set_trace()
                payoffs[p,o] += trials.mean()['fitness']
        payoffs_list.append(copy(payoffs))

    for i,p in enumerate(payoffs_list,start=1):
        p/=i
    return list(enumerate(payoffs_list,start=1))



@plotter(matchup_grid, plot_exclusive_args = ['data'])
def matchup_plot(data = [],**kwargs):
    condition = dict(locals(),**kwargs)
    params = default_params(**condition)
    record = []
    for combination in data['player_types'].unique():
        for matchup in permutations(combination):
            p0,p1 = matchup
            trials = data[(data['player_types']==combination) & (data['type']==p0)]
            names = []
            for t in matchup:
                try:
                    names.append(t.short_name("agent_types"))
                except:
                    names.append(str(t))
            p0,p1 = names
            fitness = trials.mean()['fitness']
            try:
                fitness /= params['games'].rounds
            except:
                print Warning("matchup_matrix didn't find a round parameter in the game")
            record.append({'recipient prior':p0, 'opponent prior':p1, 'reward':fitness})
            if p0 == p1:
                break
    meaned = pd.DataFrame(record).pivot(index = 'recipient prior',columns = 'opponent prior',values = 'reward')
    plt.figure(figsize = (10,10))
    sns.heatmap(meaned,annot=True,fmt="0.2f")

def history_maker(observations,agents,start=0,annotation = {}):
    history = []
    for r,observation in enumerate(observations,start):
        [agent.observe(observation) for agent in agents]
        history.append(dict({
            'round':r,
            'players':deepcopy(agents)},**annotation))
    return history

@multi_call(twinned = ['player_types','priors','Ks'])
@experiment(unordered = ['agent_types'],unpack = 'dict')
def forgiveness(player_types = NiceReciprocalAgent, Ks= 1, priors=(.75,.75), defections=3, **kwargs):
    condition = dict(locals(),**kwargs)
    params = default_params(**condition)
    game = BinaryDictator()
    genomes = [default_genome(agent_type = t, RA_K = k, RA_prior = p,**condition) for t,p,k in zip(player_types,priors,Ks)]
    world = World(params,genomes)
    agents = world.agents
    observations = [[(game,[0,1],[0,1],'keep')]]*defections
    prehistory = history_maker(observations,agents,start = -defections)
    fitness, history = world.run()
    history = prehistory+history
    return {'history':history}

id_to_letter = dict(enumerate("ABCDEF"))
@apply_to_args(twinned = ['player_types','priors','Ks'])
@plotter(binary_matchup,plot_exclusive_args = ['data','believed_type'])
def belief_plot(player_types,priors,Ks,believed_types=None,data=[],**kwargs):
    if not believed_types:
        believed_types = list(set(player_types))
    K = max(Ks)+1
    t_ids = [[list(islice(cycle(order),0,k)) for k in range(1,K+2)] for order in [(1,0),(0,1)]]
    record = []
    for d in data.to_dict('record'):
        for event in d['history']:
            #print event
            for a_id, believer in enumerate(event['players']):
                a_id = believer.world_id
                for ids in t_ids[a_id]:
                    k = len(ids)-1
                    for believed_type in believed_types:
                        record.append({
                            "believer":a_id,
                            "k":k,
                            "belief":believer.k_belief(ids,believed_type),
                            "target_id":ids[-1],
                            "round":event['round'],
                            "type":justcaps(believed_type),
                    })
    bdata = pd.DataFrame(record)
    #import pdb; pdb.set_trace()
    bt = map(justcaps,believed_types)
    f_grid = sns.factorplot(data = bdata, x = 'round', y = 'belief', row = 'k', col = 'believer', kind = 'violin', hue = 'type', row_order = range(K+1), legend = False, hue_order = bt,
                   facet_kws = {'ylim':(0,1)})
    f_grid.map(sns.pointplot,'round','belief','type', hue_order = bt, palette = sns.color_palette('muted'))
    for a_id,k in product([0,1],range(K+1)):
        ids = t_ids[a_id][k]
        axis = f_grid.facet_axis(k,a_id)
        axis.set(#xlabel='# of interactions',
            ylabel = '$\mathrm{Pr_{%s}( T_{%s} = %s | O_{1:n} )}$'% (k,id_to_letter[ids[-1]],justcaps(believed_type)),
            title = ''.join([id_to_letter[l] for l in [a_id]+ids]))
    agents = []
    for n,(t,p) in enumerate(zip(player_types,priors)):
        if issubclass(t,RationalAgent):
            if player_types[0]==player_types[1] and priors[0]==priors[1]:
                agents.append("%s(prior=%s)"%(str(t),p))
            else:
                agents.append("%s(prior=%s)"%(str(t),p))
        else:
            if player_types[0]==player_types[1]:
                agents.append("%s" % (str(t),n))
            else:
                agents.append(str(t))
    #print agents
    #plt.subplots_adjust(top = 0.9)
    #if kwargs.get('experiment',False) == 'forgiveness':
     #   f_grid.fig.suptitle("A and B's beliefs that the other is %s after A defects some number of times\nA=%s B=%s" % (justcaps(believed_type),agents[0],agents[1]))
    #else:
    #f_grid.fig.suptitle("A and B's beliefs that the other is %s\nA=%s B=%s" % (justcaps(believed_type),agents[0],agents[1]))

@plotter(binary_matchup)
def joint_fitness_plot(player_types,priors,Ks,data = []):
    agents = []
    for n,(t,p) in enumerate(zip(player_types,priors)):
        if issubclass(t,RationalAgent):
            if player_types[0]==player_types[1] and priors[0]==priors[1]:
                agents.append("%s(prior=%s) #%s"%(str(t),p,n))
            else:
                agents.append("%s(prior=%s)"%(str(t),p))
        else:
            if player_types[0]==player_types[1]:
                agents.append("%s #%s" % (str(t),n))
            else:
                agents.append(str(t))

    record = []
    for rec in data.to_dict('record'):
        record.append(dict(zip(agents,rec['fitness'])))
    data = pd.DataFrame(record)
    bw = .5
    sns.jointplot(agents[0], agents[1], data, kind = 'kde',bw = bw,marginal_kws = {"bw":bw})

def unordered_prior_combinations(prior_list):
    return map(tuple,map(sorted,combinations_with_replacement(prior_list,2)))
#@apply_to_args(twinning = ['player_types'])

#@multi_call()
def comparison_grid(size = 5, **kwargs):
    priors = [round(n,2) for n in np.linspace(0,1,size)]
    #priors = [round(n,2) for n in [0,.001,.01,.1,.5,.75,1]]
    #priors = MultiArg(unordered_prior_combinations(priors))
    priors = MultiArg(product(priors,priors))
    condition = dict(kwargs,**{'priors':priors})
    #print condition
    ret =  binary_matchup(return_keys = ('p1_fitness','fitness'),**condition)
    return ret

def RA_matchup_matrix(size = 5,**kwargs):
    data = comparison_grid(size,**kwargs)
    #types = np.round(np.linspace(0,1,size),decimals = 2)
    types = [round(n,2) for n in [0,.001,.01,.1,.5,.75,1]]
    size = len(types)
    prior_to_index = dict(map(reversed,enumerate(types)))
    index_to_prior = dict(enumerate(types))
    matrix = np.zeros((size,)*2)
    priors = sorted(list(set(data['priors'])))
    for prior in priors:
        p0,p1 = (prior_to_index[p] for p in prior)
        group = data[data['priors']==prior]
        matrix[p0,p1] = group.mean()['p1_fitness']

    if 'rounds' in kwargs:
        matrix /= kwargs['rounds']
    return matrix,types

@plotter(comparison_grid, plot_exclusive_args = ['data'])
def reward_table(data = [],**kwargs):
    record = []
    r2 = []
    priors = sorted(list(set(data['priors'])))
    #for prior,group in data.groupby('priors'):
    for prior in priors:
        p0,p1 = prior
        group = data[data['priors']==prior]
        r2.append({'recipient prior':p0, 'opponent prior':p1, 'reward':group.mean()['p1_fitness']})
        for r0,r1 in group['fitness']:
            record.append({'recipient prior':p0, 'opponent prior':p1, 'reward':r0})
    #data = pd.DataFrame(record)
    #meaned = data.groupby(['recipient prior','opponent prior']).mean().unstack()
    meaned = pd.DataFrame(r2).pivot(index = 'recipient prior',columns = 'opponent prior',values = 'reward')
    plt.figure(figsize = (10,10))
    sns.heatmap(meaned,annot=True,fmt="0.2f")

@multi_call()
@experiment(unpack = 'record',unordered = ['agent_type'],memoize = False)
def tft_scenes(agent_types, **kwargs):
    #condition = dict(locals(),**kwargs)
    genome = default_genome(agent_type = RationalAgent, agent_types = agent_types, **kwargs)
    game = BinaryDictator()
    #observations = {}
    def vs(actions, observers = "ABO"):
        observations = []
        #observers = [letter_to_id.get(p,p) for p in observers]
        for players, action in zip(["AB","BA"], actions):
            #players = [letter_to_id[p] for p in players]
            action = letter_to_action[action]
            observations.append((game, players, observers, action))
        return observations

    obs_dict = {}
    action_seqs = [("CD","DC","CD"),
                   ("CD","DC","DD")]
    for action_seq in action_seqs:
        obs = []
        for actions in action_seq:
            obs.append(vs(actions))
        obs_dict[action_seq] = obs

    record = []
    for action, observations in obs_dict.iteritems():
        observer = RationalAgent(genome,"O")
        print "KKKKKKKKKKKKK\n\n\n", observer.genome["RA_K"]
        for observation in observations:
            observer.observe(observation)
            print observation
            for agent_type in agent_types:
                model = observer.model["A"][agent_type]
                print agent_type
                print model.decide_likelihood(game,observation[1],0)[game.action_lookup[observation[0][3]]]

                
            #assert False
        
        for agent_type in agent_types:
            record.append({
                'scenario':action,
                'belief':observer.belief_that("A",agent_type),
                'type':justcaps(agent_type)
            })
    return record

def minimal_ratios(ratio_dict):
    if 1 > ratio_dict.values()[0]:
        objs,ratios = zip(*ratio_dict.items())
        ratios = [Fraction(r).limit_denominator(1000) for r in ratios]
        mult = lcm(*[f.denominator for f in ratios])
        ratios = [r*mult for r in ratios]
        ratio_dict = dict(zip(objs,ratios))
    divisor = gcd(*ratio_dict.values())
    return {k:int(v/divisor) for k,v in ratio_dict.iteritems()}

memo_bin_matchup = memoize(binary_matchup)

def mean(*numbers):
    return sum(numbers)/len(numbers)
@multi_call(verbose = 3)
@experiment(unpack = 'dict', trials = 100,verbose = 3)
def pop_matchup_simulator(player_types=(ReciprocalAgent,SelfishAgent), min_pop_size=50, moran = .01, proportion=.5, **kwargs):
    for item in ['type_to_population','matchup_function','trials','trial']:
        try:
            del kwargs[item]
        except:
            pass
    condition = kwargs

    proportions = minimal_ratios(dict(zip(player_types,(proportion,1-proportion))))

    try:
        #assert not proportion ==.9
        assert proportion in [0,1] or not (0 in proportions.values() and 1 in proportions.values())
    except Exception as e:
        print Fraction(1-proportion).limit_denominator(1000)
        print proportion
        print proportions
        raise e
    min_legal_pop_size = sum(proportions.values())
    if min_legal_pop_size < min_pop_size:
        pop_scale = int(np.ceil(min_pop_size/min_legal_pop_size))
        proportions = {k:v*pop_scale for k,v in proportions.iteritems()}

    type_to_population = proportions
    agent_types = sorted(type_to_population.keys())
    try:
        type_list = sum(([agent_type]*type_to_population[agent_type] for agent_type in agent_types),[])
    except MemoryError as ME:
        print proportion
        print proportions
        raise ME
    agent_list = list(enumerate(type_list))
    pop_size = len(agent_list)
    agent_matchups = [map(tuple,zip(*sorted(item,key = operator.itemgetter(1)))) for item in combinations(agent_list,2)]
    fitness = np.zeros(pop_size)
    type_matchups = [tuple(sorted(item)) for item in set(combinations(map(operator.itemgetter(1),agent_list),2))]
    matchup_to_fitnesses = {matchup : map(tuple, memo_bin_matchup(return_keys = 'fitness', trials = 500, player_types = matchup, **condition)['fitness'])
                            for matchup in type_matchups}
    trials = len(matchup_to_fitnesses.values()[0])

    for ids,types in agent_matchups:
        i = np.random.random_integers(0,high = trials-1)
        fitness[np.array(ids)] += array(matchup_to_fitnesses[types][i])

    fitness = zip(type_list,fitness)
    avg_fitnesses = defaultdict(int)
    for t,f in fitness:
        avg_fitnesses[t] += f

    for t in avg_fitnesses:
        avg_fitnesses[t] = avg_fitnesses[t]/type_list.count(t)

    fitnesses = softmax_utility(avg_fitnesses,moran)
    return {'fitness ratio':fitnesses[player_types[0]],
            'mean_fitness':[avg_fitnesses[t] for t in player_types],
            'p1_fitness':avg_fitnesses[player_types[0]],
            'type_fitness_pairs':fitness}



@plotter(pop_matchup_simulator, plot_exclusive_args = ['data'])
def pop_fitness_plot(player_types = (ReciprocalAgent,SelfishAgent), proportion = MultiArg([.25,.5,.75]), Ks = MultiArg([0,1]), data = None):
    #print data
    #ndata = data.groupby(['RA_K','proportion']).mean().unstack()
    #print ndata
    
    sns.pointplot(data = data, x = "proportion", y = "fitness ratio", hue = "Ks")
    #print locals()
    #fplot.set(yticklabels = np.linspace(0,1,5))

#from evolve import limit_analysis

def test_matchup_matrix(RA):
    
    SA = RA(RA_prior = 0)
    AA = RA(RA_prior = 1)
    t = RA(RA_prior = .5)
    ToM = (SelfishAgent,t,AltruisticAgent)
    player_types = (SA, RA, AA)
    agent_types = (SelfishAgent, RA, AltruisticAgent)
    TFT = gTFT(y=1,p=1,q=0)
    TvT = matchup_matrix(player_types = (AltruisticAgent,TFT), agent_types = ToM, rounds = 10)
    print TvT
    assert False
    g = default_genome()
    
   
    m = matchup_matrix(player_types = (SA,t,AA), agent_types = (SelfishAgent,t,AltruisticAgent),trials = 500)
    k = matchup_matrix(player_types = agent_types, agent_types = agent_types, RA_prior = .5,trials = 500)
    print m
    print k
    a = t(default_genome(agent_type = t, agent_types = (SelfishAgent,t,AltruisticAgent)))
    b = RA(default_genome(agent_type = RA, RA_prior = .5, agent_types = agent_types))
    print a.model["O"][t].genome
    print b.model["O"][RA].genome
    pass


@experiment(unpack = 'record',memoize = False)
def fitness_v_trials(max_trials, player_type, opponent_types, **kwargs):
    record = []
    params = default_params(**kwargs)
    rounds = params['games'].rounds
    for opponent_type in opponent_types:
        data = matchup([player_type,opponent_type], trials = max_trials,**kwargs)
        for t in range(1,max_trials+1):
            fitness = data[(data['type'] == player_type) & (data['trial']<=t)].mean()['fitness']
            record.append({"fitness":fitness/rounds,
                           "type":opponent_type,
                           "trials":t})
    return record

@plotter(fitness_v_trials,plot_exclusive_args = ['data'])
def fitness_trials_plot(max_trials,player_type,opponent_types,data=[],**kwargs):
    fig = plt.figure()
    for hue in data['type'].unique():
        d = data[data['type']==hue]
        p = plt.plot(d['trials'], d['fitness'], label=hue)
    plt.legend()

@experiment(unpack = 'record', memoize = False)
def self_pay_v_rounds(max_rounds, player_types, **kwargs):
    Xs = range(1,max_rounds)
    record = []
    for player_type in player_types:
        try:
            t_name = player_type.short_name('agent_types')
        except:
            t_name = player_type.__name__
        data = matchup(player_types = (player_type, player_type), rounds = max_rounds, trials = 50, per_round = True, **kwargs)
        sum = 0
        for r in range(1,max_rounds+1):
            sum += data[data['round']==r].mean()['fitness']
            record.append({
                "rounds":r,
                "type":t_name,
                "fitness":sum/r
            })
    return record

def self_pay_experiments():
    TFT = gTFT(y=1,p=1,q=0)
    GTFT = gTFT(y=1,p=.99,q=.33)
    MRA = ReciprocalAgent
    NRA = NiceReciprocalAgent
    SA = SelfishAgent
    AA = AltruisticAgent
    AC = AllC
    AD = AllD

    RA = MRA

    rounds = 100
    prior = .5
    Ks = [0,1,2,3]
    t = .05
    ToMs = [('self', AC, AD),
            ('self', AC, AD, TFT, Pavlov),
            ('self', AC, AD, TFT, Pavlov, GTFT),
            ('self', AC, AD, TFT, Pavlov, GTFT, RandomAgent)]

    for ToM in ToMs:
        RA_Ks = tuple(RA(RA_K = k) for k in Ks)
        self_pay_plot(rounds, player_types = RA_Ks, agent_types =ToM, RA_prior = prior)
        self_pay_plot(rounds, player_types = RA_Ks, agent_types =ToM, RA_prior = prior, tremble = t)

def belief_experiments():
    TFT = gTFT(y=1,p=1,q=0)
    GTFT = gTFT(y=1,p=.99,q=.33)
    MRA = ReciprocalAgent
    NRA = NiceReciprocalAgent
    SA = SelfishAgent
    AA = AltruisticAgent
    AC = AllC
    AD = AllD

    contest_tom = (MRA,AC,AD)
    race_tom = (MRA,AC,AD,TFT,GTFT,Pavlov)
    K = 0
    ToM = contest_tom
    plot_dir = "./plots/belief examples (K=%s, ToM = %s)/" % (K,ToM)

    for t in range(50):
        belief_plot(believed_types = contest_tom, player_types = MRA, agent_types = contest_tom, priors = .5,
                    Ks = K, rounds = 500, trials = [t], tremble = 0.05, beta = 1,
                    plot_dir = plot_dir,
                    #file_name = "k1 v k2, t = %s" % t
        )
@plotter(self_pay_v_rounds, plot_exclusive_args = ['data'])
def self_pay_plot(max_rounds, player_types, data=[], **kwargs):
    fig = plt.figure()
    for hue in data['type'].unique():
        d = data[data['type']==hue]
        p = plt.plot(d['rounds'], d['fitness'], label=hue)
    plt.legend()

if __name__ == "__main__":
    #self_pay_experiments()
    belief_experiments()
