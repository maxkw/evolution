from __future__ import division
import pandas as pd
import seaborn as sns
from experiment_utils import multi_call,experiment,plotter,MultiArg, memoize, apply_to_args
import numpy as np
from params import default_params,generate_proportional_genomes,default_genome
from world import World
from agents import ReciprocalAgent, SelfishAgent, AltruisticAgent, RationalAgent, WeAgent
from agents import gTFT, GTFT, TFT, AllC, AllD, Pavlov, RandomAgent, leading_8_dict, shorthand_to_standing
from games import RepeatedPrisonersTournament,BinaryDictator,Repeated,PrivatelyObserved,Symmetric
from collections import defaultdict
from itertools import combinations_with_replacement, combinations
from itertools import permutations
from itertools import product,islice,cycle
import matplotlib.pyplot as plt
from numpy import array
from copy import copy,deepcopy
from utils import softmax_utility, _issubclass, normalized
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
def binary_matchup(player_types, priors, Ks, **kwargs):
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

    try:
        player_types,pop = zip(*player_types)
    except TypeError:
        #print Warning("player_types is not a zipped list")
        pop = tuple(1 for t in player_types)

    condition = dict(player_types = player_types,**kwargs)
    params = default_params(**condition)

    player_types = sum([[t]*p for t,p in zip(player_types,pop)],[])
    genomes = [default_genome(agent_type = t, **condition) for t in player_types]

    world = World(params,genomes)
    fitness, history = world.run()

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
        ids = [a.world_id for a in world.agents]
        for event in history:
            r = event['round']
            for t, a_id, p, b, l, n_l in zip(player_types, ids, event['payoff'], event['beliefs'], event['likelihoods'], event['new_likelihoods']):
                if kwargs.get('unpack_beliefs', False):
                    atypes = genomes[a_id]['agent_types']
                    if b:
                        o_id = (a_id+1)%2
                        l = np.exp(l[o_id])
                        n_l = np.exp(n_l[o_id])
                        b = b[o_id]
                        for believed_type in kwargs['believed_types']:
                            bt_index = atypes.index(believed_type)
                            try:
                                attr_to_val = {
                                    'belief' : b[bt_index],
                                    'likelihood': normalized(l[bt_index]),
                                    'new_likelihood': n_l[bt_index]
                                }
                            except:
                                print n_l
                                raise
                            for attr,val in attr_to_val.iteritems():
                                record.append({'type' : repr(t),
                                               'id' : a_id,
                                               'round' : r,
                                               'attribute' : attr,
                                               'value' : val,
                                               'believed_type':repr(believed_type),
                                               'fitness' : p})
                else:
                    record.append({'type' : t,
                                   'id' : a_id,
                                   'round' : r,
                                   'fitness' : p})
    return record

def beliefs(believer, opponent_types, believed_types, **kwargs):
    dfs = []
    b_name = repr(believer)#.short_name('agent_types')
    for opponent in opponent_types:
        data = matchup(player_types = (believer,opponent),
                       actual_type = repr(opponent),
                       believed_types = believed_types,
                       per_round = True, unpack_beliefs = True, **kwargs)
                       
        if believer == opponent:
            dfs.append(data[data['id'] == 0])
        else:
            dfs.append(data[data['type'] == b_name])
    return pd.concat(dfs, ignore_index = True)

@plotter(beliefs)
def plot_beliefs(believer, opponent_types, believed_types, data = [],**kwargs):
    print data
    #import pdb; pdb.set_trace()
    fgrid = sns.factorplot(data = data, x = 'round', y = 'value', col = 'actual_type', row = 'attribute', kind = 'point', hue = "believed_type", row_order = ('belief','new_likelihood','likelihood'), margin_titles = True,
                           facet_kws = {'ylim':(-.05,1.05)}
    )

    #(fgrid
    # .set_xlabels("P(t = Type)")
     #.set_titles("")
     #.set_ylabels5
                
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
def forgiveness(player_types, Ks= 1, priors=(.75,.75), defections=3, **kwargs):
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
    f_grid.set(xticks=[])
    for a_id,k in product([0,1],range(K+1)):
        ids = t_ids[a_id][k]
        axis = f_grid.facet_axis(k,a_id)
        axis.set(#xlabel='# of interactions',
            ylabel = '$\mathrm{Pr_{%s}( T_{%s} = %s | O_{1:n} )}$'% (k,id_to_letter[ids[-1]],justcaps(believed_type)),
            title = ''.join([id_to_letter[l] for l in [a_id]+ids]))
    agents = []
    for n,(t,p) in enumerate(zip(player_types,priors)):
        if _issubclass(t,RationalAgent):
            if player_types[0]==player_types[1] and priors[0]==priors[1]:
                agents.append("%s(prior=%s)"%(str(t),p))
            else:
                agents.append("%s(prior=%s)"%(str(t),p))
        else:
            if player_types[0]==player_types[1]:
                agents.append("%s" % (str(t),n))
            else:
                agents.append(str(t))

@apply_to_args(twinned = ['player_types','priors','Ks'])
@plotter(binary_matchup,plot_exclusive_args = ['data','believed_type'])
def coop_plot(player_types,priors,Ks,believed_types=None,data=[],**kwargs):
    if not believed_types:
        believed_types = list(set(player_types))
    K = max(Ks)
    t_ids = [[list(islice(cycle(order),0,k)) for k in range(0,K+1)] for order in [
        (1,0),
        (0,1)
    ]]
    #scale = lambda n: n*.99+.005
    logit = lambda p:  np.log(p/(1-p))
    E = 0.000000000000000001
    logiter = lambda p: logit(max(min(p,1-E), p))
    record = []
    for d in data.to_dict('record'):
        for event in d['history'][1:]:
            #print event
            game = event['games'][0]
            
            for a_id, believer in enumerate(event['players']):
                players = [a_id,(a_id+1)%2]
                # print "here"
                # print a_id
                # print players
                # print believer.belief_that((a_id+1)%2,ReciprocalAgent)
                # print believer.decide_likelihood(game,players,kwargs.get('tremble',0))[game.actions.index('give')]
                for t in ['belief','coop']:
                    if t == 'belief':
                        it = believer.belief_that((a_id+1)%2,ReciprocalAgent)
                    else:
                        it = believer.decide_likelihood(game,players,kwargs.get('tremble',0))[game.actions.index('give')]
                    print t,'\n'*11
                    record.append({
                        'believer':a_id,
                        'k':0,
                        'value': it,
                        'round':event['round'],
                        'type':t.short_name(),
                    })
    bdata = pd.DataFrame(record)
    #import pdb; pdb.set_trace()
    bt =  ['belief','coop']
    f_grid = sns.factorplot(data = bdata, x = 'round', y = 'value', row = '', col = 'believer', kind = 'point', hue = 'type', legend = False, facet_kws = {'ylim':(0,1)}, ci = None)
    #f_grid.set(yscale = "logit")
    


@plotter(binary_matchup)
def joint_fitness_plot(player_types,priors,Ks,data = []):
    agents = []
    for n,(t,p) in enumerate(zip(player_types,priors)):
        if _issubclass(t,RationalAgent):
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

@experiment(unpack = 'record')
def self_pay_v_rounds(max_rounds, player_types, e_trials = 50, **kwargs):
    Xs = range(1,max_rounds)
    record = []
    for player_type in player_types:
        try:
            t_name = player_type.short_name('agent_types')
        except:
            t_name = player_type.__name__
        data = matchup(player_types = (player_type, player_type), rounds = max_rounds, trials = e_trials, per_round = True, **kwargs)
        sum = 0
        for r in range(1,max_rounds+1):
            record.append({
                "rounds":r,
                "type":t_name,
                "fitness": data[data['round']==r].mean()['fitness']
            })
    return record

def self_pay_experiments():
    TFT = gTFT(y=1,p=1,q=0)
    GTFT = gTFT(y=1,p=.99,q=.33)
    MRA = ReciprocalAgent
    SA = SelfishAgent
    AA = AltruisticAgent
    AC = AllC
    AD = AllD

    plot_dir = "./plots/self_pay/"
    file_name = "ToM = %s, beta = %s, tremble = %s, rounds = %r"
    RA = MRA

    rounds = 1000
    prior = .5
    Ks = [0,1,2]
    t = .05
    ToMs = [
        ('self', AD)
        #('self', AC, AD),
        #('self', AC, AD, TFT, Pavlov),
        #('self', AC, AD, TFT, Pavlov, GTFT),
        #('self', AC, AD, TFT, Pavlov, GTFT, RandomAgent)
    ]
    betas = [1,3,10]
    betas = [.5,1,1.5,2,2.5,3,3.5,4,4.5]
    betas = [3]
    trembles = [
        #0,
        0.05
    ]
    max_k = 5
    Ks = range(max_k+1)
    rounds_list = [max_k+50]
    for trials in [100]:#[n*10 for n in range(1,11)]:
        for ToM,beta,tremble,rounds in product(ToMs,betas,trembles,rounds_list):
            RA_Ks = tuple(RA(RA_K = k) for k in Ks)
            self_pay_plot(rounds, player_types = RA_Ks, agent_types = ToM, RA_prior = prior, beta = beta, e_trials = trials,
                          tremble = tremble,
                          plot_dir = plot_dir,
                          file_name = file_name % (ToM,beta,tremble,rounds))

    #for ToM,beta,tremble,rounds in product(ToMs,betas,trembles,rounds_list):
    #    for trials in [n*10 for n in range(1,6)]:
    #        RA_Ks = tuple(RA(RA_K = k) for k in Ks)
    #        self_pay_plot(rounds, player_types = RA_Ks, agent_types = ToM, RA_prior = prior, beta = beta, e_trials = trials,
    #                      tremble = tremble,
    #                      plot_dir = plot_dir,
    #                      file_name = file_name % (ToM,beta,tremble,rounds))

def belief_experiments2():

    MRA = ReciprocalAgent
    SA = SelfishAgent
    AA = AltruisticAgent
    AC = AllC
    AD = AllD

    contest_tom = (MRA,AC,AD,RandomAgent)
    race_tom = (MRA,AC,AD,TFT,GTFT,Pavlov)
    K = 2
    ToM = contest_tom
    plot_dir = "./plots/belief examples (K=%s, ToM = %s)/" % (K,ToM)

    #for t in [50]:#range(1,50):
    for t in [10]:
        coop_plot(believed_types = contest_tom, player_types = MRA, agent_types = contest_tom, priors = .5,
                    Ks = K, rounds = 10, trials = t, tremble = 0.05, beta = 3,
                    plot_dir = plot_dir, plot_trials = True
                    #file_name = "k1 v k2, t = %s" % t
        )
        #belief_plot(believed_types = contest_tom, player_types = MRA, agent_types = contest_tom, priors = .5,
        #            Ks = K, rounds = 500, trials = [t], tremble = 0.05, beta = 10,
        #            plot_dir = plot_dir,
        #            #file_name = "k1 v k2, t = %s" % t
        #)

@plotter(self_pay_v_rounds, plot_exclusive_args = ['data'])
def self_pay_plot(max_rounds, player_types, data=[], **kwargs):
    fig = plt.figure()
    for hue in data['type'].unique():
        d = data[data['type']==hue]
        p = plt.plot(d['rounds'], d['fitness'], label=hue)
    axes = plt.gca()
    axes.set_ylim(0,3)
    plt.legend()

def belief_experiments():

    plot_dir = "./plots/belief_experiments/"
    WA = WeAgent
    #WA._nickname = "WeAgent"
    MRA = ReciprocalAgent
    SA = SelfishAgent
    AA = AltruisticAgent
    AC = AllC
    AD = AllD
    #TFT= gTFT(y=1,p=1,q=0)
    #GTFT = gTFT(y=1,p=.99,q=.33)
    RA = WeAgent

    priors = [
        .1,
        #.5,
        #.75
    ]

    ToMs = [
        ('self', AC, AD, TFT, GTFT, Pavlov)
    ]

    betas = [
        #1,
        #3,
        #5,
        10,
    ]

    trembles = [
        #0,
        0.05
    ]

    for prior, ToM, beta,tremble in product(priors,ToMs,betas,trembles):
        agent = RA(RA_prior = prior, agent_types = ToM, beta = beta)
        everyone = (AC, AD, TFT, GTFT, Pavlov)
        
        for t in trembles:
            for trial in range(1,11):
                plot_beliefs(agent,
                             (agent,)+everyone,
                             (agent,)+everyone,
                             tremble = tremble,
                             plot_dir = plot_dir,
                             #games = game,
                             file_name = "belief - trial "+str(trial),
                             trials = [trial])
@experiment(unpack = 'record', memoize=False)
def coop_prob(cost=1, benefit=3, **kwargs):
    bd = BinaryDictator(cost = cost, benefit = benefit)
    actions = bd.actions
    #W = WeAgent(default_genome(agent_type = WeAgent, agent_types = ('self', AllD)), world_id = "A")
    record = []
    for beta, belief in product([1, 3, 5, 10, 100], np.linspace(0,1,200)):
        W = WeAgent(default_genome(agent_type = WeAgent, agent_types = ('self', AllD), RA_prior = belief, beta = beta), world_id = "A")
        for a,p in zip(actions, W.decide_likelihood(bd,"AB",0)):
            record.append({
                "beta": beta,
                "action": a,
                "prob": p,
                "belief": belief,
            })
    return record

@plotter(coop_prob, plot_exclusive_args = ['data'])
def plot_coop_prob(data = []):
    data = data[data['action'] == 'give']
    fig = plt.figure(figsize=(6,6))

    for hue in data['beta'].unique():
        d = data[data['beta']==hue]
        p = plt.plot(d['belief'], d['prob'], label=hue)

    plt.ylim([0, 1.05])
    plt.xlim([0, 1])
    plt.yticks([0,0.5,1])
    plt.xticks([0.25, 0.5, 0.75, 1])
    plt.xlabel('Reciprocity Prior')
    plt.ylabel('P(Action = Give)')
    sns.despine()
    plt.legend()
    plt.tight_layout()

def test_standing(**kwargs):
    from games import Symmetric
    standing_types = tuple(s_type for name,s_type in sorted(leading_8_dict().iteritems(),key = lambda x:x[0]))
    #standing_types = (shorthand_to_standing('ggggbgbbnnnn'),)
    standing_types = (standing_types[0],)
    W = WeAgent(agent_types = ('self',)+standing_types, RA_prior = .49, beta = 5)
    tremble = 0
    decision = BinaryDictator(cost = 1, benefit = 3,tremble = tremble)
    game = Repeated(10, Symmetric(PrivatelyObserved(decision)))
    #game = Repeated(10,PrivatelyObserved(Symmetric(decision)))
    p_types = (W,)+standing_types
    #matchup_plot(player_types = (W,)+standing_types, games = game,tremble = .05)
    # for t in range(1,10):
    plot_beliefs(W,p_types,p_types,file_name = 'WA_beliefs_Leading8', games = game)
    for t in range(1,10):
        plot_beliefs(W,p_types,p_types,file_name = 'WA_beliefs_Leading8 - trial '+str(t), games = game, trials = [t])
    #    plot_beliefs(W,(W,),(W,),file_name = 'self_belief'+str(t), games = game, trials = [t])


    L1 = leading_8_dict()['L1']
    L1 = shorthand_to_standing('ggggbgbbnnnn')
    a = L1(default_genome(agent_type = L1),"A")
    print a.image['B']
    print a.decide_likelihood(decision,"AB")
    a.observe([(decision,"BC","ABC",'keep')])
    print a.image['B']
    print a.decide_likelihood(decision,"AB")
    

if __name__ == "__main__":
    #belief_experiments()
    test_standing()
    #plot_coop_prob(extension = ".png")
    assert 0
    RA = ReciprocalAgent
    TFT = gTFT(y=1,p=1,q=0)
    GTFT = gTFT(y=1,p=.99,q=.33)
    Ks = tuple(RA(RA_K = k) for k in [0,1,2])
    As = tuple(WeAgent(beta = b) for b in [1,3,5,10])

    matchup_plot(player_types = (WeAgent, AllD), agent_types = ('self', AllD))
    belief_experiments()

    
    assert 0
    for t in [t*10 for t in range(1,11)]:
        self_pay_plot(200, player_types = As, agent_types = ('self', AllD, AllC, TFT, GTFT, Pavlov), RA_prior = .5,  e_trials = t, extension = '.png', tremble = .05)
    #self_pay_experiments()
    #belief_experiments()
