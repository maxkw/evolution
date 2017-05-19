from __future__ import division
from collections import Counter, defaultdict
from itertools import product, permutations, izip
from utils import normalized, softmax, excluding_keys
from math import factorial
import numpy as np
from copy import copy
import operator
from experiments import NiceReciprocalAgent, SelfishAgent, ReciprocalAgent, AltruisticAgent
from experiment_utils import multi_call, experiment, plotter, MultiArg, memoize, apply_to_args
import matplotlib.pyplot as plt
import seaborn as sns
from experiments import binary_matchup, memoize, matchup_matrix, matchup_plot,matchup_matrix_per_round
from params import default_genome
from indirect_reciprocity import gTFT, AllC, AllD, Pavlov, RandomAgent,WeAgent
from params import default_params
from steady_state import limit_analysis, complete_analysis
import pandas as pd
from datetime import date

def logspace(start = .001,stop = 1, samples=10):
    mult = (np.log(stop)-np.log(start))/np.log(10)
    plus = np.log(start)/np.log(10)
    return np.array([0]+list(np.power(10,np.linspace(0,1,samples)*mult+plus)))

def int_logspace(start, stop, samples, base=2):
    return sorted(list(set(np.logspace(start, stop, samples, base=base).astype(int))))

def agent_sim(payoff, pop, s, mu):
    pop_size = sum(pop)
    type_count = len(payoff)
    I = np.identity(type_count)
    pop = np.array(pop)
    assert type_count == len(pop)
    while True:
        yield pop
        fitnesses = [np.dot(pop - I[t], payoff[t]) for t in xrange(type_count)]
        fitnesses = softmax(fitnesses, s)
        # fitnesses = [np.exp(s*np.dot(pop - I[t], payoff[t])) for t in xrange(type_count)]
        # total_fitness = sum(fitnesses)
        actions = [(b,d) for b,d in permutations(xrange(type_count),2) if pop[d]!=1]
        probs = []
        for b,d in actions:
            death_odds = pop[d] / pop_size
            # birth_odds = (1-mu) * fitnesses[b] / total_fitness + mu * (1/type_count)
            birth_odds = (1-mu) * fitnesses[b] + mu * (1/type_count)
            prob = death_odds * birth_odds
            probs.append(prob)
        actions.append((0, 0))
        probs.append(1 - np.sum(probs))

        probs = np.array(probs)
        action_index = np.random.choice(len(probs), 1, p = probs)[0]
        (b,d) = actions[action_index]
        pop = pop + I[b]-I[d]

@experiment(unpack = 'record', memoize = False)
def agent_simulation(generations, pop, player_types, **kwargs):
    payoffs = matchup_matrix(player_types = player_types, **excluding_keys(kwargs, 's', 'mu'))
    params = default_params(**kwargs)
    populations = agent_sim(payoffs, pop, params['s'], params['mu'])

    record = []
    for n, pop in izip(xrange(generations), populations):
        for t, p in zip(player_types, pop):
            record.append({'generation' : n,
                           'type' : t.short_name('agent_types'),
                           'population' : p})
    return record


@plotter(agent_simulation,plot_exclusive_args = ['data'])
def sim_plotter(generations, pop, player_types, data =[]):
    for hue in data['type'].unique():
        plt.plot(data[data['type']==hue]['generation'], data[data['type']==hue]['population'], label=hue)

    plt.legend()

@experiment(unpack = 'record', memoize = False, verbose = 3)
def limit_v_evo_param(param, player_types, **kwargs):
    payoffs = matchup_matrix(player_types = player_types, **kwargs)
    # matchup_plot(player_types = player_types, **kwargs)

    if param == 'pop_size':
        Xs = np.unique(np.geomspace(2, 2**10, 200, dtype=int))
    elif param == 's':
        Xs = logspace(start = .001, stop = 10, samples = 100)
    else:
        print param
        raise

    params = default_params(**kwargs)
    
    record = []
    for x in Xs:
        params[param] = x
        for t, p in zip(player_types, limit_analysis(payoffs, **params)):
            record.append({
                param : x,
                "type" : t.short_name("agent_types"),
                "proportion" : p
            })
    return record

@experiment(unpack = 'record', memoize = False, verbose = 3)
def limit_v_sim_param(param, player_types, **kwargs):
    if param == "RA_prior":
        Xs = np.linspace(0,1,21)[1:-1]
    elif param == "beta":
        Xs = logspace(.5,6,11)
    else:
        raise

    record = []
    for x in Xs:
        payoffs = matchup_matrix(player_types = player_types, **dict(kwargs,**{param:x}))

        for t,p in zip(player_types, limit_analysis(payoffs, **default_params(**{param:x}))):
            record.append({
                param:x,
                "type":t.short_name("agent_types"),
                "proportion":p
            })
    return record
@experiment(unpack = 'record', memoize = False, verbose = 3)
def limit_v_rounds(player_types, max_rounds = 100,  **kwargs):
    matrices = matchup_matrix_per_round(player_types, max_rounds = max_rounds, **kwargs)
    params = default_params(**kwargs)
    record = []
    
    for r, payoff in matrices:

        for t,p in zip(player_types, limit_analysis(payoff, **params)):
            record.append({
                'rounds':r,
                "type":t.short_name("agent_types"),
                "proportion":p
            })
    return record

def limit_v_param(param,player_types,**kwargs):
    if param in ['beta','RA_prior']:
        return limit_v_sim_param(param,player_types,**kwargs)
    elif param in ['pop_size','s']:
        return limit_v_evo_param(param,player_types,**kwargs)
    elif param == 'bc':
        return limit_v_bc(player_types,**kwargs)
    elif param == 'rounds':
        return limit_v_rounds(player_types,**kwargs)
    else:
        raise

@plotter(limit_v_param, plot_exclusive_args = ['experiment','data'])
def limit_param_plot(param, player_types, data = [], **kwargs):
    fig = plt.figure()
    for hue in data['type'].unique():
        d = data[data['type']==hue]
        p = plt.plot(d[param], d['proportion'], label=hue)
    if param in ["beta"]:
        plt.axes().set_xscale('log',basex=10)
    if param in ['pop_size', 'rounds']:
        plt.axes().set_xscale('log',basex=2)
    elif param == 's':
        plt.axes().set_xscale('log')

    plt.xlabel(param)
    plt.ylim([0, 1.05])
    plt.yticks([0,0.5,1])
    sns.despine()
    plt.legend()
    plt.tight_layout()

def compare_limit_param(param, player_types, opponent_types, **kwargs):
    dfs = []
    for player_type in player_types:
        df = limit_v_param(param = param, player_types = (player_type,)+opponent_types,**kwargs)
        dfs.append(df[df['type']==player_type.short_name("agent_types")])
        
    return pd.concat(dfs,ignore_index = True)

@experiment(unpack = 'record', verbose = 3)
def limit_v_bc(player_types,**kwargs):
    params = default_params(**kwargs)
    record = []
    Bs = [1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
    for b in Bs:
        payoff = matchup_matrix(player_types = player_types, benefit = b, **kwargs)
        ssd = limit_analysis(payoff, **params)
        for t,p in zip(player_types,ssd):
            record.append({
                "bc":b,
                "type":t.short_name('agent_types'),
                "proportion":p
            })
    return record

@experiment(unpack = 'record', verbose = 2, memoize = True)
def bc_v_rounds(player_types, max_rounds, **kwargs):
    Warning('Memoize is one in the experiment!')
    params = default_params(**kwargs)
    records = []
    Rs = np.unique(np.geomspace(4, max_rounds, 10, dtype=int))
    # Bs = np.unique(np.geomspace(1.5, 10, 8, dtype=int))
    Bs = [1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    # Bs = [1.5, 2, 2.5, 3, 4, 5, 7, 10]
    for b in Bs:
        matrices = matchup_matrix_per_round(player_types, max_rounds, benefit = b, **kwargs)
        for rounds, payoffs in matrices:
            ssd = limit_analysis(payoffs, **params)
            #winner = max(zip(ssd,player_types),key = lambda t: t[0])[1]
            if rounds in Rs:
                for i, t in enumerate(player_types):
                    records.append({
                        "benefit":b,
                        "frequency":ssd[i],
                        "rounds":rounds,
                        "type": t.short_name('agent_types')
                    })
    return records

def compare_bc_v_rounds(player_types, max_rounds, opponent_types, **kwargs):
    dfs = []
    for player_type in player_types:
        df = bc_v_rounds((player_type,)+opponent_types, max_rounds, **kwargs)
        dfs.append(df[df['type']==player_type.short_name("agent_types")])
        
    return pd.concat(dfs,ignore_index = True)

@plotter(bc_v_rounds)
def bc_rounds_plot(player_types, data=[], **kwargs):
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot(index=args[1], columns=args[0], values=args[2])
        sns.heatmap(d, **kwargs)

    g = sns.FacetGrid(data = data, col = 'type')
    g.map_dataframe(draw_heatmap, 'benefit', 'rounds', 'frequency',  cbar=False, square=True, vmin=0, vmax=data['frequency'].max(),
                    #cmap=plt.cm.gray_r,
                    linewidths=.5)
    # .pivot(columns =, index = 'rounds', values = 'frequency')

    #g.set(
    #    yticks=[],
    #    xticks=[],
    #    xlabel='',
    #    ylabel='')

def AllC_AllD_race():
    today = "./plots/"+date.today().isoformat()+"/"
    prior = 0.5
    r = 10
    MRA = ReciprocalAgent
    ToM = ('self', AllC, AllD)
    opponents = (AllC, AllD)
    pop = (MRA(RA_K=0, agent_types = ToM, RA_prior=prior),
           MRA(RA_K=1, agent_types = ToM, RA_prior=prior),
           gTFT(y=1,p=1,q=0))
    
    for t in [0, 0.05]:
        limit_param_plot('s', player_types = pop, opponent_types = opponents, experiment = compare_limit_param, rounds = 10, tremble = t, file_name = 'contest_s_rounds=10_tremble=%0.2f' % t, plot_dir = today)
        limit_param_plot('s', player_types = pop, opponent_types = opponents, experiment = compare_limit_param, rounds = 100, tremble = t, file_name = 'contest_s_rounds=100_tremble=%0.2f' % t, plot_dir = today)
        limit_param_plot("rounds", player_types = pop, agent_types = ToM, opponent_types = opponents, experiment = compare_limit_param, tremble = t, file_name = 'contest_rounds_tremble=%0.2f' % t, plot_dir = today)
        
        limit_param_plot("RA_prior", player_types = (MRA(RA_K=0, agent_types = ToM), MRA(RA_K=1, agent_types = ToM)), opponent_types = opponents, experiment = compare_limit_param, rounds = r, tremble = t, file_name = 'contest_prior_tremble=%0.2f' % t, plot_dir = today)


#a_type, proportion = max(zip(player_types,ssd), key = lambda tup: tup[1])

def Pavlov_gTFT_race():
    today = "./plots/"+date.today().isoformat()+"/"
    TFT = gTFT(y=1,p=1,q=0)
    MRA = WeAgent#ReciprocalAgent
    r = 10
    
    # Replicate Nowak early 90s
    pop = (TFT, AllC, AllD, gTFT(y=1,p=.99,q=.33), Pavlov)
    for t in [0, 0.05]:
        limit_param_plot('s', pop, rounds = r, tremble = t, file_name = 'nowak_replicate_s_tremble=%.2f' % t, plot_dir = today)
    sim_plotter(5000, (0,0,100,0,0), player_types = pop, rounds = r, tremble = 0.05, mu=0.05, s=1, file_name ='nowak_replicate_sim_tremble=0.05', plot_dir = today)

    # Horse race against gTFT and Pavlov
    prior = 0.5
    beta = 10

    trembles = [0, 0.05]
    priors = [.5,.75]
    betas = [3,5,10]
    opponents = (TFT, gTFT(y=1,p=.99,q=.33), AllC, AllD, Pavlov)

    ToM = ('self',)+opponents
    agent = MRA(agent_types = ToM, beta = beta, RA_prior = prior)
    pop = (agent, AllC, AllD, TFT, gTFT(y=1,p=.99,q=.33), Pavlov)

    comparables = tuple(MRA(RA_prior=p, beta = b, agent_types=ToM) for p,b in product(priors,betas))

    for t in trembles:
        limit_param_plot('s', player_types = pop, rounds = r, tremble = t, file_name = 'horse_s_no_random_tremble=%0.2f' % t, plot_dir = today)
        limit_param_plot('rounds',
                         player_types = comparables,
                         opponent_types = opponents,
                         tremble = t,
                         experiment = compare_limit_param,
                         file_name = 'horse_rounds_no_random_tremble=%0.2f' % t,
                         plot_dir = today)

    # Add Random to the ToM
    ToM = ('self', TFT, gTFT(y=1,p=.99,q=.33), AllC, AllD, Pavlov, RandomAgent)
    for t in trembles:
        limit_param_plot('s', player_types = pop, rounds = r, tremble = t, file_name = 'horse_s_will_random_tremble=%.2f' % t,plot_dir = today)
        limit_param_plot('rounds',
                         player_types = comparables,
                         opponent_types = opponents,
                         tremble = t,
                         experiment = compare_limit_param,
                         file_name = 'horse_rounds_with_random_tremble=%.2f' % t,
                         plot_dir = today)

def bc_rounds_contest():
    WA = WeAgent
    prior = 0.5
    NRA = NiceReciprocalAgent
    MRA = ReciprocalAgent
    SA = SelfishAgent
    AA = AltruisticAgent
    AC = AllC
    AD = AllD
    TFT = gTFT(y=1,p=1,q=0)
    GTFT = gTFT(y=1,p=.99,q=.33)

    RA = WA(RA_prior = prior, agent_types = ('self', AllC, AllD))
    player_types = (RA, TFT, GTFT, Pavlov)

    for t in [0, 0.05]:
        bc_rounds_plot(
            max_rounds = 20,
            experiment = compare_bc_v_rounds,
            player_types = player_types,
            opponent_types = (AllC, AllD),
            tremble = t
        )

def bc_rounds_race():
    file_name = "ToM = %s, beta = %s, prior = %s, tremble = %s"
    plot_dir = "./plots/bc_rounds_race/"

    NRA = NiceReciprocalAgent
    MRA = ReciprocalAgent
    SA = SelfishAgent
    AA = AltruisticAgent
    AC = AllC
    AD = AllD
    TFT = gTFT(y=1,p=1,q=0)
    GTFT = gTFT(y=1,p=.99,q=.33)

    max_rounds = 20

    priors = [
        #.1,
        #.5,
        .75
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
        0,
        0.05
    ]

    for prior, ToM, beta in product(priors,ToMs,betas):
        RA = WeAgent(RA_prior = prior, agent_types = ToM, beta = beta)
        everyone = (RA, AC, AD, TFT, GTFT, Pavlov)
        for t in trembles:
            bc_rounds_plot(everyone, max_rounds = max_rounds, tremble = t,
                           plot_dir = plot_dir,
                           file_name = file_name % (ToM,beta,prior,t)
            )

def limit_rounds_race():
    file_name = "ToM = %s, beta = %s, prior = %s, tremble = %s"
    plot_dir = "./plots/limit_rounds_race/"

    NRA = NiceReciprocalAgent
    MRA = ReciprocalAgent
    SA = SelfishAgent
    AA = AltruisticAgent
    AC = AllC
    AD = AllD
    TFT = gTFT(y=1,p=1,q=0)
    GTFT = gTFT(y=1,p=.99,q=.33)

    max_rounds = 50

    priors = [
        #.1,
        #.5,
        #.75
        #.99
    ]

    ToMs = [
        ('self', AC, AD, TFT, GTFT, Pavlov)
    ]

    betas = [
        #.5,
        #1,
        #3,
        #5,
        10,
    ]

    trembles = [
        0,
        0.05
    ]

    for prior, ToM, beta in product(priors,ToMs,betas):
        RA = WeAgent(RA_prior = prior, agent_types = ToM, beta = beta)
        everyone = (RA, AC, AD, TFT, GTFT, Pavlov)
        for t in trembles:
            limit_param_plot(param = 'rounds', player_types = everyone, max_rounds = max_rounds, tremble = t,
                             plot_dir = plot_dir,
                             file_name = file_name % (ToM,beta,prior,t),
                             extension = '.png'
            )

if __name__ == "__main__":

    #AllC_AllD_race()
    Pavlov_gTFT_race()
    #bc_rounds_race()
    #limit_rounds_race()
    assert 0

    NRA = NiceReciprocalAgent
    MRA = ReciprocalAgent
    SA = SelfishAgent
    AA = AltruisticAgent
    AC = AllC
    AD = AllD
    TFT = gTFT(y=1,p=1,q=0)
    GTFT = gTFT(y=1,p=.99,q=.33)

    prior = 0.5
    Ks = [0,1]
    trembles = [0, 0.05]
    max_rounds = 20
    beta = 1
    
    #for t in trembles:
    #    bc_rounds_plot(
    #        max_rounds = max_rounds,
    #        experiment = compare_bc_v_rounds,
    #        player_types = tuple(MRA(RA_prior = prior, RA_K = k, agent_types = ('self', AC, AD)) for k in Ks) + (TFT, GTFT, Pavlov),
    #        opponent_types = (AC, AD),
    #        tremble = t,
    #        beta = beta,
    #        file_name = 'heat_tremble=%0.2f' % t)

    # assert 0
    
    everyone_ToM = ('self', AC, AD, TFT, GTFT, Pavlov, RandomAgent)
    RA = WeAgent(RA_prior = prior, agent_types = everyone_ToM, beta = beta)
    everyone = (RA, AC, AD, TFT, GTFT, Pavlov)
    for t in trembles:
        bc_rounds_plot(everyone, max_rounds = max_rounds, tremble = t)

    assert 0
    
    #limit_param_plot('bc',everyone)
    #limit_param_plot('rounds', everyone)

