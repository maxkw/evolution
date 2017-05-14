from __future__ import division
from collections import Counter, defaultdict
from itertools import product, permutations, izip
from utils import normalized, softmax, excluding_keys
from math import factorial
import numpy as np
from copy import copy
import operator
from experiments import NiceReciprocalAgent, SelfishAgent, ReciprocalAgent, AltruisticAgent
from experiment_utils import multi_call, experiment, plotter, MultiArg, cplotter, memoize, apply_to_args
import matplotlib.pyplot as plt
import seaborn as sns
from experiments import binary_matchup, memoize, matchup_matrix, matchup_plot
from params import default_genome
from indirect_reciprocity import gTFT, AllC, AllD, Pavlov, RandomAgent
from params import default_params
from steady_state import limit_analysis, complete_analysis
import pandas as pd

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

@experiment(unpack = 'record', memoize = False)
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

@experiment(unpack = 'record', memoize = False)
def limit_v_sim_param(param, player_types, **kwargs):
    if param == "RA_prior":
        Xs = np.linspace(0,1,21)[1:-1]
    elif param == "beta":
        Xs = logspace(.5,6,11)
    elif param == "rounds":
        Xs = np.unique(np.geomspace(1,20,10,dtype = int))
    else:
        raise

    record = []
    for x in Xs:
        payoffs = matchup_matrix(player_types = player_types, trials = 10, **dict(kwargs,**{param:x}))

        for t,p in zip(player_types, limit_analysis(payoffs, **default_params(**{param:x}))):
            record.append({
                param:x,
                "type":t.short_name("agent_types"),
                "proportion":p
            })
    return record


def limit_v_param(param,player_types,**kwargs):
    if param in ['rounds','beta','RA_prior']:
        return limit_v_sim_param(param,player_types,**kwargs)
    elif param in ['pop_size','s']:
        return limit_v_evo_param(param,player_types,**kwargs)
    elif param == 'bc':
        return limit_v_bc(player_types,**kwargs)
    else:
        raise

@plotter(limit_v_param, plot_exclusive_args = ['experiment','data'])
def limit_param_plot(param, player_types, data = [], **kwargs):
    fig = plt.figure()
    print data
    for hue in data['type'].unique():
        d = data[data['type']==hue]
        p = plt.plot(d[param], d['proportion'], label=hue)
    if param in ["beta"]:
        plt.axes().set_xscale('log',basex=10)
    if param == 'pop_size':
        plt.axes().set_xscale('log',basex=2)
    elif param == 's':
        plt.axes().set_xscale('log')
    plt.legend()

def compare_limit_param(param, player_types, opponent_types, **kwargs):
    dfs = []
    for player_type in player_types:
        df = limit_v_param(param = param, player_types = (player_type,)+opponent_types,**kwargs)
        dfs.append(df[df['type']==player_type.short_name("agent_types")])
        
    return pd.concat(dfs,ignore_index = True)

@experiment(unpack = 'record')
def limit_v_bc(player_types,**kwargs):
    params = default_params(**kwargs)
    record = []
    Bs = [1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
    for b in Bs:
        payoff = matchup_matrix(player_types = player_types, benefit = b, **kwargs)
        print payoff
        ssd = limit_analysis(payoff, **params)
        for t,p in zip(player_types,ssd):
            record.append({
                "bc":b,
                "type":t.short_name('agent_types'),
                "proportion":p
            })
    return record


def cb_v_rounds(player_types, **kwargs):
    max_rounds = 10
    for b in [1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]:
        pass

def AllC_AllD_race():
    prior = 0.5
    r = 10
    MRA = ReciprocalAgent
    ToM = ('self', AllC, AllD)
    opponents = (AllC, AllD)
    pop = (MRA(RA_K=0, agent_types = ToM, RA_prior=prior), MRA(RA_K=1, agent_types = ToM, RA_prior=prior), gTFT(y=1,p=1,q=0))
    for t in [0, 0.05]:
        limit_param_plot('s', player_types = pop, opponent_types = opponents, experiment = compare_limit_param, rounds = r, tremble = t)
        limit_param_plot("rounds", player_types = pop, agent_types = ToM, opponent_types = opponents, experiment = compare_limit_param, tremble = t, file_name = 'contest_rounds')
        limit_param_plot("RA_prior", player_types = (MRA(RA_K=1, agent_types = ToM), MRA(RA_K=0, agent_types = ToM)), opponent_types = opponents, experiment = compare_limit_param, rounds = r, tremble = t, file_name = 'contest_prior')


#a_type, proportion = max(zip(player_types,ssd), key = lambda tup: tup[1])

def Pavlov_gTFT_race():
    TFT = gTFT(y=1,p=1,q=0)
    MRA = ReciprocalAgent
    r = 10
    
    # Replicate Nowak early 90s
    pop = (TFT, AllC, AllD, gTFT(y=1,p=.99,q=.33), Pavlov)
    for t in [0, 0.05]:
        limit_param_plot('s', pop, rounds = r, tremble = t, file_name = 'nowak_replicate_s_tremble=%.2f' % t)
    sim_plotter(5000, (0,0,0,100,0), player_types = pop, rounds = r, tremble = 0.05, mu=0.05, s=1, file_name ='nowak_replicate_sim_tremble=0.05')

    # Horse race against gTFT and Pavlov
    prior = 0.5
    ToM = ('self', TFT, gTFT(y=1,p=.99,q=.33), AllC, AllD, Pavlov)
    pop = (MRA(RA_K=1, agent_types = ToM, RA_prior = prior), AllC, AllD, TFT, gTFT(y=1,p=.99,q=.33), Pavlov)
    opponents = (TFT, gTFT(y=1,p=.99,q=.33), AllC, AllD, Pavlov)
    trembles = [0, 0.05]
    for t in trembles:
        limit_param_plot('s', player_types = pop, rounds = r, tremble = t, file_name = 'horse_s_no_random_tremble=%0.2f' % t)
        limit_param_plot('rounds',
                         player_types = tuple(MRA(RA_K=k, RA_prior=p, agent_types=ToM) for k, p in product([1, 2], [.25, .5, .75])),
                         opponent_types = opponents,
                         tremble = t,
                         experiment = compare_limit_param,
                         file_name = 'horse_rounds_no_random_tremble=%0.2f' % t)

    # Add Random to the ToM
    ToM = ('self', TFT, gTFT(y=1,p=.99,q=.33), AllC, AllD, Pavlov, RandomAgent)
    for t in trembles:
        limit_param_plot('s', player_types = pop, rounds = r, tremble = t, file_name = 'horse_s_will_random_tremble=%.2f' % t)
        limit_param_plot('rounds',
                         player_types = tuple(MRA(RA_K=k, RA_prior=p, agent_types=ToM) for k, p in product([1, 2], [.25, .5, .75])),
                         opponent_types = opponents,
                         tremble = t,
                         experiment = compare_limit_param,
                         file_name = 'horse_rounds_with_random_tremble=%.2f' % t)

if __name__ == "__main__":
    AllC_AllD_race()
    Pavlov_gTFT_race()
    assert 0

    NRA = NiceReciprocalAgent
    MRA = ReciprocalAgent
    SA = SelfishAgent
    AA = AltruisticAgent
    AC = AllC
    AD = AllD
    TFT = gTFT(y=1,p=1,q=0)
    GTFT = gTFT(y=1,p=.99,q=.33)
    RA = MRA(RA_prior = .5, agent_types = (MRA, AC, AD, RandomAgent))
    everyone = (RA, AC, AD)
    limit_param_plot('bc',everyone)
