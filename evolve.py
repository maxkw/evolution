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

@multi_call()
@experiment(unpack = 'record', unordered = ['agent_types'], memoize = False)
def limit_steady_state(player_types = NiceReciprocalAgent, pop_size = 200, size = 3, agent_types = (AltruisticAgent, ReciprocalAgent, NiceReciprocalAgent, SelfishAgent), **kwargs):
    conditions = dict(locals(),**kwargs)
    del conditions['kwargs']
    matchup,types = RA_matchup_matrix(**excluding_keys(conditions,'pop_size','s'))
    ssd = limit_analysis(matchup, **conditions)
    #priors = np.linspace(0,1,size)

    return [{"agent_prior":prior,"percentage":pop} for prior,pop in zip(types,ssd)]

@plotter(limit_steady_state, plot_exclusive_args = ['data'])
def limit_plotter(player_types = ReciprocalAgent, rounds = MultiArg(range(1,21)), data = []):
    print data[data['rounds']==41]
    priors = list(sorted(list(set(data['agent_prior']))))
    print priors
    print len(priors)
    sns.pointplot(data = data, x = 'rounds', y = 'percentage', hue= 'agent_prior', hue_order = priors)
    #for prior in priors:
    #    sns.pointplot(data = data[data['agent_prior'] == prior], x = 'rounds', y = 'percentage', hue= 'agent_prior', hue_order = priors)


@multi_call()
@experiment(unpack = 'record', unordered = ['agent_types'], memoize = False)
def complete_steady_state(player_types = NiceReciprocalAgent, pop_size = 100, size = 3, agent_types = (AltruisticAgent, ReciprocalAgent, NiceReciprocalAgent, SelfishAgent), **kwargs):
    conditions = dict(locals(),**kwargs)
    del conditions['kwargs']
    matchup,types = RA_matchup_matrix(**conditions)
    ssd = complete_analysis(matchup, **conditions)
    #priors = np.linspace(0,1,size)

    return [{"agent_prior":prior,"percentage":pop} for prior,pop in zip(types,ssd)]

@plotter(complete_steady_state, plot_exclusive_args = ['data'])
def complete_plotter(player_types = ReciprocalAgent, rounds = MultiArg(range(1,21)), data = []):
    sns.factorplot(data = data, x = 'rounds', y = 'percentage', hue ='agent_prior', col ='player_types')
    
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
        action_index = np.random.choice(len(probs), 1, p = probs)
        (b,d) = actions[action_index]
        pop = pop + I[b]-I[d]

@experiment(unpack = 'record', memoize = False)
def agent_simulation(generations, pop, player_types, agent_types= None, **kwargs):
    if agent_types is None:
        agent_types = player_types

    payoffs = matchup_matrix(player_types = player_types, agent_types = agent_types, **kwargs)
    params = default_params()
    populations = agent_sim(payoffs, pop, params['s'], params['mu'])

    record = []
    for n, pop in izip(xrange(generations), populations):
        for t, p in zip(player_types, pop):
            record.append({'generation' : n,
                           'type' : t,
                           'population' : p})
    return record


@plotter(agent_simulation,plot_exclusive_args = ['data'])
def sim_plotter(generations, pop, player_types, agent_types= None, data =[]):
    for hue in data['type'].unique():
        plt.plot(data[data['type']==hue]['generation'], data[data['type']==hue]['population'], label=hue)

    plt.legend()

@experiment(unpack = 'record', memoize = False)
def limit_v_evo_param(param, player_types, agent_types = None, **kwargs):
    payoffs = matchup_matrix(player_types = player_types, trials = 10, **kwargs)
    matchup_plot(player_types = player_types, trials = 10, **kwargs)

    if param == 'pop_size':
        # Xs = range(2, 2**10)
        Xs = np.unique(np.geomspace(2, 2**10, 200, dtype=int))
    elif param == 's':
        Xs = logspace(start = .001, stop = 10, samples = 100)
    else:
        print param
        raise
    #print Xs

    params = default_params()
    
    record = []
    for x in Xs:
        params[param] = x
        for t, p in zip(player_types, limit_analysis(payoffs, **params)):
            try:
                t_name = t.short_name("agent_types")
            except:
                t_name = t.__name__
            record.append({
                param : x,
                "type" : t_name,
                "proportion" : p
            })
    return record

@plotter(limit_v_evo_param)
def limit_evo_plot(param, player_types, data = [], **kwargs):
    fig = plt.figure()
    for hue in data['type'].unique():
        d = data[data['type']==hue]
        p = plt.plot(d[param], d['proportion'], label=hue)
    if param == 'pop_size':
        plt.axes().set_xscale('log',basex=2)
    elif param == 's':
        plt.axes().set_xscale('log')
    plt.legend()

@experiment(unpack = 'record', memoize = False)
def limit_v_sim_param(param, player_types, **kwargs):
    
    if param == "RA_prior":
        Xs = np.linspace(0,1,41)[1:-1]
    elif param == "beta":
        Xs = logspace(0,1,11)
    elif param == "rounds":
        Xs = np.unique(np.geomspace(1,100,20,dtype = int))
    else:
        raise

    record = []
    for x in Xs:
        payoffs = matchup_matrix(player_types = player_types, trials = 10, **dict(kwargs,**{param:x}))
        for t,p in zip(player_types, limit_analysis(payoffs, **default_params(**{param:x}))):
            try:
                t_name = t.short_name("agent_types")
            except:
                t_name = t.__name__
            record.append({
                param:x,
                "type":t_name,
                "proportion":p
            })
    return record

@plotter(limit_v_sim_param)
def limit_sim_plot(param, player_types, data = [], **kwargs):
    fig = plt.figure()
    for hue in data['type'].unique():
        d = data[data['type']==hue]
        p = plt.plot(d[param], d['proportion'], label=hue)
    if param in ["beta"]:
        plt.axes().set_xscale('log',basex=10)
    plt.legend()

@experiment(unpack = 'record', memoize = False)
def limit_v_compare_param(param, player_types, opponent_types = tuple(), **kwargs):
    if param == "rounds":
        #Xs = np.unique(np.geomspace(1,100,20,dtype = int))
        Xs = range(1,21)
    
    else:
        raise

    record = []
    for x in Xs:
        for player_type in player_types:
            params = default_params(**kwargs)
            params[param] = x

            players = (player_type,)+opponent_types
            payoffs = matchup_matrix(player_types = players, **dict(kwargs,**{param:x}))
            for t,p in zip(players, limit_analysis(payoffs, **params)):
                if t == player_type:
                    record.append({
                        param:x,
                        "type":t.short_name(without = ["agent_types"]),
                        "proportion":p
                    })
    return record

@plotter(limit_v_sim_param)
def limit_sim_plot(param, player_types, data = [], **kwargs):
    fig = plt.figure()
    for hue in data['type'].unique():
        d = data[data['type']==hue]
        p = plt.plot(d[param], d['proportion'], label=hue)
    if param in ["beta"]:
        plt.axes().set_xscale('log',basex=10)
    plt.legend()

def limit_v_param(param,player_types,**kwargs):
    if param in ['rounds','beta','RA_prior']:
        return limit_v_sim_param(param,player_types,**kwargs)
    elif param in ['pop_size','s']:
        return limit_v_evo_param(param,player_types,**kwargs)
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
    if param == 'pop_size':
        plt.axes().set_xscale('log',basex=2)
    elif param == 's':
        plt.axes().set_xscale('log')
    plt.legend()

@experiment(unpack = 'record', memoize = False)
def compare_limit_evo(param, player_types, opponent_types = tuple(), **kwargs):
    if param == 'pop_size':
        # Xs = range(2, 2**10)
        Xs = np.unique(np.geomspace(2, 2**10, 200, dtype=int))
    elif param == 's':
        Xs = logspace(start = .001, stop = 10, samples = 100)
    else:
        print param
        raise
    params = default_params(**kwargs)
    record = []
    for player_type in player_types:
        players = (player_type,)+opponent_types
        payoffs = matchup_matrix(player_types = players,  **kwargs)
        for x in Xs:
            for t,p in zip(players, limit_analysis(payoffs, **dict(params,**{param:x}))):
                if t == player_type:
                    record.append({
                        param:x,
                        "type":t_name,
                        "proportion":p
                    })
    return record

def compare_limit_param(param,player_types,opponent_types,**kwargs):
    dfs = []
    for player_type in player_types:
        df = limit_v_param(param = param, player_types = (player_type,)+opponent_types,**kwargs)
        try:
            t_name = player_type.short_name("agent_types")
        except:
            t_name = player_type.__name__
        dfs.append(df[df['type']==t_name])
    return pd.concat(dfs,ignore_index = True)




if __name__ == "__main__":

    NRA = NiceReciprocalAgent
    MRA = ReciprocalAgent
    SA = SelfishAgent
    AA = AltruisticAgent
    AC = AllC
    AD = AllD
    TFT = gTFT(y=1,p=1,q=0)
    GTFT = gTFT(y=1,p=.99,q=.33)
    tremble = 0
    beta = 3
    prior = .5
    rounds = [10, 50, 100, 200]
    RAs = [
            MRA,
            # NRA
    ]

    M = MRA(RA_prior = .5)
    GTFT = gTFT(y=1,p=.99)
    M = MRA(RA_prior = .5, RA_K = 2, agent_types = ('self', AC, AD, TFT, Pavlov, RandomAgent))
    F = MRA(RA_prior = .75, RA_K = 2, agent_types = ('self', AC, AD, TFT, Pavlov))
    priored = tuple(MRA(RA_prior = n, RA_K = 0, agent_types = ('self', AC, AD, TFT, Pavlov, GTFT)) for n in [.1,.3,.5,.7,.9])
    limit_param_plot('rounds', player_types = (F, AC, AD, TFT, Pavlov, GTFT), tremble = .05)
    assert 0
    #matchup_plot(player_types = (M,AllC,AllD), rounds = 50, tremble = .1)
    #assert 0
    #limit_param_plot("pop_size", player_types = (M,F), opponent_types = (AC, AD, TFT, Pavlov), rounds = 50, tremble = 0)
    assert 0
    
    limit_param_plot("pop_", player_types = priored, opponent_types = (AC,AD), agent_types = ('self',AC,AD), experiment = compare_limit_param)
    #limit_sim_plot(param = 'rounds', player_types = (M,AC,AD), agent_types = (M,AC,AD))
    #limit_sim_plot(experiment = limit_v_compare_param, param='rounds', player_types = (TFT,M), opponent_types = (AC,AD), agent_types = (M,AC,AD))
    assert 0
    limit_sim_plot(experiment = limit_v_compare_param, param='rounds', player_types = (MRA(RA_K = 0),MRA(RA_K=1),MRA(RA_K = 2)), opponent_types = (AC,AD), agent_types = ('self',AC,AD))
    p = (MRA,AC,AD)
    
    limit_sim_plot(param = "RA_prior", player_types = p, agent_types = p)
    
    #limit_evo_plot(experiment = compare_limit_evo, param = 's', player_types = (TFT,M),opponent_types = (AC, AD), agent_types = (M,AC,AD))
    #limit_evo_plot(experiment = compare_limit_evo, param = 'pop_size', player_types = (TFT,M),opponent_types = (AC, AD), agent_types = (M,AC,AD))
    assert 0
    Ks = range(2)
    for RA, K in product(RAs, Ks):

        pop1 = (RA,AA,SA)
        # limit_evo_plot('pop_size', pop1)
        # limit_evo_plot('s', pop1)
        # limit_sim_plot('RA_prior', pop1)
        pop2 = (RA,AC,AD)
        # limit_evo_plot('pop_size', pop2, agent_types = pop1, RA_prior = 0.5)
        # limit_evo_plot('s', pop2, agent_types = pop1, RA_prior = 0.5)
        # limit_evo_plot('pop_size', pop2)
        # limit_evo_plot('s', pop2)
        # # limit_sim_plot('RA_prior', pop2)
        # # limit_sim_plot('beta', pop2)
        # # limit_evo_plot('pop_size', pop2, tremble = tremble)
        pop3 = (RA,AC,AD,TFT)
        limit_evo_plot('pop_size', pop3, agent_types = pop3, RA_prior=prior, RA_K=K, beta=beta, rounds=r)
        # limit_evo_plot('s', pop3, agent_types = pop3, RA_prior=prior, RA_K=K, beta=beta, rounds=r)

        # sim_plotter(50000, (0,0,100,0), player_types = pop3, agent_types = pop1, K=K, beta = beta, RA_prior = prior, rounds=50)
        
        # old_pop = (TFT,AC,AD)
        # limit_evo_plot('pop_size', old_pop, rounds=r)
    # limit_evo_plot('s', old_pop)
