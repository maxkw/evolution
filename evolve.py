from __future__ import division
from collections import Counter, defaultdict
from itertools import product, permutations, izip
from utils import normalized, softmax, excluding_keys, logspace, int_logspace, memoized
from math import factorial
import numpy as np
from copy import copy
from experiment_utils import multi_call, experiment, plotter, MultiArg, memoize, apply_to_args
import matplotlib.pyplot as plt
import seaborn as sns
from experiments import binary_matchup, memoize, matchup_matrix, matchup_plot,matchup_matrix_per_round
from params import default_genome, default_params
import agents as ag
from agents import gTFT, AllC, AllD, Pavlov, RandomAgent, WeAgent, SelfishAgent, ReciprocalAgent, AltruisticAgent
from steady_state import mm_to_limit_mcp, mcp_to_ssd, steady_state, mcp_to_invasion, limit_analysis, evo_analysis, simulation
#from steady_state import cp_to_transition, complete_softmax, matchups_and_populations, sim_to_rmcp
import pandas as pd
from datetime import date
from agents import leading_8_dict, shorthand_to_standing

TODAY = "./plots/"+date.today().isoformat()+"/"

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

    # Populations is an infinite iterator so need to combine it with a
    # finite iterator which sets the number of generations to look at.
    for n, pop in izip(xrange(generations), populations):
        for t, p in zip(player_types, pop):
            record.append({'generation' : n,
                           'type' : t.short_name('agent_types'),
                           'population' : p})
    return record



# @plotter(agent_simulation,plot_exclusive_args = ['data'])
# def sim_plotter(generations, pop, player_types, data =[]):
#     for hue in data['type'].unique():
#         plt.plot(data[data['type']==hue]['generation'],
#                  data[data['type']==hue]['population'].astype(int),
#                  label=hue)

#     plt.legend()
#     plt.xlabel(r'Time $\leftarrow$')
#     plt.ylabel('Count')
#     sns.despine()
#     plt.tight_layout()

def complete_agent_sim(player_types, s, **kwargs):
    type_to_index = dict(map(reversed, enumerate(sorted(player_types))))
    original_order = np.array([type_to_index[t] for t in player_types])
    #player_types = sorted(player_types)
    types, initial_pop = zip(*player_types)
    print types
    print sorted(types)
    pop_size = sum(initial_pop)

    rmcp = sim_to_rmcp(types, pop_size = pop_size, analysis_type = 'complete', **kwargs)
    rcp = np.array([mcp[0] for mcp in rmcp])
    _, populations = matchups_and_populations(player_types, pop_size, analysis_type = 'complete')

    softmax_rcp = complete_softmax(rcp,populations,s)
    transition = cp_to_transition(softmax_rcp[-1], populations, pop_size = pop_size, **kwargs).T

    pop_ids = np.array(range(len(populations)))
    pop_to_id = dict(map(reversed,enumerate(populations)))
    id_to_pop = dict(enumerate(populations))

    pop_id = pop_to_id[initial_pop]
    while True:
        yield np.array(id_to_pop[pop_id])
        t = transition[pop_id]
        neighbors = pop_ids[t!=0]
        trans_probs = t[t!=0]
        pop_id = np.random.choice(neighbors, p=trans_probs)

def complete_sim_live(player_types, start_pop, s=1, mu = .000001, seed = 0, **kwargs):
    np.random.seed(seed)
    pop_size = sum(start_pop)
    type_count = len(player_types)
    I = np.identity(type_count)
    pop = np.array(start_pop)
    assert type_count == len(pop)

    type_to_index = dict(map(reversed, enumerate(sorted(player_types))))
    original_order = np.array([type_to_index[t] for t in player_types])
    #player_types = sorted(player_types)
    @memoize
    def sim(pop):
        #print np.array(range(3))[original_order]
        f = simulation(zip(player_types,pop), trials = 20, **kwargs)[-1]
        print 'types', player_types

        print 'sorted', sorted(player_types)
        print 'fixed',np.array(sorted(player_types))[original_order]
        print "pop", pop
        non_players = np.array(pop)==0
        print "raw", f
        player_payoffs = f[non_players==False]
        f[non_players == False] = softmax(player_payoffs,s)
        f[non_players] = 0
        print "soft", f
        #assert 0
        
        return f

    while True:
        yield pop
        fitnesses = sim(pop)
        actions = [(b,d) for b,d in permutations(xrange(type_count),2) if pop[d]!=0]
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
        #print actions
        #print probs
        action_index = np.random.choice(len(probs), 1, p = probs)[0]
        (b,d) = actions[action_index]
        pop = map(int,pop + I[b]-I[d])

@experiment(unpack = 'record', memoize = False)
def complete_agent_simulation(generations, player_types, start_pop, s, seed = 0, **kwargs):
    populations = complete_sim_live(player_types, start_pop, s, seed = seed, **kwargs)
    record = []

    #types,_ = zip(*player_types)
    # Populations is an infinite iterator so need to combine it with a
    # finite iterator which sets the number of generations to look at.
    for n, pop in izip(xrange(generations), populations):
        for t, p in zip(player_types, pop):
            record.append({'generation' : n,
                           'type' : t.short_name('agent_types'),
                           'population' : p})
    return record

@plotter(complete_agent_simulation,plot_exclusive_args = ['data'])
def complete_sim_plot(generations, player_types, data =[], **kwargs):
    plt.figure(figsize = (3.5,3))
    for hue in data['type'].unique():
        plt.plot(data[data['type']==hue]['generation'],
                 data[data['type']==hue]['population'].astype(int),
                 label=hue)

    make_legend()
    plt.xlabel(r'Time $\rightarrow$')
    
    plt.ylabel('Count')
    sns.despine()
    plt.tight_layout()

def splits(n):
    """
    starting from 0, produces the sequence 2, 3, 5, 9, 17...
    This sequence is the answer to the question 'if you have 2 points and find the midpoint,
    then find the midpoints of each adjacent pair of points, and do that over and over,
    how many points will you have after n iterations?'

    excellent for populating real-valued parameter ranges since they recycle points
    when used as
    np.linspace(min, max, splits(n))
    """
    assert n>=0
    i = n+1
    return (2**i-0**i)/2 + 1

#@experiment(unpack = 'record', memoize = False, verbose = 3)
def ssd_v_param(param, player_types, return_rounds=False, **kwargs):
    """
    This should be optimized to reflect the fact that
    in terms of complexity
    rounds=s>=pop_size>=anything_else

    for 'rounds' you do a single analysis and then plot each round
    for 's' you should only make the RMCP once and then do the analysis for different s
    for 'pop_size',
       in direct reciprocity you only need to make the payoff matrix once
       in indirect you need to make an rmcp for each value of pop_size
    for anything else, the whole thing needs to be rerun


    """
    # Test to make sure each agent interacts with a new agent each
    # time. Otherwise its not true 'indirect' reciprocity.

    Xs = {
        # 'RA_prior': np.linspace(0,1,21)[1:-1],
        'RA_prior': np.linspace(0, 1, 21),
        'benefit': np.linspace(2, 10, 5),
        'beta': np.linspace(1, 11, 6),
        'pop_size': np.unique(np.geomspace(2, 2**10, 100, dtype=int)),
        's': logspace(start = .001, stop = 1, samples = 100),
        'observability':np.linspace(0,1,splits(2)),
        #'tremble': np.linspace(0, 0.4, 41),
        'tremble':np.round(np.linspace(0,.4, splits(3)),2),
        # 'tremble': np.linspace(0, 0.05, 6),
        'intervals' : [2, 4, 8],
        #'gamma' : [0, .5, .8, .9]
        'expected_interactions': np.linspace(1,10,splits(2)),
    }
    record = []
    
    if param == "rounds":
        expected_pop_per_round = evo_analysis(player_types = player_types, **kwargs)
        for r, pop in enumerate(expected_pop_per_round, start = 1):
            for t, p in zip(player_types, pop):
                record.append({
                    'rounds': r,
                    'type': t.short_name('agent_types'),
                    'proportion': p
                })
                
        return pd.DataFrame.from_records(record)

    elif param in Xs:
        for x in Xs[param]:
            expected_pop_per_round = evo_analysis(player_types = player_types, **dict(kwargs,**{param:x}))


            if return_rounds:
                start = 1
            else:
                start = len(expected_pop_per_round)-1

            for r, pop in enumerate(expected_pop_per_round[start:], start = 1):
                for t, p in zip(player_types, pop):
                    record.append({
                        param: x,
                        'rounds': r,
                        'type': t.short_name('agent_types'),
                        'proportion': p
                    })

        return pd.DataFrame.from_records(record)

    else:
        raise Exception('Param %s is not implemented. Add the range to Xs' % param)

def compare_ssd_v_param(param, player_types, opponent_types, **kwargs):
    dfs = []
    for player_type in player_types:
        df = ssd_v_param(param = param, player_types = (player_type,)+opponent_types, **kwargs)
        dfs.append(df[df['type']==player_type.short_name("agent_types")])
    return pd.concat(dfs, ignore_index = True)

def make_legend():
    legend = plt.legend(frameon=True)
    for texts in legend.get_texts():
        if 'WeAgent' in texts.get_text():
            texts.set_text('Reciprocal')
        elif 'SelfishAgent' in texts.get_text():
            texts.set_text('Selfish')
        elif 'AltruisticAgent' in texts.get_text():
            texts.set_text('Altruistic')

    return legend
    

@plotter(ssd_v_param, plot_exclusive_args = ['experiment','data', 'stacked', 'graph_kwargs'])
def limit_param_plot(param, player_types, data = [], stacked = False, graph_kwargs={}, **kwargs):
    fig, ax = plt.subplots()

    # TODO: Investigate this
    # Some weird but necessary data cleaning
    data[data['proportion']<0] = 0
    data = data[data['type']!=0]
    data = data[[param, 'proportion', 'type']].pivot(columns='type', index=param, values='proportion')
    type_order = dict(map(reversed,enumerate([t.short_name('agent_types') for t in player_types])))
    data.reindex_axis(sorted(data.columns, key = lambda t:type_order[t]), 1)

    if stacked:
        if param in ['expected_interactions']:
            data.plot.area(stacked = True, ylim = [0, 1], xlim = [1,10], figsize = (3.5,3), legend=False)
        elif param in ['rounds']:
             data.plot.area(stacked = True, ylim = [0, 1], figsize = (3.5,3), legend=False)
        else:
            data.plot.area(stacked = True, ylim = [0, 1], figsize = (3.5,3), legend=False)
        if 'legend' not in graph_kwargs or graph_kwargs['legend']:
            legend = make_legend()
            # for texts in legend.get_texts():
            #     if texts.get_text() == 'WeAgent':
            #         texts.set_text('Reciprocal')
            #     elif texts.get_text() == 'SelfishAgent':
            #         texts.set_text('Selfish')
            #     elif texts.get_text() == 'AltruisticAgent':
            #         texts.set_text('Altruistic')
        if param in ['rounds','expected_interactions']:
            plt.xlabel('Expected Interactions\n' r'$1/(1-\gamma)$')
            if param == 'expected_interactions':
                plt.xticks(range(1,11))
        elif param == 'observability':
            plt.xlabel('Probability of observation\n' r'$\omega$')
            plt.xticks([0, .2, .4, .6, .8, 1])
        elif param == 'tremble':
            plt.xlabel(r'Noise Probability ($\epsilon$)')
        else:
            plt.xlabel(param)


    else:
        data.plot(ax=ax, ylim = [0, 1.05], **graph_kwargs)
            
        if param in ['pop_size']:
            plt.axes().set_xscale('log',basex=2)
        elif param == 's':
            plt.axes().set_xscale('log')

        # elif param in ["beta"]:
        #     plt.axes().set_xscale('log',basex=10)
        # elif if param in ['rounds']:
        #     pass

        plt.xlabel(param)
        plt.legend()
        
    plt.yticks([0,0.5,1])
    plt.ylabel('Frequency')
        
    sns.despine()
    plt.tight_layout()

def param_v_rounds(param, player_types, rounds, **kwargs):
    return ssd_v_param(param, player_types, return_rounds=True, rounds=rounds, **kwargs)

def compare_param_v_rounds(param, player_types, opponent_types, direct, rounds, **kwargs):
    dfs = []
    for player_type in player_types:
        df = param_v_rounds(param, (player_type,)+opponent_types, direct, rounds, **kwargs)
        dfs.append(df[df['type']==player_type.short_name("agent_types")])
    return pd.concat(dfs,ignore_index = True)

@plotter(param_v_rounds)
def param_v_rounds_heat(param, player_types, experiment=param_v_rounds, data=[], **kwargs):
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot(index=args[1], columns=args[0], values=args[2])
        sns.heatmap(d, **kwargs)
        
    g = sns.FacetGrid(data = data, col = 'type')
    g.map_dataframe(draw_heatmap, param, 'rounds', 'proportion',  cbar=False, square=True,
                    vmin=0,
                    vmax=1,
                    # vmax=data['frequency'].max(),
                    cmap=plt.cm.gray_r,
                    linewidths=.5)

@plotter(param_v_rounds)
def param_v_rounds_plot(param, player_types, experiment=param_v_rounds, data=[], **kwargs):
    g = sns.FacetGrid(data = data, col = 'type', hue=param)
    g.map(plt.plot, 'rounds', 'proportion')
    g.set_titles("{col_name}")
    g.axes[0][0].legend(title=param, loc='best')

if __name__ == "__main__":
    opponents = (
        ag.AltruisticAgent,
        ag.SelfishAgent,
    )

    ToM = ('self', ) + opponents
    pop = opponents + (WeAgent(agent_types=ToM, prior = .5),)
    #pop = opponents 
    # for n in range(20):
    complete_sim_plot(generations = 1500,
                      rounds = 10,
                      player_types = pop,
                      start_pop = (0,10,0),
                      game = 'direct',
                      beta = 10,
                      #RA_prior = 0.5,
                      cost = 1,
                      benefit = 3,
                      plot_dir = TODAY,
                      file_name = 'agentsim',
                      # gamma = .9,
                      # rounds = 100,
                      s = 1,
                      mu= .001,
                      observability = 0,
                      seed = 0
    )
