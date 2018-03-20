from __future__ import division
import pandas as pd
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product, permutations, izip

import agents as ag

from utils import excluding_keys, softmax, memoize
from experiment_utils import experiment, plotter
from experiments import matchup_matrix
from params import default_params
from agents import WeAgent
from steady_state import evo_analysis, simulation
from steady_state import simulation_from_dict, matchups_and_populations
from multiprocessing import Pool

TODAY = "./plots/"+date.today().isoformat()+"/"

def complete_sim_live(player_types, start_pop, s=1, mu = .000001, seed = 0, **kwargs):
    pop_size = sum(start_pop)
    type_count = len(player_types)
    I = np.identity(type_count)
    pop = np.array(start_pop)
    assert type_count == len(pop)

    matchups, populations = matchups_and_populations(player_types, pop_size, "complete")
    matchup_pop_dicts = [dict(player_types = zip(*pop_pair), **kwargs) for pop_pair in product(matchups, populations)]
    pool = Pool(8)
    payoffs = pool.map(simulation_from_dict, matchup_pop_dicts)
    pool.close()

    type_to_index = dict(map(reversed, enumerate(sorted(player_types))))
    original_order = np.array([type_to_index[t] for t in player_types])

    @memoize
    def sim(pop):
        f = simulation(zip(player_types,pop), **kwargs)[-1]

        non_players = np.array(pop)==0

        player_payoffs = f[non_players==False]
        f[non_players == False] = softmax(player_payoffs, s)
        f[non_players] = 0
        
        return f

    np.random.seed(seed)
    while True:
        yield pop

        fitnesses = sim(pop)
        actions = [(b,d) for b,d in permutations(xrange(type_count),2) if pop[d]!=0]
        probs = []
        for b,d in actions:
            death_odds = pop[d] / pop_size
            birth_odds = (1-mu) * fitnesses[b] + mu * (1/type_count)
            prob = death_odds * birth_odds
            probs.append(prob)
            
        actions.append((0, 0))
        probs.append(1 - np.sum(probs))

        probs = np.array(probs)
        action_index = np.random.choice(len(probs), 1, p = probs)[0]
        (b,d) = actions[action_index]
        pop = map(int,pop + I[b]-I[d])

@experiment(unpack = 'record', memoize = False)
def complete_agent_simulation(generations, player_types, start_pop, s, seed = 0, trials = 100, **kwargs):
    populations = complete_sim_live(player_types, start_pop, s, seed = seed, trials = trials, **kwargs)
    record = []

    # Populations is an infinite iterator so need to combine it with a
    # finite iterator which sets the number of generations to look at.
    for n, pop in izip(xrange(generations), populations):
        for t, p in zip(player_types, pop):
            record.append({'generation' : n,
                           'type' : t.short_name('agent_types'),
                           'population' : p})
            
    return record

@plotter(complete_agent_simulation, plot_exclusive_args = ['data', 'graph_kwargs', 'stacked'])
def complete_sim_plot(generations, player_types, data =[], graph_kwargs={}, **kwargs):
    data['population'] = data['population'].astype(int)
    data = data[['generation', 'population', 'type']].pivot(columns='type', index='generation', values='population')
    type_order = dict(map(reversed,enumerate([t.short_name('agent_types') for t in player_types])))
    data.reindex(sorted(data.columns, key = lambda t:type_order[t]), axis = 1)

    fig, ax = plt.subplots(figsize = (3.5,3))
    data.plot(ax = ax, legend=False,
              ylim=[0, sum(kwargs['start_pop'])],
              xlim=[0, generations],
              **graph_kwargs)
    
    make_legend()
    plt.xlabel(r'Time $\rightarrow$')
    
    plt.ylabel('Count')
    sns.despine()
    plt.tight_layout()


#@experiment(unpack = 'record', memoize = False, verbose = 3)

def ssd_v_param(param, player_types, return_rounds=False, record_params ={}, **kwargs):
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

    # Xs = {
    #     # 'RA_prior': np.linspace(0,1,21)[1:-1],
    #     'RA_prior': np.linspace(0, 1, 21),
    #     'benefit': np.linspace(2, 10, 5),
    #     'beta': np.linspace(1, 11, 6),
    #     'pop_size': np.unique(np.geomspace(2, 2**10, 100, dtype=int)),
    #     's': logspace(start = .001, stop = 1, samples = 100),
    #     'observability': np.round(np.linspace(0, 1, 11),2),
    #     'tremble': np.round(np.linspace(0, 0.4, 41),2),
    #     'expected_interactions': np.linspace(1, 10, 10),
    #     'intervals' : [2, 4, 8],
    # }
    
    record = []

    if param == "rounds":
        expected_pop_per_round = evo_analysis(player_types = player_types, **kwargs)
        for r, pop in enumerate(expected_pop_per_round, start = 1):
            for t, p in zip(player_types, pop):
                record.append(dict({
                    'rounds': r,
                    'type': t.short_name('agent_types'),
                    'proportion': p
                }, **record_params))
                
        return pd.DataFrame.from_records(record)

    if 'param_vals' in kwargs:
        vals = kwargs['param_vals']
        del kwargs['param_vals']

        for x in vals:
            expected_pop_per_round = evo_analysis(player_types = player_types, **dict(kwargs,**{param:x}))

            # Only return all of the rounds if return_rounds is True
            if return_rounds:
                start = 1
            else:
                start = len(expected_pop_per_round)-1

            for r, pop in enumerate(expected_pop_per_round[start:], start = 1):
                for t, p in zip(player_types, pop):
                    record.append(dict({
                        param: x,
                        'rounds': r,
                        'type': t.short_name('agent_types'),
                        'proportion': p
                    }, **record_params))

        return pd.DataFrame.from_records(record)

    else:
        raise Exception('`param_vals` %s is not defined. Pass this variable' % param)


def ssd_v_params(params, player_types, return_rounds = False, **kwargs):
    '''`params`: <dict> with <string> keys that name the parameter and
    values that are lists of the parameters to range over.

    '''
    
    record = []
    
    for pvs in product(*params.values()):
        ps = dict(zip(params, pvs))
        expected_pop_per_round = evo_analysis(player_types = player_types, **dict(kwargs, **ps))

        # Only return all of the rounds if return_rounds is True
        if return_rounds:
            start = 1
        else:
            start = len(expected_pop_per_round)-1

        for r, pop in enumerate(expected_pop_per_round[start:], start = 1):
            for t, p in zip(player_types, pop):
                record.append(dict({'rounds': r,
                                    'type': t.short_name('agent_types'),
                                    'proportion': p},
                                   **ps))

    return pd.DataFrame.from_records(record)

@plotter(ssd_v_params)
def params_heat(params, player_types, data = [], graph_kwargs={}, **kwargs):
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot(index=args[1], columns=args[0], values=args[2])
        ax = sns.heatmap(d, **kwargs)
        ax.invert_yaxis()
        
    assert len(params)==2
    g = sns.FacetGrid(data = data, col = 'type')
    g.map_dataframe(draw_heatmap, params.keys()[1], params.keys()[0], 'proportion',  cbar=False, square=True,
                    vmin=0,
                    vmax=1,
                    # vmax=data['frequency'].max(),
                    cmap=plt.cm.gray_r,
                    linewidths=.5)

def ssd_param_search(param, param_lim, player_types, target_player, tolerance, **kwargs):
    def is_mode(ep):
        # takes in the output of a evo_analysis and returns whether or not the target is the mode. 
        pass

    # First check the max and min and see if target_player is *ever* the mode. If not return some kind of 0

    # If min or max gives you a mode. Check the tolerance, which is defined as the difference in proportion between the mode and second most prevalent. The smaller the tolerance the more fine grained the algorithm will search. 

    # This should just be a recursive binary search
    current = np.mean(param_lim)

    
    
    
    
    

def compare_ssd_v_param(param, player_types, opponent_types, **kwargs):
    dfs = []
    for player_type in player_types:
        df = ssd_v_param(param = param, player_types = (player_type,)+opponent_types, **kwargs)
        dfs.append(df[df['type']==player_type.short_name("agent_types")])
    return pd.concat(dfs, ignore_index = True)

def make_legend():
    legend = plt.legend(frameon=True)
    for i, texts in enumerate(legend.get_texts()):
        if 'WeAgent' in texts.get_text():
            texts.set_text('Reciprocal')
        elif 'SelfishAgent' in texts.get_text():
            texts.set_text('Selfish')
        elif 'AltruisticAgent' in texts.get_text():
            texts.set_text('Altruistic')

    return legend

@plotter(ssd_v_param, plot_exclusive_args = ['experiment','data', 'stacked', 'graph_kwargs', 'graph_funcs'])
def limit_param_plot(param, player_types, data = [], stacked = False, graph_funcs=None, graph_kwargs={}, **kwargs):
    fig, ax = plt.subplots(figsize = (3.5, 3))

    # TODO: Investigate this, some weird but necessary data cleaning
    data[data['proportion']<0] = 0
    data = data[data['type']!=0]

    data = data[[param, 'proportion', 'type']].pivot(columns='type', index=param, values='proportion')
    type_order = dict(map(reversed,enumerate([t.short_name('agent_types') for t in player_types])))
    data.reindex(sorted(data.columns, key = lambda t:type_order[t]), axis = 1)

    if stacked:
        data.plot.area(stacked = True, ax=ax, ylim = [0, 1], legend=False, **graph_kwargs)
            
        if 'legend' not in graph_kwargs or graph_kwargs['legend']:
            legend = make_legend()

        if param == 'rounds':
            plt.xlim([1, kwargs['rounds']])
        else:
            plt.xlim([min(kwargs['param_vals']), max(kwargs['param_vals'])])
            
            
        if param in ['rounds','expected_interactions']:
            plt.xlabel('Expected Interactions\n' r'$1/(1-\gamma)$')
            # if param == 'expected_interactions':
                # plt.xticks(range(1,11))
                
        elif param == 'observability':
            plt.xlabel('Probability of observation\n' r'$\omega$')

            plt.xticks([0, .2, .4, .6, .8,  1])
            plt.xticks(np.linspace(0,1,5))

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
        
    plt.yticks([0, 0.5, 1])
    plt.ylabel('Frequency')

    if graph_funcs is not None:
        graph_funcs(ax)
    
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

def some_test():
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


if __name__ == "__main__":
    pass
