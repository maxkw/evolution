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
from steady_state import mm_to_limit_mcp, mcp_to_ssd, steady_state, mcp_to_invasion, limit_analysis
import pandas as pd
from datetime import date
from agents import leading_8_dict, shorthand_to_standing

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


@plotter(agent_simulation,plot_exclusive_args = ['data'])
def sim_plotter(generations, pop, player_types, data =[]):
    for hue in data['type'].unique():
        plt.plot(data[data['type']==hue]['generation'], data[data['type']==hue]['population'], label=hue)

    plt.legend()

@experiment(unpack = 'record', memoize = False, verbose = 3)
def ssd_v_param(param, player_types, direct = False, **kwargs):
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
    unique_interactions = kwargs['pop_size'] * (kwargs['pop_size'] - 1)
    if unique_interactions <= kwargs['rounds']:
        raise Exception("There are more rounds than unique interactions. Raise pop_size or lower rounds.")

    Xs = {
        # 'RA_prior': np.linspace(0,1,21)[1:-1],
        'RA_prior': np.linspace(0, 1, 21),
        'benefit': [1.5, 2, 2.5, 3],
        'beta': np.linspace(1, 11, 6),
        'pop_size': np.unique(np.geomspace(2, 2**10, 100, dtype=int)),
        's': logspace(start = .001, stop = 1, samples = 100)
    }
    record = []
    
    if param == "rounds":
        expected_pop_per_round = limit_analysis(player_types = player_types, direct = direct, **kwargs)
        for r, pop in enumerate(expected_pop_per_round, start = 1):
            for t, p in zip(player_types, pop):
                record.append({
                    'rounds': r,
                    'type': t.short_name('agent_types'),
                    'proportion': p
                })

        return record

    elif param in Xs:
        for x in Xs[param]:
            ssd = limit_analysis(player_types = player_types, direct = direct, **dict(kwargs,**{param:x}))[-1]
            for t, p in zip(player_types, ssd):
                record.append({
                    param: x,
                    "type": t.short_name("agent_types"),
                    "proportion": p
                })
        return record

    else:
        raise Exception('Param %s is not implemented. Add the range to Xs' % param)

def compare_ssd_v_param(param, player_types, opponent_types, **kwargs):
    dfs = []
    for player_type in player_types:
        df = ssd_v_param(param = param, player_types = (player_type,)+opponent_types, **kwargs)
        dfs.append(df[df['type']==player_type.short_name("agent_types")])
    return pd.concat(dfs, ignore_index = True)

@plotter(ssd_v_param, plot_exclusive_args = ['experiment','data'])
def limit_param_plot(param, player_types, data = [], **kwargs):
    fig = plt.figure()
    for hue in data['type'].unique():
        d = data[data['type']==hue]
        p = plt.plot(d[param], d['proportion'], label=hue)

    if param in ['pop_size']:
        plt.axes().set_xscale('log',basex=2)
    elif param == 's':
        plt.axes().set_xscale('log')

    # if param in ["beta"]:
        # plt.axes().set_xscale('log',basex=10)
    # if param in ['rounds']:
        # pass

    plt.xlabel(param)
    plt.ylim([0, 1.05])
    plt.yticks([0,0.5,1])
    sns.despine()
    plt.legend()
    plt.tight_layout()

####
# ''' BC / ROUND HEAT MAP CODE '''
####
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


####
# Premade experiments
####

def AllC_AllD_race():
    today = "./plots/"+date.today().isoformat()+"/"
    
    ToM = ('self', AllC, AllD)
    opponents = (AllC, AllD)
    pop = (WeAgent(agent_types = ToM), ag.TFT)
    
    for t in [0, 0.05]:
        background_params = dict(
            experiment = compare_ssd_v_param,
            direct = True,
            RA_prior = 0.5,
            beta = 5,
            player_types = pop,
            opponent_types = opponents,
            agent_types = ToM,
            tremble = t,
            pop_size = 100, 
            plot_dir = today
        )
        
        # limit_param_plot('s', rounds = 100, file_name = 'contest_s_rounds=100_tremble=%0.2f' % t, **background_params)
        # limit_param_plot('s', rounds = 10, file_name = 'contest_s_rounds=10_tremble=%0.2f' % t, **background_params)
        # limit_param_plot("rounds", rounds = 100, s=1, file_name = 'contest_rounds_tremble=%0.2f' % t, **background_params)
        limit_param_plot("RA_prior", rounds = 10, s=1, file_name = 'contest_prior_tremble=%0.2f' % t, **background_params)
        # limit_param_plot("beta", rounds = 10, s=1, file_name = 'contest_beta_tremble=%0.2f' % t, **background_params)


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
                         experiment = compare_ssd_v_param,
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
                         experiment = compare_ssd_v_param,
                         file_name = 'horse_rounds_with_random_tremble=%.2f' % t,
                         plot_dir = today)

def bc_rounds_contest():
    WA = WeAgent
    prior = 0.5

    RA = WA(RA_prior = prior, agent_types = ('self', ag.AllC, ag.AllD))
    player_types = (RA, ag.TFT, ag.GTFT, ag.Pavlov)

    for t in [0, 0.05]:
        bc_rounds_plot(
            max_rounds = 20,
            experiment = compare_bc_v_rounds,
            player_types = player_types,
            opponent_types = (ag.AllC, ag.AllD),
            tremble = t
        )



def bc_rounds_race():
    file_name = "ToM = %s, beta = %s, prior = %s, tremble = %s"
    plot_dir = "./plots/bc_rounds_race/"

    max_rounds = 20

    priors = [
        #.1,
        .5,
        # .75
    ]

    ToMs = [
        ('self', ag.AllC, ag.AllD, ag.TFT, ag.GTFT, ag.Pavlov)
    ]

    betas = [
        #1,
        #3,
        5,
        # 10,
    ]

    trembles = [
        0,
        0.05
    ]

    for prior, ToM, beta in product(priors,ToMs,betas):
        RA = WeAgent(RA_prior = prior, agent_types = ToM, beta = beta)
        everyone = (RA, ag.AllC, ag.AllD, ag.TFT, ag.GTFT, ag.Pavlov)
        for t in trembles:
            bc_rounds_plot(everyone, max_rounds = max_rounds, tremble = t,
                           plot_dir = plot_dir,
                           file_name = file_name % (ToM,beta,prior,t)
            )

def limit_rounds_race():
    file_name = "ToM = %s, beta = %s, prior = %s, tremble = %s"
    plot_dir = "./plots/limit_rounds_race/"

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
    # image_contest()
    AllC_AllD_race()
    # Pavlov_gTFT_race()
    # bc_rounds_race()
    # limit_rounds_race()
    assert 0

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

