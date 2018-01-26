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
from cycler import cycler

import pandas as pd
from datetime import date
from evolve import param_v_rounds_heat, param_v_rounds, compare_param_v_rounds
from evolve import limit_param_plot, ssd_v_param, compare_ssd_v_param, param_v_rounds_plot

TODAY = "./plots/"+date.today().isoformat()+"/"

def Compare_Old(**kwargs):
    trembles = [0, 0.05]
    
    for t in trembles:
        classic = (ag.AllC, ag.AllD, ag.Pavlov, ag.GTFT, ag.TFT)
        background_params = dict(
            rounds = 50,
            benefit = 3,
            tremble = t,
            direct = True,
            game = 'direct',
            pop_size = 100, 
            s = .5,
            RA_prior = 0.5,
            beta = 5,
        )

        background_params.update(kwargs)
        
        # Without WeAgent
        limit_param_plot("rounds",
                         player_types = classic,
                         agent_types = ('self',) + classic,
                         file_name = 'Compare_Old NoWE tremble=%0.2f' % t,
                         experiment = ssd_v_param,
                         stacked = True,
                         **background_params)

        # With WeAgent
        limit_param_plot("rounds",
                         player_types = classic + (WeAgent, ),
                         agent_types = ('self',) + classic,
                         file_name = 'Compare_Old WE tremble=%0.2f' % t,
                         experiment = ssd_v_param,
                         stacked = True,
                         **background_params)

        # With WeAgent but noToM
        limit_param_plot("rounds",
                         player_types = classic + (WeAgent, ),
                         agent_types = ('self', SelfishAgent, AltruisticAgent, RandomAgent),
                         file_name = 'Compare_Old WE RandomToM tremble=%0.2f' % t,
                         experiment = ssd_v_param,
                         stacked = True,
                         **background_params)

        # ToM Comparison
        limit_param_plot("rounds",
                         player_types = (WeAgent, ) + tuple(ReciprocalAgent(RA_K=k) for k in range(2)),
                         agent_types = ('self',) + classic,
                         opponent_types = classic,
                         file_name = 'Compare_Old ToM tremble=%0.2f' % t,
                         experiment = compare_ssd_v_param,
                         graph_kwargs = dict(color = ['C5'], style = ['--', ':', '-']),
                         **background_params)



        
def AllC_AllD_race():
    opponents = (AllD, AllC)
    # opponents = (ag.SelfishAgent,ag.AltruisticAgent)
    ToM = ('self', ) + opponents
    pop = (WeAgent(agent_types = ToM),)+opponents
    # pop = (ReciprocalAgent(agent_types = ToM, RA_K=0),)+opponents
           # ag.TFT)


    trembles = [
        #0,
        .05
    ]

    for t in trembles:
        background_params = dict(
            #experiment = ssd_v_param,
            direct = True,
            game = 'direct',
            RA_prior = 0.5,
            beta = 10,
            player_types = pop,
            #opponent_types = opponents,
            agent_types = ToM,
            tremble = t,

            pop_size = 100, 
            # plot_dir = TODAY,
            #file_name = "gradated"

        )
        
        # limit_param_plot('s', rounds = 100, file_name = 'contest_s_rounds=100_tremble=%0.2f' % t, **background_params)
        # limit_param_plot('s', rounds = 10, file_name = 'contest_s_rounds=10_tremble=%0.2f' % t, **background_params)
        limit_param_plot("rounds", rounds = 50, s=1,
                         file_name = 'contest_rounds_tremble=%0.2f' % t,
                         **background_params)

        # limit_param_plot("RA_prior", rounds = 10, s=1, file_name = 'contest_prior_tremble=%0.2f' % t, **background_params)
        # limit_param_plot("beta", rounds = 10, s=1, file_name = 'contest_beta_tremble=%0.2f' % t, **background_params)


#a_type, proportion = max(zip(player_types,ssd), key = lambda tup: tup[1])
        

def Pavlov_gTFT_race():
    TFT = gTFT(y=1,p=1,q=0)
    MRA = WeAgent#ReciprocalAgent
    r = 10
    
    # Replicate Nowak early 90s
    pop = (TFT, AllC, AllD, gTFT(y=1,p=.99,q=.33), Pavlov)
    for t in [0, 0.05]:
        limit_param_plot('s', pop, rounds = r, tremble = t, file_name = 'nowak_replicate_s_tremble=%.2f' % t, plot_dir = TODAY)
    sim_plotter(5000, (0,0,100,0,0), player_types = pop, rounds = r, tremble = 0.05, mu=0.05, s=1, file_name ='nowak_replicate_sim_tremble=0.05', plot_dir = TODAY)

    # Horse race against gTFT and Pavlov
    prior = 0.5
    beta = 5

    trembles = [0, 0.05]
    opponents = (TFT, gTFT(y=1,p=.99,q=.33), AllC, AllD, Pavlov)

    ToM = ('self',)+opponents
    agent = MRA(agent_types = ToM, beta = beta, RA_prior = prior)
    pop = (agent, AllC, AllD, TFT, gTFT(y=1,p=.99,q=.33), Pavlov)

    comparables = tuple(MRA(RA_prior=p, beta = b, agent_types=ToM) for p,b in product(priors,betas))

    for t in trembles:
        limit_param_plot('s', player_types = pop, rounds = r, tremble = t, file_name = 'horse_s_no_random_tremble=%0.2f' % t, plot_dir = TODAY)
        limit_param_plot('rounds',
                         player_types = comparables,
                         opponent_types = opponents,
                         tremble = t,
                         experiment = compare_ssd_v_param,
                         file_name = 'horse_rounds_no_random_tremble=%0.2f' % t,
                         plot_dir = TODAY)

    # Add Random to the ToM
    ToM = ('self', TFT, gTFT(y=1,p=.99,q=.33), AllC, AllD, Pavlov, RandomAgent)
    for t in trembles:
        limit_param_plot('s', player_types = pop, rounds = r, tremble = t, file_name = 'horse_s_will_random_tremble=%.2f' % t,plot_dir = TODAY)
        limit_param_plot('rounds',
                         player_types = comparables,
                         opponent_types = opponents,
                         tremble = t,
                         experiment = compare_ssd_v_param,
                         file_name = 'horse_rounds_with_random_tremble=%.2f' % t,
                         plot_dir = TODAY)

def bc_rounds_contest():
    for t in [0, 0.05]:
        for opp in [
                (AllD, AllC)
        ]:
            RA = WeAgent(RA_prior = .5, beta = 5, agent_types = ('self',) + opp)
            player_types = (RA, ag.TFT, ag.GTFT, ag.Pavlov
            )
            
            params = dict(param = 'benefit',
                          rounds = 10,
                          player_types = player_types,
                          opponent_types = opp,
                          tremble = t,
                          s = 1,
                          pop_size = 100,
                          game = 'direct',
                          direct = True,
                          experiment = compare_param_v_rounds,
                          file_name = 'bc_rounds_contest_tremble=%.2f_opp=%s' % (t, opp)
            )
            
            param_v_rounds_heat(**params)

def bc_rounds_race():
    file_name = "ToM = %s, beta = %s, prior = %s, tremble = %s"
    plot_dir = "./plots/bc_rounds_race/"

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
            params = dict(param = 'benefit',
                          rounds = 10,
                          player_types = everyone,
                          tremble = t,
                          s = 1,
                          pop_size = 100,
                          game = 'direct',
                          direct = True,
                          experiment = param_v_rounds,
                          file_name = file_name % (ToM,beta,prior,t)
            )
            
            param_v_rounds_heat(**params)

def interval_direct(**kwargs):
    opponents = (ag.SelfishAgent, ag.AltruisticAgent)

    background_params = dict(
        rounds = 10,
        benefit = 15,
        cost = 5,
        tremble = 0,
        direct = True,
        game = 'direct',
        pop_size = 100, 
        s = .5,
        RA_prior = 0.5,
        beta = 5,
        agent_types = ('self', ) + opponents,
        player_types = (WeAgent, ) + opponents, 
        file_name = 'Intervals direct opp=%s' % str(opponents),
    )

    background_params.update(kwargs)

    param_v_rounds_plot("intervals",
                        **background_params)

def AllC_AllD_race_test():
    opponents = (
        SelfishAgent,
        #AllC
    )
    # opponents = (ag.SelfishAgent,ag.AltruisticAgent)
    ToM = ('self', ) + opponents #+(AllC,)
    
    pop = (WeAgent(agent_types = ToM, beta = 5, RA_prior = 0.5),)+opponents
    # pop = (ReciprocalAgent(agent_types = ToM, RA_K=0),)+opponents
           # ag.TFT)


    trembles = [
        #0,
        .05
    ]

    for t in trembles:


        background_params = dict(
            #experiment = ssd_v_param,
            direct = False,
            game = 'dynamic',
            #parallelized = False,
            analysis_type = 'complete',
            #analysis_type = 'limit',
            #RA_prior = 0.5,
            #beta = 10,
            player_types = pop,
            #opponent_types = opponents,
            #agent_types = ToM,
            tremble = t,
            pop_size = 10,
            plot_dir = TODAY,
            s = 1,
            observability = 0,
            extension = '.png',
            trials = 20,
            #mu = .01,
            beta = 5,
            gamma = .9,
            #file_name = "gradated"
        )
        file_name = 'type=%s, game=%s, analytic=%s, pop_size=%s' % (
            background_params['analysis_type'],
            background_params['game'],
            background_params['direct'],
            background_params['pop_size'],
        )
        limit_param_plot("rounds",
                         file_name = file_name,
                         **background_params)

if __name__ == "__main__":
    AllC_AllD_race_test()
    # ToM_indirect()
    # Compare_Old()
    # test()
    # AllC_AllD_race()
    # bc_rounds_contest()
    # Pavlov_gTFT_race()
    #bc_rounds_race()
    #limit_rounds_race()
    assert 0

    MRA = ReciprocalAgent
    SA = SelfishAgent
    AA = AltruisticAgent
    AC = AllC
    AD = AllD
    TFT = gTFT(y=1,p=1,q=0)
    GTFT = gTFT(y=1,p=.99,q=.33)

    
    
