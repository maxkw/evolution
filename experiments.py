from indirect_reciprocity import World, ReciprocalAgent,SelfishAgent,AltruisticAgent,default_params,constant_stop_condition,prior_generator
from games import PrisonersDilemma

import numpy as np
import os.path

#plotting imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def RA_v_AA(path = 'sims/RAvAA.pkl', overwrite = False):
    """
    How long does it take for RA to realize AA's type
    """
    
    print 'Running RA v. AA Experiment'
    if os.path.isfile(path) and not overwrite: 
        print path, 'exists. Delete or set the overwrite flag.'
        return

    params = default_params()
    # agent_types = [ReciprocalAgent, AltruisticAgent]
    agent_types = [ReciprocalAgent, SelfishAgent, AltruisticAgent]
    params['agent_types_world'] = agent_types
    params['games'] = PrisonersDilemma()
    params['stop_condition'] = [constant_stop_condition,10]
    params['p_tremble'] = 0
    # params['beta'] = 5
    data = []
    N_runs = 50
    for RA_prior in np.linspace(0.5, .9, 3):
        params['RA_prior'] = RA_prior
        print 'running prior', RA_prior

        for r_id in range(N_runs):
            np.random.seed(r_id)
            prior = prior_generator(agent_types,RA_prior)
            w = World(params, [
                {'type': ReciprocalAgent,
                 'RA_prior': RA_prior,
                 'agent_types':agent_types,
                 'agent_types_world':agent_types,
                 'prior': prior,
                 'prior_precision': params['prior_precision'],
                 'beta': params['beta'],
                 'RA_K':1
                },
                {'type': ReciprocalAgent,
                 'RA_prior': RA_prior,
                 'agent_types':agent_types,
                 'agent_types_world':agent_types,
                 'prior': prior,
                 'prior_precision': params['prior_precision'],
                 'beta': params['beta'],
                 'RA_K':1
                },
                {'type': AltruisticAgent, 'beta': params['beta'], 'RA_K':1},
                {'type': SelfishAgent, 'beta': params['beta'], 'RA_K':1},
            ])
            
            fitness, history = w.run()

            for h in history:
                
                data.append({
                    'round': h['round'],
                    'RA_prior': prior[ReciprocalAgent],
                    'belief': h['belief'][0][1][ReciprocalAgent],
                    'belief2': h['belief'][1][0][ReciprocalAgent],
                })

            data.append({
                'round': 0,
                'RA_prior': prior[ReciprocalAgent],
                'belief': prior[ReciprocalAgent],
                'belief2': prior[ReciprocalAgent],
            })
            
    df = pd.DataFrame(data)
    df.to_pickle(path)

def RA_v_AA_plot(in_path = 'sims/RAvAA.pkl',
                 out_path='writing/evol_utility/figures/RAvAA.pdf'):
    df = pd.read_pickle(in_path)

    sns.factorplot('round', 'belief', hue='RA_prior', data=df, ci=68)
    sns.despine()
    plt.ylim([0,1])
    plt.ylabel('P(1 is RA | Interactions)'); plt.xlabel('Round #')
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

    sns.factorplot('round', 'belief2', hue='RA_prior', data=df, ci=68)
    sns.despine()
    plt.ylim([0,1])
    plt.ylabel('P(1 is RA | Interactions)'); plt.xlabel('Round #')
    plt.tight_layout()
    plt.savefig('writing/evol_utility/figures/RAvAA2.pdf'); plt.close()


RA_v_AA(overwrite=True)
RA_v_AA_plot()

from indirect_reciprocity import default_genome
def diagnostics():

    params = default_params()

    
    typesClassic = agent_types = (ReciprocalAgent,SelfishAgent,AltruisticAgent)
    
    params['stop_condition'] = [constant_stop_condition,10]
    params['p_tremble'] = 0
    params['RA_prior'] = .8 #0.33
    params['RA_prior_precision'] = 0
    prior = prior_generator(agent_types,params['RA_prior'])
    observers = range(3)
    w = World(params, [default_genome(ReciprocalAgent,agent_types,params),
                       default_genome(SelfishAgent,agent_types,params),
                       default_genome(ReciprocalAgent,agent_types,params)])
    RA = w.agents[0]
    
    K = 0

    print "before seeing 2 be a jerk:",RA.belief[2][SelfishAgent] 
    print "P(2=SA)=",RA.belief[2][SelfishAgent]
    print
    observations= [
        (w.game, [0, 2], observers, 'give'),
        (w.game, [2, 0], observers, 'keep'),
    ]

    RA.observe_k(observations*5, K)

    print "After seeing 2 be a jerk:"
    print "P(2=SA)=",RA.belief[2][SelfishAgent]
    print
    
    print "Before seeing 1 be nice to 2:"
    print "P(1=AA)=",RA.belief[1][AltruisticAgent]
    print
    observations= [
        (w.game, [1, 2], observers, 'give'),
        (w.game, [2, 1], observers, 'keep'),
    ]

    RA.observe_k(observations*5, K)
    print "After seeing 1 be nice to 2:"
    print "P(1=AA)=",RA.belief[1][AltruisticAgent]
    print

#diagnostics()
