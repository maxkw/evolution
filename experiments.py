from indirect_reciprocity import World, default_params,constant_stop_condition,prior_generator,default_genome
from indirect_reciprocity import RationalAgent, ReciprocalAgent, NiceReciprocalAgent, AltruisticAgent, SelfishAgent
from games import RepeatedPrisonersTournament

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
    agent_types = [ReciprocalAgent,
                   NiceReciprocalAgent,
                   AltruisticAgent,
                   SelfishAgent]
    params['agent_types'] = agent_types
    params['games'] = RepeatedPrisonersTournament(10)
    params['p_tremble'] = 0
    params['RA_K'] = 1

    N_runs = 50
    historical_record = []
    record_history = historical_record.append
    rational_types = filter(lambda t: issubclass(t,RationalAgent), agent_types)
    player_types = agent_types+rational_types
    #player_types = [ReciprocalAgent]+agent_types
    for RA_prior in np.linspace(1.0/len(agent_types),.9, 3):
    #for RA_prior in np.linspace(.5,.9, 3):
        print 'running prior', RA_prior
        params['RA_prior'] = RA_prior = {ReciprocalAgent:RA_prior}
        prior = prior_generator(agent_types,RA_prior)
        
        for r_id in range(N_runs):
            np.random.seed(r_id)
            w = World(params, [default_genome(params,agent_type) for agent_type in player_types])
            fitness, history = w.run()
            record_history((prior,history))
    print "Done. All conditions have been tested."

    def history_to_belief_data(historical_record,agent_id,target_ids):
        data = []
        append = data.append
        for prior,history in historical_record:
            for agent_type in agent_types:
                for target_id in target_ids:
                    for h in history:
                        append({
                            'round': h['round'],
                            'RA_prior': prior[ReciprocalAgent],
                            'belief': h['belief'][agent_id][target_id][agent_type],
                            'Type': agent_type,
                            'ID':target_id
                        })

                    append({
                        'round': 0,
                        'RA_prior': prior[ReciprocalAgent],
                        'belief': prior[agent_type],
                        'Type': agent_type,
                        'ID':target_id
                    })
        return data
    
    N_types = len(agent_types)
    data = history_to_belief_data(historical_record,N_types,range(N_types))
    #data = history_to_belief_data(historical_record,0,[1,2,3])
    df = pd.DataFrame(data)
    df.to_pickle(path)

def RA_v_AA_plot(in_path = 'sims/RAvAA.pkl',
                 out_path='writing/evol_utility/figures/RAvAA.pdf'):
    df = pd.read_pickle(in_path)

    sns.factorplot('round', 'belief', hue='RA_prior', col='Type',row='ID',data=df, ci=68)
    sns.despine()
    plt.ylim([0,1])
    plt.ylabel('P(1 is RA | Interactions)'); plt.xlabel('Round #')
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

    # sns.factorplot('round', 'belief2', hue='RA_prior', data=df, ci=68)
    # sns.despine()
    # plt.ylim([0,1])
    # plt.ylabel('P(1 is RA | Interactions)'); plt.xlabel('Round #')
    # plt.tight_layout()
    # plt.savefig('writing/evol_utility/figures/RAvAA2.pdf'); plt.close()


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


def forgiveness_experiment(path = 'sims/forgiveness.pkl', overwrite = False):
    """
    When two reciprocal agents interact, how likely are they to figure out that they are both reciprocal agent types?
    This will depend on the RA_prior. 

    Compare with something like TFT which if it gets off on the wrong foot will never recover. There are forgiving versions of TFT but they are not context sensitive. Experiments here should explore how the ReciprocalAgent is a more robust cooperator since it can reason about types. 

    TODO: This could be an interesting place to explore p_tremble, and show that agents can recover. 
    """
    print 'Running Forgiveness Experiment'
    if os.path.isfile(path) and not overwrite: 
        print path, 'exists. Delete or set the overwrite flag.'
        return
    
    params = default_params()
    params['N_agents'] = 2
    params['agent_types_world'] = [ReciprocalAgent]
    params['agent_types_model'] = [ReciprocalAgent,SelfishAgent]
    N_round = 10
    params['stop_condition'] = [constant_stop_condition,N_round]
    data = []
    N_runs = 500
    for RA_prior in np.linspace(0.5, 0.95, 4):
        params['RA_prior'] = RA_prior
        print 'running prior', RA_prior

        for r_id in range(N_runs):
            np.random.seed(r_id) # Increment a new seed for each run
            w = World(params, generate_random_genomes(params['N_agents'],
                                                      params['agent_types_world'],
                                                      params['agent_types_model'],
                                                      params['RA_prior'],
                                                      params['prior_precision'],
                                                      params['beta']))
            fitness, history = w.run()
            for nround in range(len(history)):
                avg_beliefs = np.mean([history[nround]['belief'][0][w.agents[1].world_id][ReciprocalAgent],
                                       history[nround]['belief'][1][w.agents[0].world_id][ReciprocalAgent]])
                #print avg_beliefs.dtype
                data.append({
                    'RA_prior': RA_prior,
                    'avg_beliefs': avg_beliefs,
                    'round': nround+1
                })

            data.append({
                'RA_prior': RA_prior,
                'avg_beliefs': RA_prior,
                'round': 0
            })


    df = pd.DataFrame(data)
    df.to_pickle(path)

def forgiveness_plot(in_path = 'sims/forgiveness.pkl', out_path='writing/evol_utility/figures/forgiveness.pdf'):
    df = pd.read_pickle(in_path)

    sns.factorplot(x='round', y='avg_beliefs', hue='RA_prior', data=df)
    sns.despine()
    plt.ylim([0,1.05])
    plt.ylabel('P(Other is reciprocal | Round)'); plt.xlabel('Round')
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

    
    
def protection_experiment(path = 'sims/protection.pkl', overwrite = False):
    """
    If a ReciprocalAgent and a Selfish agent are paired together. How quickly will the
    ReicprocalAgent detect it. Look at how fast this is learned as a function of the prior. 
    """
    
    print 'Running Protection Experiment'
    if os.path.isfile(path) and not overwrite: 
        print path, 'exists. Delete or set the overwrite flag.'
        return

    params = default_params()
    params['agent_types_world'] = agent_types =  [ReciprocalAgent, SelfishAgent]
        
    params['stop_condition'] = [constant_stop_condition,10]
    data = []
    N_runs = 500
    for RA_prior in np.linspace(0.5, .95, 4):
        params['RA_prior'] = RA_prior
        print 'running prior', RA_prior

        for r_id in range(N_runs):
            np.random.seed(r_id)
            w = World(params, [
                {'type': ReciprocalAgent,
                 'RA_prior': RA_prior,
                 'agent_types':[ReciprocalAgent,SelfishAgent],
                 'agent_types_world':[ReciprocalAgent,SelfishAgent],
                 'prior':prior_generator(agent_types,RA_prior),
                 'agent_types_model':[ReciprocalAgent,SelfishAgent],
                 'prior_precision': params['prior_precision'],
                 'beta': params['beta'],
                 'RA_K':1
                },
                {'type': SelfishAgent, 'beta': params['beta'],'RA_K':1},
            ])
            
            fitness, history = w.run()

            for h in history:
                data.append({
                    'round': h['round'],
                    'RA_prior': RA_prior,
                    'belief': h['belief'][0][1][ReciprocalAgent],
                })

            data.append({
                'round': 0,
                'RA_prior': RA_prior,
                'belief': RA_prior,
            })

    #df = pd.DataFrame(data)
    #df.to_pickle(path)

def protection_plot(in_path = 'sims/protection.pkl',
                    out_path='writing/evol_utility/figures/protection.pdf'):
    df = pd.read_pickle(in_path)

    sns.factorplot('round', 'belief', hue='RA_prior', data=df, ci=68)
    sns.despine()
    plt.ylim([0,1])
    plt.ylabel('P(Other is reciprocal | Interactions)'); plt.xlabel('Round #')
    plt.tight_layout()
    plt.savefig(out_path); plt.close()
    
def fitness_rounds_experiment(pop_size = 4, path = 'sims/fitness_rounds.pkl', overwrite = False):
    """
    Repetition supports cooperation. Look at how the number of rounds each dyad plays together and 
    the average fitness of the difference agent types. 
    """
    
    print 'Running Fitness Rounds Experiment'
    if os.path.isfile(path) and not overwrite: 
        print path, 'exists. Delete or set the overwrite flag.'
        return

    agent_types = [ReciprocalAgent,SelfishAgent]
    params = default_params()
    params.update({
        "N_agents":pop_size,
        "RA_K": 1,
        "agent_types": agent_types,
        "agent_types_world": agent_types,
        "RA_prior":.8
    })
    N_runs = 5
    data = []
    n_rounds = 3
    for rounds in np.linspace(1,n_rounds ,n_rounds, dtype=int):
    
        print "Rounds:",rounds
        for r_id in range(N_runs):
            np.random.seed(r_id)

            params['stop_condition'] = [constant_stop_condition,rounds]
                  
            w = World(params, generate_random_genomes(**params))
            fitness, history = w.run()
            #print fitness
            genome_fitness = Counter()
            genome_count = Counter()

            for a_id, a in enumerate(w.agents):
                genome_fitness[type(a)] += fitness[a_id]
                genome_count[type(a)] += 1

            average_fitness = {a:genome_fitness[a]/genome_count[a] for a in genome_fitness}

            moran_fitness = softmax_utility(average_fitness, params['moran_beta'])

            for a in moran_fitness:
                data.append({
                    'rounds': rounds,
                    'genome': a,
                    'fitness': moran_fitness[a]
                })

        df = pd.DataFrame(data)
        df.to_pickle(path)
    return w

def fitness_rounds_plot(in_path = 'sims/fitness_rounds.pkl', out_path='writing/evol_utility/figures/fitness_rounds.pdf'):

    df = pd.read_pickle(in_path)
    sns.factorplot('rounds', 'fitness', hue='genome', data=df,)
    sns.despine()
    plt.ylim([0,1.05])
    plt.ylabel('Fitness ratio'); plt.xlabel('# of repetitions')
    plt.tight_layout()
    plt.savefig(out_path); plt.close()
