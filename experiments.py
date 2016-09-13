from indirect_reciprocity import World, default_params,constant_stop_condition,prior_generator,default_genome,generate_random_genomes,generate_proportional_genomes
from indirect_reciprocity import RationalAgent, ReciprocalAgent, NiceReciprocalAgent, AltruisticAgent, SelfishAgent
from games import RepeatedPrisonersTournament
from itertools import product
import numpy as np
import os.path
from os import makedirs
from utils import pickled, unpickled,softmax_utility
from collections import Counter,defaultdict

#plotting imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def type_tournament(agent_types, game = RepeatedPrisonersTournament(10), proportions = None, path = 'plots/', overwrite = False, plot = False):
    """
    Face each type against every other type
    Plot beliefs over time of all rationals
    """
    RatType = NiceReciprocalAgent

    N_runs = trials = 10
    
    experiment_name = "v".join(filter(str.isupper,t.__name__) for t in agent_types)
    condition = 'trials %s - %s' % (trials, game.name)
    save_dir = 'experiments/%s/%s/' % (experiment_name,condition)
    
    print 'Running '+experiment_name+' Experiment!'
    if os.path.isfile(save_dir+"data.pkl") and not overwrite: 
        print 'Experiment data already exists! Delete or set the overwrite flag to rerun experiment.\nData can be found at:',"./"+save_dir+"data.pkl"
    else:
    
        params = default_params()
        params['agent_types'] = agent_types
        params['games'] = game
        params['p_tremble'] = 0
        params['RA_K'] = 1

        type2index = dict(map(reversed,enumerate(agent_types)))
        rat_index = type2index[RatType]
        print agent_types
        print rat_index
        
        
        historical_record = []; record_history = historical_record.append
    
        rational_types = filter(lambda t: issubclass(t,RationalAgent), agent_types)
        player_types = agent_types+rational_types
        #player_types = [ReciprocalAgent]+agent_types
        print "Playing:", game.name
        for RA_prior in np.linspace(.1,.9, 7):
            print '...with prior', RA_prior
            #for RA_prior in np.linspace(.5,.9, 3):
            
            params['RA_prior'] = RA_prior = {RatType:RA_prior}
            prior = prior_generator(agent_types,RA_prior)
        
            for r_id in range(N_runs):
                print "Trial:",r_id
                np.random.seed(r_id)
                w = World(params, [default_genome(params,agent_type) for agent_type in player_types])
                
                fitness, history = w.run()
                record_history((prior,history))
        print "Done. All conditions have been tested."
        id2type = dict(enumerate(player_types))
        print id2type

        def justcaps(t):
            return filter(str.isupper,t.__name__)
        def history_to_belief_data(historical_record,agent_id,target_ids):
            belief_data,likelihood_data = [],[]
            belief_append = belief_data.append
            likelihood_append = likelihood_data.append
            init_likelihood = 1.0/len(agent_types)
            for prior,history in historical_record:
                for agent_type in agent_types:
                    type_index = type2index[agent_type]
                    for target_id in target_ids:
                        for h in history:
                            belief_append({
                                'round': h['round'],
                                'RA_prior': prior[rat_index],
                                'belief': h['belief'][agent_id][target_id][type_index],
                                'Believed': justcaps(agent_type),
                                'Real':justcaps(id2type[target_id])
                            })
                            likelihood_append({
                                'round': h['round'],
                                'RA_prior': prior[rat_index],
                                'likelihood': h['likelihood'][agent_id][target_id][type_index],
                                'Believed': justcaps(agent_type),
                                'Real':justcaps(id2type[target_id])                                

                            })
                        belief_append({
                            'round': 0,
                            'RA_prior': prior[rat_index],
                            'belief': prior[type_index],
                            'Believed': justcaps(agent_type),
                            'Real':justcaps(id2type[target_id])
                        })
                        likelihood_append({
                            'round': 0,
                            'RA_prior': prior[rat_index],
                            'likelihood': init_likelihood,
                            'Believed': justcaps(agent_type),
                            'Real':justcaps(id2type[target_id])
                        })
            return belief_data,likelihood_data

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data = []
        N_types = len(agent_types)
        N_rational = len(rational_types)
        for agent_type,agent_id in zip(rational_types,range(N_types,N_types+N_rational)):
            data.append(history_to_belief_data(historical_record,agent_id,range(N_types)))
        #data.append(history_to_belief_data(historical_record,0,[3,1,2]))
        #print data

        pickled(data,save_dir+"data.pkl")

        #fitness_record = []
        #belief_record = []
        #f_extend,b_extend = fitness_record.extend,belief_record.extend
        #for prior,history in historical_record:
        #    fit_data = fitness_data(history,params["moran_beta"])
        #    bel_data = belief_data(history,agent_types,RatType)
        #    
        #    for f_entry in fit_data:
        #        f_entry["RA_prior"] = prior[rat_index]
        #        
        #    for b_entry in bel_data:
        #        b_entry["RA_prior"] = prior[rat_index]

         #   f_extend(fit_data)
         #   b_extend(bel_data)
            
        #fitness_data_plot(pd.DataFrame(fitness_record))
        #belief_data_plot(agent_types,pd.DataFrame(belief_record),save_dir+"aggregate_belief.pdf")


    if plot:
        type_tournament_plot(agent_types,save_dir,save_dir)
  #      path += "%s_%s.pdf"
    
    
  #  N_types = len(agent_types)
  #  N_rational = len(rational_types)
  #  for agent_type,agent_id in zip(rational_types,range(N_types,N_types+N_rational)):
  #      belief, likelihood = 
  #      type_initials = filter(str.isupper,agent_type.__name__)
  #      tourney_plotter(belief,'belief', path % (type_initials,'belief'))
  #     tourney_plotter(likelihood,'likelihood',path % (type_initials,'likelihood'))

def type_tournament_plot(agent_types,pkl_dir="plots/",plot_dir="plots/"):
    name = "v".join(filter(str.isupper,t.__name__) for t in agent_types)
    print "Plotting data at","./"+pkl_dir+"data.pkl","to","./"+plot_dir
    data = unpickled(pkl_dir+"data.pkl")
    
    rational_types = filter(lambda t: issubclass(t,RationalAgent), agent_types)
    N_types = len(agent_types)
    N_rational = len(rational_types)
    path = plot_dir+"%s_%s.pdf"
    for (belief,likelihood),agent_type,agent_id in zip(data,rational_types,range(N_types,N_types+N_rational)):
        type_initials = filter(str.isupper,agent_type.__name__)
        tourney_plotter(agent_types,belief,'belief', path % (type_initials,'belief'))
        #tourney_plotter(agent_types,likelihood,'likelihood',path % (type_initials,'likelihood'))
    print "Done!"
        
def tourney_plotter(agent_types,data,field,out_path):
    data = pd.DataFrame(data)
    figure = sns.factorplot('round', field , hue='RA_prior', col='Believed',row='Real',data=data, ci=68,legend_out = True)#, legend = False, facet_kws = {"legend_out":False})
    #figure.add_legend()
    #sns.despine()
    #figure.set_titles('','','')
    #for agent_id,(type_id,agent_type) in product(range(len(agent_types)),enumerate(agent_types)):
    #    axis = figure.facet_axis(agent_id,type_id)
    #axis.set(xlabel='Round #',ylabel = 'Pr( %s is %s|observations)'% (agent_id,agent_type.__name__))

    #for n,agent_type in enumerate(agent_types):
    #    figure.facet_axis(0,n).set(title = agent_type.__name__)
    #    figure.facet_axis(n,0).set(ylabel = "Agent #%s"%n)
    #plt.subplots_adjust(top = 0.9)
    #figure.fig.suptitle("Belief that agent Y is of type X, given observations")
    plt.ylim([0,1])
    
    #plt.ylabel('P(1 is RA | Interactions)'); plt.xlabel('Round #')
    #plt.tight_layout()
    plt.savefig(out_path); plt.close()


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


#RA_v_AA(overwrite=True)
#RA_v_AA_plot()

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

    print "before seeing 2 be a jerk:",RA.belief[2][1] 
    print "P(2=SA)=",RA.belief[2][1]
    print
    observations= [
        (w.game, [0, 2], observers, 'give'),
        (w.game, [2, 0], observers, 'keep'),
    ]

    RA.observe_k(observations*5, K)

    print "After seeing 2 be a jerk:"
    print "P(2=SA)=",RA.belief[2][1]
    print
    
    print "Before seeing 1 be nice to 2:"
    print "P(1=AA)=",RA.belief[1][2]
    print
    observations= [
        (w.game, [1, 2], observers, 'give'),
        (w.game, [2, 1], observers, 'keep'),
    ]

    RA.observe_k(observations*5, K)
    print "After seeing 1 be nice to 2:"
    print "P(1=AA)=",RA.belief[1][2]
    print

#diagnostics()

classic_types = [NiceReciprocalAgent,SelfishAgent,AltruisticAgent]
def fitness_rounds_experiment(pop_size = 10, agent_types = classic_types, condition=None , proportions= None, visibility = "private", overwrite = False,plot=False):
    """
    Repetition supports cooperation. Look at how the number of rounds each dyad plays together and 
    the average fitness of the difference agent types. 
    """
    rounds = 10
    trials = 10
    experiment_name = "fitness"
    #if not condition:
    #    condition = 'rounds_%s_pop_%s_trials_%s' % (rounds,pop_size,trials)
    #else:
    print condition, proportions
    condition = 'cond_%s_rounds_%s_pop_%s_trials_%s' % (condition,rounds,pop_size,trials)
    save_dir = 'experiments/%s/%s/' % (experiment_name,condition)
    print 'Fitness Rounds Experiment'
    if os.path.isfile(save_dir+visibility+"_fitness.pkl") and not overwrite: 
        print 'Game data exists. Delete or set the overwrite flag to re-run experiment.'
        print save_dir+visibility
    else:
        print "Running experiment now..."
        
        game = RepeatedPrisonersTournament(rounds,visibility)
        RatType = NiceReciprocalAgent
        type2index = dict(map(reversed,enumerate(agent_types)))
        rat_index = type2index[RatType]
        print "\t rounds = %s" % rounds
        print "\t trials = %s" % trials
        print "\t population size = %s" % pop_size
        print "\t proportion = %s" % proportions
    
        params = default_params()
        params.update({
            "N_agents":pop_size,
            "RA_K": 1,
            "agent_types": agent_types,
            "agent_types_world": agent_types,# [NiceReciprocalAgent,SelfishAgent],
            "RA_prior":.8,
            "games": game,
        })
        if not proportions:
            proportions = { NiceReciprocalAgent:.3,
                            SelfishAgent:.3,
                            AltruisticAgent:.3 }
        data = []
        historical_record = []
        print "Playing:", game.name
        for RA_prior in np.linspace(.1,.9, 7):#np.linspace(1.0/len(agent_types),.9, 3):
            print '...with prior', RA_prior
            params['RA_prior'] = RA_prior = {RatType:RA_prior}
            prior = prior_generator(agent_types,RA_prior)
        
        
            for r_id in range(trials):
                print "Trial:",r_id
                np.random.seed(r_id)
                w = World(params, generate_proportional_genomes(params,proportions)) #generate_random_genomes(**params))
                fitness, history = w.run()
                historical_record.append((prior,history))

        print "Done playing!"
        print "Processing play data..."

        fitness_record = []
        belief_record = []
        f_extend,b_extend = fitness_record.extend,belief_record.extend
        for prior,history in historical_record:
            fit_data = fitness_data(history,params["moran_beta"])
            bel_data = belief_data(history,agent_types,RatType)
        
            for f_entry in fit_data:
                f_entry["RA_prior"] = 1#prior[RatType]    
            for b_entry in bel_data:
                b_entry["RA_prior"] = prior[rat_index]

            f_extend(fit_data)
            b_extend(bel_data)

        print "Done processing!"

    

        if not os.path.exists(save_dir):
            print "Game directory does not exist. It will be created."
            print save_dir
            os.makedirs(save_dir)

        print "Saving fitness and belief data"
        pd.DataFrame(fitness_record).to_pickle(save_dir+visibility+"_fitness.pkl")
        pd.DataFrame(belief_record).to_pickle(save_dir+visibility+"_belief.pkl")

    if plot:
        print "Plotting is enabled. Will begin plotting now..."
        belief_data_plot(agent_types,save_dir,out_path = save_dir+visibility+"_belief.pdf",
                         pkl_name = visibility+"_belief.pkl")
        fitness_data_plot(save_dir, pkl_name = visibility+"_fitness.pkl",
                          out_path = save_dir+visibility+"_fitness.pdf")
        print "Plotting complete!"

def belief_data(history,agent_types,rational_type):
    """
    only works if 'agent_types' is the same used in all genomes
    """
    agents = history[0]['players']
    type2agents = defaultdict(list)
    type2index = dict(map(reversed,enumerate(agent_types)))
    for agent in agents:
        type2agents[type(agent)].append(agent)
    
    rational_agents = type2agents[rational_type]
    prior = rational_agents[0].genome["prior"]
    data = []
    append = data.append
    for actual,believed in product(agent_types,agent_types):
        append({
            'round': 0,
            'belief': prior[type2index[believed]],
            'Believed Type': believed,
            'Actual Type':actual,
        })
        for event,ra,oa in product(history,rational_agents,type2agents[actual]):
            
            if ra.world_id != oa.world_id:
                append({
                    'round': event['round'],
                    'belief': event['belief'][ra.world_id][oa.world_id][type2index[believed]],
                    'Believed Type': believed,
                    'Actual Type':actual,
                })
    return data

def belief_data_plot(agent_types,belief_dir,out_path=None,pkl_name = "belief.pkl"):
    if not out_path:
        out_path = belief_dir+"belief.pdf"
    
    data = pd.read_pickle(belief_dir+pkl_name)
    figure = sns.factorplot('round', 'belief' , hue='RA_prior', col='Believed Type',row='Actual Type',data=data, ci=68,legend_out = True, legend = False, facet_kws = {"legend_out":False})
    figure.set_titles('','','')
    for n,agent_type in enumerate(agent_types):
        figure.facet_axis(0,n).set(title = agent_type.__name__)
        figure.facet_axis(n,0).set(ylabel = "Actual %s" % agent_type.__name__)
    plt.ylim([0,1])
    plt.tight_layout()
    plt.savefig(out_path); plt.close()
    
def fitness_data(history,beta):
    agents = history[0]['players']
    data = []
    append = data.append
    for event in history:
        genome_fitness = Counter()
        genome_count  = Counter()
        for a_id,a in enumerate(agents):
            genome_fitness[type(a).__name__] += event['payoff'][a_id]
            genome_count[type(a).__name__] += 1
        average_fitness = {a:genome_fitness[a]/genome_count[a] for a in genome_fitness}
        moran_fitness = softmax_utility(average_fitness, beta)
        for a in moran_fitness:
            append({
                'round': event['round'],
                'type': a,
                'fitness': moran_fitness[a]
            })
    return data

def fitness_data_plot(fitness_dir, out_path=None, pkl_name = None):
    if not out_path:
        out_path = fitness_dir+"fitness.pdf"
    if not pkl_name:
        pkl_name = 'fitness.pkl'
    df = pd.read_pickle(fitness_dir+pkl_name)
    sns.factorplot('round','fitness', hue='type', row = 'RA_prior', data=df, ci = 68,legend = False)
    sns.despine()
    plt.ylim([0,1.05])
    plt.title("Fitness")
    #plt.ylabel('Fitness ratio'); plt.xlabel('# of repetitions')
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

def fitness_conditions():
    pop = 4#default is 10
    conditions = [[.333,.333,.333],]
                  #[.7,.2,.1],
                  #[.8,.1,.1],
                  #[.2,.7,.1],
                  #[.1,.8,.1]]

    
    for cond,prop in enumerate(conditions):
        for visibility in ["private"]:#,"random"]:#,"public"]:
            propo = {t:p for t,p in zip(classic_types,prop)}
            #print propo
            
            fitness_rounds_experiment(pop,condition = cond, proportions = propo, visibility = visibility, overwrite = True, plot = False)
            fitness_rounds_experiment(pop,condition = cond, proportions = propo, visibility = visibility, overwrite = False, plot = True)


if __name__ == '__main__':
    type_tournament([NiceReciprocalAgent,SelfishAgent,AltruisticAgent], overwrite = True, plot = True)
    #fitness_conditions()
    #type_tournament([
    #    ReciprocalAgent,
    #    NiceReciprocalAgent,
    #    AltruisticAgent,
    #    SelfishAgent,
    #], plot = True)
    

"""
private vs public
different kinds of observability
game generator should make distributions over 

decider chooses between participants
negative rewards
generative models



"""
"""
four cols
1
sets up coop and morality as evolutionary thinngs for social
tit for tat, indirect reciprocity
the good the bad and the ugly
It works only for a single game
end: bayesian agents
evaluate stuff thrugh models flexible
level-k reasoning

2
talk about how these  models capture common sense inference
forgivenes
protection
ambiguous
observability

3 
evolution
parameters that allow for invadability
dynamic games
generalizations over invadability

4
social learning
remember punishment
conclusions
next steps
social networks
punishment
social learning
ages/experience
overlapping generations

stress: they never play the same game twice.
"""
