from indirect_reciprocity import World, default_params,constant_stop_condition,prior_generator,default_genome,generate_random_genomes,generate_proportional_genomes
from indirect_reciprocity import RationalAgent, ReciprocalAgent, NiceReciprocalAgent, AltruisticAgent, SelfishAgent,TitForTat
from games import RepeatedPrisonersTournament,BinaryDictator
from itertools import product,permutations
import numpy as np
import os.path
from os import makedirs
from utils import pickled, unpickled,softmax_utility
from collections import Counter,defaultdict

#plotting imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from copy import copy,deepcopy
       
class Experiment(object):
    def __init__(self,name,conditions,processing_list):
        """
        name is a string
        conditions is too
        processing list is a triple with a 
            filename(no extension)
            a function that takes a historical record and returns a pd.DataFrame
            a function that takes a pd.DataFrame and plots it
        """
        self.name = name
        self.conditions = conditions
        self.processing_list = processing_list
        
    def __call__(*args,**kwargs):
        if not kwargs:
            kwargs = self.defaults
            kwargs.update({'overwrite':False,
                           'process':True,
                           'plot':False})
        self.run(**kwargs)
    def run(self, overwrite = False, process_data = False, plot = True):
        name,condition,save_dir = self.name, self.conditions,self.save_dir
        print 'Starting '+name+' experiment!'

        #Run the actual experiment and pickle the historical data...        
        if not overwrite and os.path.isfile(save_dir+"history.pkl"):
            print 'Experiment data already exists! Delete or set the overwrite flag to rerun experiment.\nData can be found at:',"./"+save_dir+"history.pkl"

            if process_data:
                print "Loading cached history..." 
                history = unpickled(save_dir+"history.pkl")
                print "\t...done!"
        else:

            if not os.path.exists(save_dir):
                print "Game directory does not exist. It will be created."
                print save_dir
                os.makedirs(save_dir)
                
            print "Running experiment..."
            history = pickled(self.procedure(),save_dir+"history.pkl")
            print "Experiment succesfully ran! History saved to: %s" % save_dir+"history.pkl"
  
        #Slice the historical data...
        if overwrite or process_data:
            self.process_data(history)
        elif plot:
            print "Print flag is set. Checking for plotting data."
            all_data_is_there = all([os.path.isfile(save_dir+file_name+".pkl")
                                     for file_name,x,y in self.processing_list])
            if all_data_is_there:
                print "Using cached data for plotting."
            else:
                print "Insufficient plotting data."
                self.process_data(history)
            

        #Plot the data...
        if plot:
            self.plot_data()

            
    def process_data(self,history):
        save_dir = self.save_dir
        print "Processing history for plotting..."
        for file_name,process_function,plot_function in self.processing_list:
            #file_name = self.conditions+" - "+file_name
            pickled(process_function(history),save_dir+file_name+".pkl")
            print "\t...%s data saved to:\n\t\t./%s" % (file_name,save_dir+file_name+".pkl")
        print ""
        
    def plot_data(self):
        save_dir = self.save_dir
        print "Commencing plotting..."
        for file_name,process_function,plot_function in self.processing_list:
            outfile = save_dir+self.name+self.conditions+" - "+file_name+".pdf"
            plot_function(unpickled(save_dir+file_name+".pkl"), outfile)
            print "\t...%s plot saved to:\n\t\t./%s" % (file_name,outfile)
        print ""


def history_to_belief_data(agent_types,agent_id,target_ids,historical_record):
    """
    agent_types is a list of agent types over which rational agents have beliefs
    agent_id is the ID of the agent whose beliefs are being examined
    target_ids is a list of IDs of the agents to whom the beliefs to be desplayed correspond
    historical_record is a list of tuples of the form (RA_prior,history)
    """
    def justcaps(t):
        return filter(str.isupper,t.__name__)
    
    id2type = dict(enumerate(agent_types))
    
    rational_types = filter(lambda t: issubclass(t,RationalAgent), agent_types)
    rat_index = agent_types.index(rational_types[0])
    belief_data = []; belief_append = belief_data.append

    agent_types_ = [a for a in agent_types if a is not TitForTat]
    type2index = dict(map(reversed,enumerate(agent_types_)))
    for prior,history in historical_record:
        for agent_type in agent_types_:
            type_index = type2index[agent_type]
            for target_id in target_ids:
                belief_append({
                    'round': 0,
                    'prior':prior[rat_index],
                    'belief':prior[type_index],
                    'believed':justcaps(agent_type),
                    'id':target_id,
                    'actual':justcaps(id2type[target_id]),
                })
                for event in history:
                    belief_append({
                        'round': event['round'],
                        'prior':prior[rat_index],
                        'belief':event['belief'][agent_id][target_id][type_index],
                        'believed':justcaps(agent_type),
                        'id':target_id,
                        'actual':justcaps(id2type[target_id]),
                    })
    return belief_data

def belief_plotter(data,out_path):
    data = pd.DataFrame(data)
    figure = sns.factorplot('round', 'belief', hue='prior', col='believed',row='actual',data=data, ci=68,legend_out = True)#, legend = False, facet_kws = {"legend_out":False})
    plt.ylim([0,1])
    plt.tight_layout()
    plt.savefig(out_path); plt.close()
                    

class type_tourney(Experiment):
    def __init__(self,agent_types):
        self.name = name = "v".join(filter(str.isupper,t.__name__) for t in agent_types)
        self.trials = trials = 25
        self.conditions = "trials - %s" % trials
        self.agent_types = agent_types
        self.save_dir = 'experiments/%s/%s/' % (self.name,self.conditions)
        #self.rational_type = rational_type = NiceReciprocalAgent
        #self.agent_types = agent_types = [rational_type,SelfishAgent]
        n = len(agent_types)
        self.processing_list = [('belief',partial(history_to_belief_data,agent_types,n,range(n)),
                                 belief_plotter)]
    def procedure(self):
        trials = self.trials
        agent_types = self.agent_types
        
        rational_types = filter(lambda t: issubclass(t,RationalAgent), agent_types)
        rational_type = rational_types[0]
        player_types = agent_types+rational_types

        record = [];  append = record.append
        for RA_prior in np.linspace(.1,.9,7):
            agent_types_ = [a for a in agent_types if a is not TitForTat]
            params = default_params(agent_types_,RA_prior)
            params['RA_prior'] = RA_prior = {rational_type:RA_prior}
            prior = prior_generator(agent_types_,RA_prior)
            genomes = [default_genome(params,agent_type) for agent_type in player_types]

            for trial in range(trials):
                np.random.seed(trial)
                w = World(params,genomes)
                fitness, history = w.run()
                append((prior,history))
        return record
                


#ATFT = [NiceReciprocalAgent,SelfishAgent,ATFT]
#type_tourney(

class first_impressions(Experiment):
    def __init__(self,trials = 10, K = 1, kind = 'seq'):
        assert kind in ['seq','perm']
        self.K = K
        self.trials = trials
        self.kind = kind
        self.name = "first_impressions"
        self.conditions = "(trials=%s,K=%s,kind='%s')" % (trials,K,kind)
        self.save_dir = 'experiments/%s/%s/' % (self.name,self.conditions)
        self.rational_type = rational_type = ReciprocalAgent
        self.agent_types = agent_types = [rational_type,SelfishAgent]
        self.processing_list = [('belief',partial(history_to_primed_data,agent_types,0,[1]),
                                 primed_plotter)]
    def procedure(self):
        trials = self.trials
        agent_types = self.agent_types
        rational_type = self.rational_type
        BD = BinaryDictator()

        def char_to_observation(action_char):
            action = "give" if action_char is "C" else "keep"
            return [(BD,[1,0],[0,1],action)]
    
        def observations(actions_string):
            return map(char_to_observation,actions_string)

        total_actions = 3
        def int_to_actions_string(number):
            """
            given a number return a binary representation
            where 0s are Ds and 1s are Cs
            """
            return ("{0:0"+str(total_actions)+"b}").format(number).replace('1','C').replace('0','D')

        action_sequences = []
        if self.kind == 'seq':
            action_strings = ["C"*n+"D" for n in range(5)]
        elif self.kind == 'perm':
            action_strings = [int_to_actions_string(n) for n in range(8)]

        game = RepeatedPrisonersTournament(10)
        record = []; append = record.append
        
        for RA_prior in [.8]:#np.linspace(.1,.9,5):
            print "RA_prior = "+ str(RA_prior)
            params = default_params(agent_types,RA_prior)
            params['games'] = game
            params['RA_K'] = self.K
            genomes = [default_genome(params,rational_type)]*2
            for action_string in action_strings:
                print "Actions= "+action_string
                for trial in range(trials):
                    np.random.seed(trial)
            
                    #create the world
                    world = World(params,genomes)
                    
                    #have each of the agents observe each of the actions individually
                    for observation in observations(action_string):
                        #print observation
                        #new_observation = BD.play(world.agents,world.agents,tremble = 0)[1]
                        for agent in world.agents:
                            agent.observe(observation)
                            #agent.observe(observation+new_observation)

                    primed_belief = [copy(agent.belief) for agent in world.agents]

                    #run the world with the primed agents and save the history
                    fitness,history = world.run()
                    history += [{'prior':RA_prior,'round':0, 'belief': primed_belief}]
                    append(((RA_prior,action_string),history))
            
        return record

def history_to_primed_data(agent_types,agent_id,target_ids,record):
    data =[];append = data.append
    def justcaps(t):
        return filter(str.isupper,t.__name__)
    
    prior = record[0][1][0]['players'][0].genome['prior']
    type_to_index = dict(map(reversed,enumerate(agent_types)))
    type_to_shorthand = {t:justcaps(t) for t in agent_types}
    
    for ((prior,actions),history), agent_type in product(record,agent_types):
        shorthand = type_to_shorthand[agent_type]
        type_index = type_to_index[agent_type]
        for target_id in target_ids:
            for event in history:
                append({
                    #'prior': prior,
                    'round': event['round'],
                    'actions': actions,
                    'belief': event['belief'][agent_id][target_id][type_index],
                    'type': shorthand,
                    'ID':target_id
                })
    return data
        
def primed_plotter(data,out_path):
    data = pd.DataFrame(data)
    figure = sns.factorplot('round','belief', hue = None, col='type',row='actions',data=data, ci=68,legend_out = True,aspect = 1.5,size = 5, kind = 'strip')#kind = 'violin',scale ='count', width=.9,cut = .5,inner = 'box')
    #figure = sns.factorplot('round','belief', hue = None, col='type',row='actions',data=data, ci=68,legend_out = True,aspect = 1.5,size = 5)#kind = 'violin',scale ='count', width=.9,cut = .5,inner = 'box')
    #size = figure.get_size_inches()
    #figure.set_size_inches((size[0]*1,size[1]*1.5))
    figure.set(yticks=np.linspace(0,1,8))
    y_buff = .15
    plt.ylim([0-y_buff,1+y_buff])
    plt.savefig(out_path); plt.close()


class first_impressions_2(Experiment):
    def __init__(self,trials = 10, kind = 'seq',passive = False, sequential = False):
        assert kind in ['seq','perm']
        #self.K = K
        self.passive = passive
        self.trials = trials
        self.kind = kind
        self.name = "first_impressions_2"
        self.conditions = "(trials=%s, kind='%s', passive=%s, sequential=%s)" % (trials,kind,passive,sequential)
        self.save_dir = 'experiments/%s/%s/' % (self.name,self.conditions)
        self.rational_type = rational_type = ReciprocalAgent
        self.agent_types = agent_types = [rational_type,SelfishAgent]
        self.processing_list = [('belief',partial(first_impressions_data,agent_types,[0,1]),
                                 first_impressions_plotter)]
        self.sequential = sequential
    def procedure(self):
        trials = self.trials
        agent_types = self.agent_types
        rational_type = self.rational_type
        sequential = self.sequential
        
        BD = BinaryDictator()
        plot_prehistory = True

        def char_to_observation(action_char):
            action = "give" if action_char is "C" else "keep"
            return [(BD,[0,1],[0,1],action)]
    
        def observations(actions_string):
            return map(char_to_observation,actions_string)

        total_actions = 3
        def int_to_actions_string(number):
            """
            given a number return a binary representation
            where 0s are Ds and 1s are Cs
            """
            return ("{0:0"+str(total_actions)+"b}").format(number).replace('1','C').replace('0','D')

        action_sequences = []
        if self.kind == 'seq':
            action_strings = ["C"*n+"D" for n in range(5)]
        elif self.kind == 'perm':
            action_strings = [int_to_actions_string(n) for n in range(8)]

        if sequential:
            game = RepeatedSequentialBinary(10)
        else:
            game = RepeatedPrisonersTournament(10)
            
        record = []; append = record.append

        
        for K in [0,1,2]:
            for RA_prior in [.75]:#np.linspace(.1,.9,5):
                print "RA_prior = "+ str(RA_prior)
                params = default_params(agent_types,RA_prior)
                params['games'] = game
                params['RA_K'] = K
                genomes = [default_genome(params,rational_type)]*2
                for action_string in ["DDD"]:
                                      #,"DCD","CDD","DDC"]:#action_strings:
                    print "Actions= "+action_string
                    for trial in range(trials):
                        np.random.seed(trial)
            
                        #create the world
                        world = World(params,genomes)
                        #C = rational_type(default_genome(params,rational_type))
                        
                        #have each of the agents observe each of the actions individually
                        prehistory = []
                        action_len = len(action_string)
                        prehistory.append({'round':-action_len,
                                           'belief': [copy(a.belief) for a in world.agents],
                                           'prior': RA_prior,
                                           'players': [deepcopy(agent) for agent in world.agents],
                        })
                        for i, observation in enumerate(observations(action_string),-(action_len-1)):
                            #print observation
                            if not self.passive:
                                new_observation = BD.play(world.agents[np.array([1,0])],world.agents,tremble = 0)[1]
                                if not sequential:
                                    for agent in world.agents:
                                        agent.observe(observation+new_observation)
                                else:
                                    for obs in [observation,new_observation]:
                                        for agent in world.agents:
                                            agent.observe(obs)
                            else:
                                for agent in world.agents:
                                    agent.observe(observation)
                            prehistory.append({'round':i,
                                               'belief': [copy(a.belief) for a in world.agents],
                                               'prior': RA_prior,
                                               'players': [deepcopy(agent) for agent in world.agents],
                                               })

                        #run the world with the primed agents and save the history
                        fitness,history = world.run()
                        if plot_prehistory:
                            history = prehistory+history
                        else:
                            history = prehistory[-1:]+history
                        append(((K,action_string,trial),history))
        return record
                        
def first_impressions_data(agent_types,agent_ids,record):
    data =[];append = data.append
    def justcaps(t):
        return filter(str.isupper,t.__name__)
    #print record
    prior = record[0][1][-1]['players'][0].genome['prior']
    type_to_index = dict(map(reversed,enumerate(agent_types)))
    type_to_shorthand = {t:justcaps(t) for t in agent_types}

    def belief_getter(agent,a_id,k):
        n_id = (a_id+1)%2
        print k,n_id
        if k == 0:
            return agent.belief[n_id][type_to_index[ReciprocalAgent]]
        else:
            try:
                return belief_getter(agent.rational_models[n_id],n_id,k-1)
            except KeyError:
                print "there was a keyerror"
                agent._use_defaultdicts()
                return belief_getter(agent.models[n_id][ReciprocalAgent],n_id,k-1)
    for ((K,actions,trial),history), agent_type in product(record,[ReciprocalAgent]):
        shorthand = type_to_shorthand[agent_type]
        type_index = type_to_index[agent_type]
        for agent_id,target_id in permutations(agent_ids,2):
            for event in history:
                agent = event['players'][agent_id]._use_defaultdicts()
                for k in range(K+1):
                    append({
                        'K': K,
                        'k':k,
                        'round': event['round'],
                        'actions': actions,
                        'belief': belief_getter(agent,agent_id,k),
                        'believer':agent_id,
                        'type': shorthand,
                })
    return data



def first_impressions_plotter(raw_data,out_path):
    for K_level in range(3):
        data = pd.DataFrame(raw_data)
        data = data[(data['actions'] == 'DDD') & (data['K']== K_level)]
        
        figure = sns.factorplot('round','belief', hue = 'actions',
                                col='believer',row='k',data=data, ci=68,legend = False,aspect = 1, size = 4.5, #kind = 'point')
                                kind = 'violin',scale ='area', width=.9,cut = 0,inner = 'point',bw = .2)
        #figure = sns.factorplot('round','belief', hue = None, col='type',row='actions',data=data, ci=68,legend_out = True,aspect = 1.5,size = 5)#kind = 'violin',scale ='count', width=.9,cut = .5,inner = 'box')
        #size = figure.get_size_inches()
        #figure.set_size_inches((size[0]*1,size[1]*1.5))
        figure.set(yticks=[x/10.0 for x in range(11)])
        y_buff = .09
        plt.ylim([0-y_buff,1+y_buff])
        
        figure.set_titles('','','')
        id_to_letter = dict(enumerate("AB"))
        """
        range(3) bc k in [0,1,2]
        a_id is the agent, t_id is who the agent is thinking about
        """
        for k,(a_id,t_id) in product(range(K_level+1),permutations(range(2))):
            axis = figure.facet_axis(k,a_id)
            sns.pointplot(x = "round", y="belief", color = "red",data = data[(data['believer']==a_id) & (data['k']==k)],ax=axis)
            axis.set(#xlabel='# of interactions',
                ylabel = '$\mathrm{Pr_{%s}( T_{%s} = RA | O_{1:n} )}$'% (k,id_to_letter[t_id]),
                title = "%s's K=%s belief that %s is RA" % (id_to_letter[a_id],k,id_to_letter[t_id]))
        
        plt.subplots_adjust(top = 0.93)
        figure.fig.suptitle("A and B's beliefs that the other is RA when A's first 3 moves are D")
    
        k_string = " K=%s" % K_level
        plt.savefig(out_path[:-4]+k_string+out_path[-4:]); plt.close()


def fi2():
    for passiveness in [True]:#[True,False]:
        first_impressions_2(100,'perm',passive=passiveness).run(False,False,True)















    
    
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
        
        

        
        def history_to_belief_data(historical_record,agent_id,target_ids):
            def justcaps(t):
                return filter(str.isupper,t.__name__)
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
            return belief_data#,likelihood_data

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
    
    
    #plt.ylabel('P(1 is RA | Interactions)'); plt.xlabel('Round #')
    #plt.tight_layout()
    plt.savefig(out_path); plt.close()

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

def fitness_conditions_():
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

def first_impressions_():
    BD = BinaryDictator()
    NRA = ReciprocalAgent
    agent_types = [ReciprocalAgent,SelfishAgent]#,AltruisticAgent]
       
    def observation(key):
        action = "give" if key is "C" else "keep"
        return [(BD,[0,1],['obs',0,1],action)]
    
    def observations(actions):
        return map(observation,actions)
    
    type2index = dict(map(reversed,enumerate(agent_types)))
    data = [];append = data.append
    for RA_prior in np.linspace(.1,.9,7):
        genome = default_genome(default_params(agent_types,RA_prior),NRA)
        for agent_type in agent_types:
            append({"belief":genome['prior'][type2index[agent_type]],
                    "prior":RA_prior,
                    "observation":0,
                    "agent":0,
                    "type":agent_type})
        for n in range(10):
            observer = NRA(genome,"obs")
            for o in observations("C"*n+"D"):
                observer.observe_k(o,1)
            for agent_type in agent_types:
                append({"belief":observer.belief_that(0,agent_type),
                        "prior":RA_prior,
                        "observation":n+1,
                        "agent":0,
                        "type":agent_type})
    df = pd.DataFrame(data)

    figure = sns.factorplot('observation', 'belief' , hue='prior', col='type',row='agent',data=df, ci=68,legend_out = True, legend = False, facet_kws = {"legend_out":True, "figsize":(3,5)})
    #figure.set_titles('','','')
    for n,agent_type in enumerate(agent_types):
        figure.facet_axis(0,n).set(title = agent_type.__name__)
    #    figure.facet_axis(n,0).set(ylabel = "Actual %s" % agent_type.__name__)
    plt.ylim([0,1])
    plt.tight_layout()
    plt.savefig("./experiments/first_impressions.pdf"); plt.close()

    
if __name__ == '__main__':
    if True:
        #for K,kind in product(range(1,3),['seq','perm']):
        #    first_impressions(100,K,kind).run(True,False,False)
        
        #for K,kind in product(range(1,3),['seq','perm']):
        #    first_impressions(100,K,kind).run(False,False,True)
        #first_impressions(100,1,'seq').run(False,False,True)
        fi2()
    #tourney_types = [ReciprocalAgent,SelfishAgent,TitForTat,AltruisticAgent]
    #type_tourney(tourney_types).run(True,False,True)
    #first_impressions()
    #type_tournament([NiceReciprocalAgent,SelfishAgent,AltruisticAgent], overwrite = True, plot = True)
    #fitness_conditions()
    #type_tournament([
    #    ReciprocalAgent,
    #    NiceReciprocalAgent,
    #    AltruisticAgent,
    #    SelfishAgent,
    #], plot = True)
    
"""
observations that occur between belief updates are IID, the same system of probabilities
generates all of them. each such set of observations may or may not be dependent on previous
sets. the order in which sets are generated matters? if not then all observations are IID.

Payoff :: Actions x PlayingAgents
a mapping from actions to the payoffs of each agent
Payoff[Action][PlayingAgent] is the payoff

Order  :: PlayingAgents x Agents
the order of agents relative to the canonical ordering
if the entry is 1 then one is the other

Belief ::  Agents x Types
belief that each agent is of a given type

RationalBeliefs :: RationalAgents x Agents x Types

Alpha  :: Types x RationalTypes
how much each Type is valued by a particular RationalType

IsRationalType :: RationalTypes x Types
the entry i,j is 1 if j is the canonical ordering of the ith RationalType in the list of Types

AgentTypes :: Types x Agents 
i,j is 1 if agent j is of type i

RationalIndices :: Agents x RationalTypes
RationalIndices = (IsRationalType x AgentTypes)'
entry a,t is 1 if agent a is of type t, otherwise it is 0

Weight :: PlayingAgents x StaticTypes
How much each StaticType values the payoff of an agent as a function of their position

IsIrrationalType :: StaticTypes x Types
the entry i,j is 1 if j is the canonical ordering of the ith StaticType in the list of Types

Value :: Actions x Types
Value = Payoff x (Order x Belief x Alpha x IsRationalType + Weight x IsIrrationalType)
How much each Action is valued by each RationalType

Softmax :: A x B -> A x B
returns the softmax along each column

Likelihoods = Softmax(Value) :: Actions x Types
likelihood that a particular Action will be chosen by a particular Type

Action :: 1 x Actions

Likelihood :: 1 x Types
Action x Likelihoods

Observation = (Payoff,Action,Order)
Prior :: 1 x Types

NewBelief = prod(,Action x Likelihoods

KBelief :: RationalAgents x K x Agents x Agent x Types

RationalBeliefs :: RationalAgents x Agents x Types

Observers :: Observations x Agents
o,a is 1 if agent a observed observation o
Orders ::  Observations


rational_observers :: Observations x RationalAgents
rational_observers = Observers x RationalIndices
"""

"""
for (Payoff,Action,Order,Observers) in Observations:
    #raw_likelihoods :: RationalAgents x
    observer_beliefs = Observers x RationalIndices x RationalBeliefs
    raw_likelihoods = dot(Payoff,dot(dot(dot(dot(Order,RationalBeliefs),Alpha),IsRationalType))+dot(Weight,IsIrrationalType))
    
    softmax_likelihoods
    RationalBeliefs[flatten(RationalObservers)==1] = 
 """   

"""
Belief :: K x Agent x Agent
likelihood = Likelihood(k,A,B)
likelihood(k,A,B) = 
belief(k,A,B) = (Prior * likelihood/dot(Prior,likelihood)
"""

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
#dcc
#ccd

"""
feed beliefs then play out

show bimodality (violin plot)

forgiveness should show that reciprocals can perma-fuck-up with each other

different kinds of first impressions:
   permutations of observations

   quantity of observations

abstract tit for tat (beta inf)

AB|
"""
