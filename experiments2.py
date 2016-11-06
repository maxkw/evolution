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

sns.set(style="ticks")
class Experiment(object):
    def __init__(self,name,conditions,processing_list,pickle_history=True,pickle_data=True):
        """
        name is a string
        conditions is too
        processing list is a list of triples consisting of a 
            filename(no extension)
            a function that takes a historical record and returns a pd.DataFrame
            a function that takes a pd.DataFrame and plots it
        """
        self.name = name
        self.conditions = conditions
        self.processing_list = processing_list
        self.pickle_history = pickle_history
        self.pickle_data = pickle_data
        
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
        if not overwrite and self.pickle_history and os.path.isfile(save_dir+"history.pkl"):
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
            results = self.procedure()

            if self.pickle_history:
                history = pickled(results,save_dir+"history.pkl")
            else:
                history = results
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

provoked,unprovoked = ("A defects then B defects", "Only B defects")
class meta_judgement(Experiment):
    def __init__(self,K=2):
        self.K = K
        self.name = self.__class__.__name__
        self.conditions = "(K=%s)" % (K,)
        
        
        self.save_dir = 'experiments/%s/%s/' % (self.name,self.conditions)
        
        self.rational_type = rational_type = ReciprocalAgent
        self.agent_types = agent_types = [rational_type,SelfishAgent]
        self.processing_list = [('belief',
                                 partial(observer_slice,K),
                                 partial(observer_plotter,K)),
                                ('delta belief',
                                 delta_belief_slice,
                                 delta_belief_plotter)]

        self.pickle_history = True
        self.pickle_data = True

        
    def procedure(self):
        agent_types = self.agent_types
        rational_type = self.rational_type
        BD = BinaryDictator()

        def char_to_observation(action_char,order=[0,1]):
            action = "give" if action_char is "C" else "keep"
            return [(BD,order,[0,1,"O"],action)]
    
        def observations(actions_string):
            return map(char_to_observation,actions_string)
            
        

        def set_belief(believer,thinker_id,target_id,belief):
            believer.model[thinker_id].belief[target_id] = prior_generator(agent_types,belief)
        cases = [(provoked,[char_to_observation("D",[0,1]),char_to_observation("D",[1,0])]),
                 ("Forgiveness",[char_to_observation("D",[0,1]),char_to_observation("C",[1,0])]),
                 ("CD",[char_to_observation("D",[0,1])+char_to_observation("C",[1,0]),char_to_observation("C",[0,1])]),
                 ("DC",[char_to_observation("C",[0,1])+char_to_observation("C",[1,0]),char_to_observation("D",[0,1])]),
                 (unprovoked,[char_to_observation("D",[1,0])])]

        RA_prior = .75
        params = default_params(agent_types,RA_prior)
        params['RA_K'] = self.K
        
        record = []; append = record.append
        for CAB, CBA in [(RA_prior,RA_prior)]:#product(np.linspace(.25,.75,3),repeat=2):
        
            observer = RationalAgent(default_genome(params,RationalAgent),'O')
            set_belief(observer,0,1,{rational_type:CAB})
            set_belief(observer,1,0,{rational_type:CBA})
          
            for case_name,observations in cases:
                print case_name
                new_observer = deepcopy(observer)
                history = [
                    {'round':0,
                     'observer':deepcopy(new_observer),
                     'CAB':CAB,
                     'CBA':CBA,
                     'proir':RA_prior,
                     'case':case_name
                    }]
            
                for i,observation in enumerate(observations):
                    print observation
                    new_observer.observe(observation)
                    print new_observer.belief_that(0,ReciprocalAgent)
                    print new_observer.belief_that(1,ReciprocalAgent)
                    history.append({'round':i+1,
                                    'observer':deepcopy(new_observer),
                                    'CAB':CAB,
                                    'CBA':CBA,
                                    'prior':RA_prior,
                                    'case':case_name
                    })
                record.append(((RA_prior,CAB,CBA,case_name),history))
        return record
                        

def identity(x):
    return x


def delta_belief_slice(record):
    data = [];append = data.append
    for (conditions,history) in record:
        prior,cab,cba,case = conditions
        observer = history[-1]['observer']
        for event, agent_id in product(history,[0,1]):
            append({'Condition':case,
                    'CBA':cba,
                    'CAB':cab,
                    'Agent':"Agent A" if agent_id == 0 else "Agent B",
                    'Final Belief':observer.belief_that(agent_id,ReciprocalAgent)
                    })
    return data

def delta_belief_plotter(data,out_path):
    data = pd.DataFrame(data)
    data = data[(data['Condition'] == provoked) | (data['Condition'] == unprovoked)]
    data = data[(data['Agent'] =="Agent B")]
    fplot = sns.factorplot(data = data, x="Condition",y="Final Belief",col="CBA",ci= None,row="CAB",hue="Agent",kind='bar', legend = False)
    fplot.despine(bottom=True)
    plt.tick_params(bottom='off')
    fplot.set(yticks=np.linspace(0,1,5))
    axis = fplot.facet_axis(0,0)
    axis.set_yticklabels(['','0.25','0.50','0.75','1.0'])
    axis.set(title="",xlabel="Observed Actions" ,ylabel="Belief that B is Reciprocal")
    for item in axis.get_xticklabels():
        item.set_fontsize(8)
        
    #plt.legend(frameon=True,prop={'size':10})
    fplot.fig.subplots_adjust(top =.90)
    plt.axhline(y=.75,ls=':')
    fplot.fig.suptitle("Provoked vs Unprovoked Defection")
    plt.savefig(out_path);plt.close()

def observer_slice(K,record):
    agent_ids = [0,1]
    agent_types = [ReciprocalAgent,SelfishAgent]
    #K -= 1
    data = [];append = data.append
    def justcaps(t):
        return filter(str.isupper,t.__name__)
    
    
    type_to_index = dict(map(reversed,enumerate(agent_types)))
    type_to_shorthand = {t:justcaps(t) for t in agent_types}

    def belief_getter(agent,a_id,k):
        n_id = (a_id+1)%2
        print k,n_id
        if k == 0:
            return agent.belief_that(a_id,ReciprocalAgent)
        else:
            return belief_getter(agent.model[a_id].agent,n_id,k-1)

    for (conditions,history), agent_type in product(record,[ReciprocalAgent]):
        prior,cab,cba,case = conditions
        shorthand = type_to_shorthand[agent_type]
        type_index = type_to_index[agent_type]
        for agent_id,target_id in permutations(agent_ids,2):
            for event in history:
                observer = event['observer']
                for k in range(K+1):
                    append({
                        'k':k,
                        'CAB':cab,
                        'CBA':cba,
                        'round': event['round'],
                        'case': event['case'],
                        'belief': belief_getter(observer,agent_id,k),
                        'believer':agent_id,
                        'type': shorthand,
                })
    return data

def observer_plotter(K,raw_data,out_path):

    raw_data = pd.DataFrame(raw_data)
    raw_data = raw_data[(raw_data['CAB'] == .75) & (raw_data['CBA'] == .75)]
    for case in [provoked,unprovoked,"CD","DC"]:
        
        data = raw_data[(raw_data['case'] == case)]
        
        figure = sns.factorplot('round','belief', hue = 'case',
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
        for k,(a_id,t_id) in product(range(K+1),permutations(range(2))):
            axis = figure.facet_axis(k,a_id)
            sns.pointplot(x = "round", y="belief", color = "red",data = data[(data['believer']==a_id) & (data['k']==k)],ax=axis)
            
            axis.set(#xlabel='# of interactions',
                ylabel = '$\mathrm{Pr_{%s}( T_{%s} = RA | O_{1:n} )}$'% (k,id_to_letter[t_id]),
                title = "%s's K=%s belief that %s is RA" % (id_to_letter[a_id],k,id_to_letter[t_id]))
        
        plt.subplots_adjust(top = 0.93)
        figure.fig.suptitle("C's beliefs about A and B's beliefs that the other is RA")
    
        k_string = " case = %s" % case
        plt.savefig(out_path[:-4]+k_string+out_path[-4:]); plt.close()

for K in [0,1]:
    pass
    #meta_judgement(K).run(True,True,True)


class first_impressions3(Experiment):
    def __init__(self,K=2,RA_prior = .75):
        self.K = K
        self.name = self.__class__.__name__
        self.conditions = "(K=%s)" % (K,)
        
        
        self.save_dir = 'experiments/%s/%s/' % (self.name,self.conditions)
        
        self.rational_type = rational_type = ReciprocalAgent
        self.agent_types = agent_types = [rational_type,SelfishAgent]
        self.processing_list = [('belief',
                                 identity,
                                 first_impressions3_plotter)]

        self.pickle_history = True
        self.pickle_data = True

        
    def procedure(self):
        agent_types = self.agent_types
        rational_type = self.rational_type
        BD = BinaryDictator()

        def char_to_observation(action_char,order=[0,1]):
            action = "give" if action_char is "C" else "keep"
            return [(BD,order,[0,1,"O"],action)]
    
        def observations(actions_string):
            return map(char_to_observation,actions_string)

        def c_to_the_k_then_d(k):
            return 

        total_actions = 5
        action_strings = ["C"*n+"D" for n in range(total_actions)]

        RA_prior = .75
        params = default_params(agent_types,RA_prior)
        params['RA_K'] = self.K
        
        record = []; append = record.append
        for CAB, CBA in [(RA_prior,RA_prior)]:#product(np.linspace(.25,.75,3),repeat=2):
            for action_string in action_strings:
                observer = RationalAgent(default_genome(params,RationalAgent),'O')

                for i,observation in enumerate(observations(action_string)):                    
                    observer.observe(observation)
                record.append(
                    {
                        'belief':observer.belief_that(0,rational_type),
                        'action':action_string,
                        'cooperations':i,
                        'prior':RA_prior,
                    })
        return record

def first_impressions3_plotter(raw_data,out_path):
    data = pd.DataFrame(raw_data)
    figure = sns.pointplot(x='cooperations',y='belief',color = "red",data = data)
    figure.set(yticks=[x/10.0 for x in range(11)])
    plt.savefig(out_path);plt.close()


first_impressions3().run(True,True,True)
