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
from experiment_utils import log_init
from inspect import getargspec
sns.set(style="ticks")
class Experiment(object):
    """
    descendants of this class must have their __init__ decorated with log_init
    """
    def __init__(self, pickle_history=True, pickle_data=True, save_folder = 'experiments', cumulative = False):
        
        self.pickle_history = pickle_history
        self.pickle_data = pickle_data
        self.cumulative = cumulative
        
        self.name =  self.__class__.__name__
        self.conditions = '(%s)' % ", ".join(["%s=%s" % (arg,value) for arg,value in self.init_args.items()])
        self.trial_condition = '(%s)' % ", ".join(["%s=%s" % (arg,value) if arg is not "trials"
                                                   else "trials=%s" for arg,value in
                                                   self.init_args.items()])
        self.trial_dir = "%s/%s/%s/" % (save_folder,self.name,self.trial_condition)
        self.save_dir = "%s/%s/%s/" % (save_folder,self.name,self.conditions)
        self.cumulative_dir = "%s/%s/" % (save_folder,self.name)
        
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
        print 'Conditions:'+self.conditions
            

        #Run the actual experiment and pickle the historical data...        
        if not overwrite and os.path.isfile(save_dir+"record.pkl") and self.pickle_history:
            """
            there exists data and the overwrite flag is not set
            """
            print 'Experiment data already exists! Delete or set the overwrite flag to rerun experiment.\nData can be found at:',"./"+save_dir+"record.pkl"

            if process_data:
                """
                we're going to need the data, so load it
                """
                print "Loading cached record..." 
                record = unpickled(save_dir+"record.pkl")
                print "\t...done!"
        else:
            """
            we're making new data
            """
            if not os.path.exists(save_dir):
                print "Game directory does not exist. It will be created."
                print save_dir
                os.makedirs(save_dir)
                
            print "Running experiment..."
            record = self.procedure()

            if self.pickle_history:
                record = pickled(record,save_dir+"record.pkl")
                print "Experiment succesfully ran! Record saved to: %s" % save_dir+"record.pkl"
            else:
                print "Experiment succesfully ran! Record not saved."
  
        #Slice the historical data...
        if overwrite or process_data:
            self.process_data(record)
        elif plot:
            print "Print flag is set. Checking for plotting data."
            all_data_is_there = all([os.path.isfile(save_dir+file_name+".pkl")
                                     for file_name,x,y in self.processing_list])
            if all_data_is_there:
                print "Using cached data for plotting."
            else:
                print "Insufficient plotting data."
                if os.path.isfile(save_dir+"record.pkl"):
                    print "Unpickling stored data..."
                    record = unpickled(save_dir+"record.pkl")
                else:
                    print "Running experiment..."
                    record = self.procedure()
                    
                    if self.pickle_history:
                        record = pickled(record,save_dir+"record.pkl")
                        print "Experiment succesfully ran! Record saved to: %s" % save_dir+"record.pkl"
                    else:
                        print "Experiment succesfully ran! Record not saved."
                    
                self.process_data(record)
            

        #Plot the data...
        if plot:
            self.plot_data()

            
    def process_data(self,history):
        if self.cumulative:
            save_dir = self.cumulative_dir
        else:
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

            if self.cumulative:
                data_dir = self.cumulative_dir
            else:
                data_dir = save_dir
            data = unpickled(data_dir+file_name+".pkl")
            plot_function(data, outfile)
            print "\t...%s plot saved to:\n\t\t./%s" % (file_name,outfile)
        print ""

provoked,unprovoked = ("A defects then B defects", "Only B defects")
class meta_judgement(Experiment):
    """
    an observer sees manually created data about A and B interacting
    we plot the observer's beliefs about other's types and their meta-beliefs
    """
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
    """
    Observer sees A defect against B k times.
    for each k, how convinced are we that A is reciprocal?
    """
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


#first_impressions3().run(True,True,True)


class first_impressions_2(Experiment):
    """
    this experiment exposes agents to some observations then has them play it out normally.
    """
    @log_init
    def __init__(self,trials = 10, kind = 'seq',passive = False, sequential = False,**kwargs):
        assert kind in ['seq','perm']
 
        self.rational_type = rational_type = ReciprocalAgent
        self.agent_types = agent_types = [rational_type,SelfishAgent]
        self.processing_list = [('belief',partial(first_impressions_data,agent_types,[0,1]),
                                 first_impressions_plotter)]

    def procedure(self):
        trials = self.trials
        agent_types = self.agent_types
        rational_type = self.rational_type
        sequential = self.sequential
        
        BD = BinaryDictator(cost = 3, benefit = 6)
        plot_prehistory = True

        def char_to_observation(action_char):
            action = "give" if action_char is "C" else "keep"
            return [(BD,[0,1],[0,1],action)]
    
        def observations(actions_string):
            return map(char_to_observation,actions_string)

        total_actions = 5
        def int_to_actions_string(number):
            """
            given a number return a binary representation
            where 0s are Ds and 1s are Cs
            """
            return ("{0:0"+str(total_actions)+"b}").format(number).replace('1','C').replace('0','D')

        action_sequences = []
        if self.kind == 'seq':
            action_strings = ["C"*n+"D" for n in range(total_actions)]
        elif self.kind == 'perm':
            action_strings = [int_to_actions_string(n) for n in range(total_actions)]

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
                for action_string in ["D"]:
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
            return agent.belief_that(n_id,ReciprocalAgent)
        else:
            return belief_getter(agent.model[n_id].agent,n_id,k-1)


    for ((K,actions,trial),history), agent_type in product(record,[ReciprocalAgent]):
        shorthand = type_to_shorthand[agent_type]
        type_index = type_to_index[agent_type]
        for agent_id,target_id in permutations(agent_ids,2):
            for event in history:
                agent = event['players'][agent_id]
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
    plot_prehistory = False
    for K_level in range(3):
        data = pd.DataFrame(raw_data)
        data = data[(data['actions'] == 'D') & (data['K']== K_level)]
        if not plot_prehistory:
            data = data[(data['round'] >= 0)]
        
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

        #figure.fig.subplots_adjust(top =.80)
        plt.subplots_adjust(top = 0.93)
        figure.fig.suptitle("A and B's beliefs that the other is RA when A's first 3 moves are D")
    
        k_string = " K=%s" % K_level
        plt.savefig(out_path[:-4]+k_string+out_path[-4:]); plt.close()


def fi2():
    for passiveness in [True]:
        first_impressions_2(100,'perm',passive=passiveness).run(False,False,True)



def product_of_vals(orderedDict):
    keys,val_lists = orderedDict.keys(),orderedDict.vals()
    return [OrderedDict(zip(keys,vals)) for vals in apply(product,val_lists)]

def entry_exists(df,query_dict):
    query_string = " & ".join(["(%s == %s)" % (key,val) for key,val in query_dict.items()])
    return df.query(query_string) is not 0

def is_sequency(obj):
    if isinstance(obj,basestring):
        return False
    return isinstance(obj,collections.Sequence)
param_args = getargspec(default_params)[0]

def is_param_arg(string):
    return string in param_args

class fitness(Experiment):
    """
    under different population compositions and different global Ks
    what is the relative fitness of RA vs SA
    """
    @log_init
    def __init__(self, RA_K = [1], proportions = np.linspace(.1,.9,5), N_agents = 50, privacy="private", observability = .5, trials = 10,
                 RA_prior = .80, p_tremble = 0, numerator = ReciprocalAgent, denominator = SelfishAgent,  **kwargs):
        self.processing_list = [('fitness',fitness_slice,fitness_plotter)]
        tracked_values = self.tracked_values = ['RA_K','proportions','privacy','observability','RA_prior','p_tremble','N_agents','trial']

        self.rational_type = rational_type = ReciprocalAgent
        self.opposing_type = opposing_type = SelfishAgent

        args = copy(self.init_args)
        args['trial'] = range(trials)

        ### create the static params dict used to make the world 
        params = {key:val for key,val in args.items()
                  if is_param_arg(key) and not is_sequency(val)}
        self.agent_types = params['agent_types'] = [rational_type,opposing_type]
        self.game = params['games'] = RepeatedPrisonersTournament(10,visibility = privacy, observability=.5)
        static_params = self.default_params = default_params(**params)


        ### create the different conditions under which the experiment will run
        sequence_args = {key:val for key,val in args.items()
                    if is_sequency(val)}

        candidate_conditions = product_of_vals(sequence_args)
        self.target_conditions = target_conditions = []
        for condition in candidate_conditions:
            temp_params = copy(static_params)
            temp_params.update(condition)
            full_condition = OrderedDict({key:val for key,val in temp_params.items()
                                   if key in tracked_values})
            if not entry_exists(stored_data,full_condition):
                append(condition)
        
    def experiment(self):
        record = [];append = record.append
        for condition in self.target_conditions:
            trial,proportions = condition["trial","proportions"]
            np.random.seed(trial)

            params = default_params(**condition)
            
            world = World(params,generate_proportional_genomes(params,proportions))
            fitness,history = world.run()

            ordered_types = [type(agent) for agent in world.agents]
            fitnesses = default_dict(int)
            for agent_type,fitness_score in zip(ordered_types,fitness):
                fitnesses[agent_type] += fitness_score
                
            for agent_type in params["agent_types_world"]:
                fitnesses[agent_type] /= ordered_types.count(agent_type)

            condition['relative_avg_fitness'] = fitnesses[self.rational_type]/fitnesses[self.opposing_type]
            
            append(condition)

        return record
    def procedure(self):
        agent_types = self.agent_types
        rational_type = self.rational_type
        opposing_type = self.opposing_type
        RA_prior = self.RA_prior
        pop_size = self.N_players
        record = []; append = record.append
        trial_start = 0
        
        trials = self.trials
        
        if False:
            for t in reversed(range(trials)):
                trial_dir = self.trial_dir % t
                if os.path.isfile(trial_dir +"history.pkl"):
                    trial_start = t
                    record = unpickled(trial_dir+"history.pkl")
                    append = record.append
                    break
        assert len(record) is not 0 if trial_start is not 0 else True
        trials = range(trial_start,trials)
    
        
        conditions = product(self.Ks,self.proportions,trials)
        for K,proportion,trial in conditions:
            print "K = %s, RA pop = %s, trial = %s/%s" % (K,proportion,trial+1,self.trials)
            np.random.seed(trial)
            params = default_params(agent_types,RA_prior,N_agents = pop_size)
            params['RA_K'] = K
            params['tremble'] = self.tremble
            params['games'] = self.game
            proportions = {rational_type:proportion,
                           SelfishAgent:1-proportion}
            world = World(params,generate_proportional_genomes(params,proportions))
            fitness,history = world.run()
            ordered_types = [type(agent) for agent in world.agents]

            fitnesses = {rational_type:0, opposing_type:0}
            for agent_type,fitness_score in zip(ordered_types,fitness):
                fitnesses[agent_type] += fitness_score
                
            for agent_type in [rational_type,opposing_type]:
                fitnesses[agent_type] /= ordered_types.count(agent_type)
            relative_avg_fitness = fitnesses[rational_type]/fitnesses[opposing_type]
            append(((trial,K,proportion,pop_size,rational_type,opposing_type,self.privacy,self.observability),relative_avg_fitness))

        return record
class hashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.iteritems())))


class fitness_v_selfish(Experiment):
    """
    under different population compositions and different global Ks
    what is the relative fitness of RA vs SA
    """
    @log_init
    def __init__(self, RA_K = [1], proportions = np.linspace(.1,.9,5), N_agents = 50, visibility = "private",
                 observability = .5, trial = 10, RA_prior = .80, p_tremble = 0, agent_type = ReciprocalAgent, rounds = 10, **kwargs):

        self.processing_list = [('fitness',identity,fitness_plotter)]

        expected_args = copy(self.init_args)
        if 'trial' in expected_args and not is_sequency(expected_args['trial']):
            expected_args['trial'] = range(expected_args['trial'])
    
        ### create the static params dict used to make the world 
        static_params = {key:val for key,val in expected_args.items() if not is_sequency(val)}

        ### create the different conditions under which the experiment will run
        variable_params = {key:val for key,val in expected_args.items() if is_sequency(val)}

        stored_data = pd.read_pickle(self.cumulative_dir+'data.pkl')

        #correctly format the parameters for each condition
        if variable_params:
            candidate_conditions = product_of_vals(variable_params)
        else:
            candidate_conditions = [static_params]

        self.target_conditions = [];append = self.target_conditions.append
        for params in candidate_conditions:
            condition = copy(static_params)
            condition.update(params)
            
            #for convenient referencing
            params = multiRefOrderedDict(condition)
            
            #only add to target_conditions if no data exists for this condition
            if not dict_query(stored_data,fitness_data_prettify(condition)):
                append(condition)
            

    def procedure(self):
        if self.target_conditions:
            record = [];append = record.append
            for condition in self.target_conditions:
                trial,agent_type,p = condition["trial","agent_type","proportions"]
                np.random.seed(trial)
                
                params = copy(condition)
                params['agent_types'] = [agent_type,SelfishAgent]
                params['games'] = RepeatedPrisonersTournament(**condition)
                params = default_params(**condition)
                
                proportions = {p:agent_type, 1-p:SelfishAgent}
                
                world = World(params,generate_proportional_genomes(params,proportions))
                fitness,history = world.run()
                
                ordered_types = [type(agent) for agent in world.agents]
                fitnesses = default_dict(int)
                for agent_type,fitness_score in zip(ordered_types,fitness):
                    fitnesses[agent_type] += fitness_score
                
                for agent_type in params["agent_types_world"]:
                    fitnesses[agent_type] /= ordered_types.count(agent_type)

                condition['relative_avg_fitness'] = fitnesses[condition['agent_type']]/fitnesses[SelfishAgent]
            
                append(fitness_data_prettify(condition))
        pd.DataFrame(record).to_pickle(self.cumulative_dir+'data.pkl')
        return record
    
def fitness_data_prettify(record):
    rename_dict = {"RA_K":"K",
                   "proportions":"Population Percentage",
                   "N_agents":"Population Size",
                   "fitness":"Relative Average Fitness",
                   "privacy":"Privacy",
                   "observability":"Observability",}
    data = [dict((rename_dict[key],val) if key in rename_dict else (key,val) for key,val in entry) for entry in record]

    return data


def dict_query(df,vals_dict):
    item_queries = [];append = item_queries.append
    for key,vals in vals_dict.items():
        if is_sequency(vals):
            append("(%s)" % " | ".join(["(%s == %s)" % (key,val) for val in vals]))
        else:
            append("(%s == %s)" % (key,vals))
    query_string =  " & ".join(item_queries)
    return df.query(query_string)

def fitness2_plotter(data,out_path):
    data = pd.read_pickle(self.cumulative_dir+'data.pkl')
    data = dict_query(data,self.init_args)
    sns.pointplot(x = "Population Percentage",y = "Relative Average Fitness", data = data,hue = "K")
    plt.savefig(out_path); plt.close()


fitness(trials = 10,tremble = 0,pop_size = 100,proportions = np.linspace(.07,.1,9),privacy='private').run(True,False,True)





"end"
