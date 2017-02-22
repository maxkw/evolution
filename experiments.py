from __future__ import division
import pandas as pd
import seaborn as sns
from experiment_utils import multi_call,experiment,plotter
import numpy as np

from params import default_params,generate_proportional_genomes,default_genome
from indirect_reciprocity import World,ReciprocalAgent,SelfishAgent,AltruisticAgent
from games import RepeatedPrisonersTournament,BinaryDictator,Repeated,PrivatelyObserved,Symmetric
from collections import defaultdict
from itertools import combinations_with_replacement as combinations
from itertools import permutations
from itertools import product,islice,cycle
import matplotlib.pyplot as plt
from numpy import array
from copy import copy,deepcopy


###multi_call

def fitness_v_selfish_plot(data,save_file):
    #import pdb; pdb.set_trace()
    print data
    sns.pointplot(x = "proportion", y = "return", data = data, hue = "RA_K")
    plt.savefig(save_file); plt.close()

#@multi_call()

@experiment(fitness_v_selfish_plot)
def fitness_v_selfish(RA_K = [1], proportion = [round(n,5) for n in np.linspace(.1,.9,5)], N_agents = 10, visibility = "private", observability = .5, trial = range(10), RA_prior = .80, p_tremble = 0, agent_type = ReciprocalAgent, rounds = 10):

    """
    Make a population of size 'N_agents' where 'proportion' percent of the population is of type 'agent_type'
    and the remainder is of type 'SelfishAgent'. have them play RepeatedPrisonersTournament with each other
    return the ratio of average payoffs to agent_type vs average payoffs to SelfishAgent.
    """

    condition = locals()
    condition['agent_types'] = [agent_type,SelfishAgent]
    condition['games'] = RepeatedPrisonersTournament(**condition)

    params = default_params(**condition)

    proportions = {agent_type:proportion,
                   SelfishAgent:1-proportion}

    world = World(params,generate_proportional_genomes(params,proportions))
    #print world.agents
    fitness,history = world.run()

    ordered_types = [type(agent) for agent in world.agents]
    fitnesses = defaultdict(int)

    for a_type,fitness_score in zip(ordered_types,fitness):
        fitnesses[a_type] += fitness_score

    for a_type in params["agent_types_world"]:
        fitnesses[a_type] /= ordered_types.count(a_type)

    #print fitnesses
    relative_avg_fitness = fitnesses[agent_type]/fitnesses[SelfishAgent]

    return relative_avg_fitness



@multi_call(unordered = ['agent_types'])
def binary_matchup(agent_types=[(ReciprocalAgent,SelfishAgent)],RA_prior=[.25,.50,.75],trial=100,rounds=range(1,20),cost=[0,1,2,3],benefit=[0,1,2,3]):
    condition = locals()
    condition['games'] = RepeatedPrisonersTournament(**condition)
    params = default_params(**condition)
    genomes = [default_genome(agent_type = agent_type, **condition) for agent_type in agent_types]
    world = World(params,genomes)

    fitness,history = world.run()
    return fitness


@plotter()
def binary_matchup_plot(data, save_dir="./plots/", save_file="binary_matchup.pdf"):


    dicts = data.to_dict('index')
    types = []
    l=[]
    for row in dicts.itervalues():
        types = agent_types = [str(a) for a in row['agent_types']]
        new_entries = dict(zip(agent_types,row['return']))
        row.update(new_entries)
        l.append(row)
    ndata = pd.DataFrame(l)
    #print ndata['SelfishAgent']
    #sns.factorplot(data=ndata,x='SelfishAgent',y='ReciprocalAgent',row='RA_prior', kind='point')
    reward_ticks = list(set(list(ndata['ReciprocalAgent'])+list(ndata['SelfishAgent'])))
    max_tick = int(max(reward_ticks))
    min_tick = int(min(reward_ticks))
    ticks = range(min_tick, max_tick,2)#np.linspace(min_tick,max_tick,12)
    for RA_prior in set(ndata['RA_prior']):
        print ndata.query('RA_prior == %s' % RA_prior)
        figure = sns.jointplot("ReciprocalAgent","SelfishAgent",kind = "kde", data=ndata.query('RA_prior == %s' % RA_prior), color="g",
                                #xlim=(min_tick,max_tick),ylim=(min_tick,max_tick), kind = "kde"
        )
        #plt.ylim([min_tick,max_tick])
        #plt.xlim([min_tick,max_tick])
        print type(figure)
        #figure.set(yticks = ticks,xticks = ticks)
        save_str = "binary_matchup - prior=%s.pdf" % RA_prior
        plt.savefig(save_dir+save_str)



@multi_call()
def first_impressions(RA_K=2,RA_prior=[.25,.5,.75], rational_type = ReciprocalAgent, agent_types = [[ReciprocalAgent,SelfishAgent]],N_cooperation=5):
    """
    An observer 'O' sees an agent 'A' cooperate 'N_cooperation' times with another agent 'B' before defecting once.
    What is O's belief that A is of type 'rational_type'?
    """
    params = default_params(RA_K =  RA_K, RA_prior = RA_prior, agent_types = agent_types)
    action_strings = ["C"*n+"D" for n in range(N_cooperation)]
    BD = BinaryDictator()

    def char_to_observation(action_char,order=[0,1]):
        action = "give" if action_char is "C" else "keep"
        return [(BD,order,[0,1,"O"],action)]

    def observations(actions_string):
        return map(char_to_observation,actions_string)

    record = []; append = record.append
    for OAB, OBA in [(RA_prior,RA_prior)]:
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


#fitness_v_selfish(N_agents = 20,trial =  range(10))


#DecisionSeq([(Symmetric(BinaryDictator(cost=cost,benefit=benefit)),[0,1]) for cost,benefit in cb_list])

def unordered_prior_combinations(prior_list = np.linspace(.75,.25,3)):
    return map(tuple,map(sorted,combinations(prior_list,2)))

priors_for_RAvRA = map(tuple,map(sorted,combinations(np.linspace(.75,.25,3),2)))
print priors_for_RAvRA
diagonal_priors = [(n,n) for n in np.linspace(.75,.25,3)]

@multi_call(unordered = ['agent_types'],verbose=1)
def RAvRA(priors = priors_for_RAvRA, agent_types = [(ReciprocalAgent,SelfishAgent,AltruisticAgent),(ReciprocalAgent,SelfishAgent)],trial = 100, RA_K = 1,games = RepeatedPrisonersTournament()):
    condition = locals()
    params = default_params(**condition)
    genomes = [default_genome(agent_type = ReciprocalAgent,RA_prior = prior,**condition) for prior in priors]
    genome = genomes[0]
    #print genome['prior']
    #print genomes[0]
    world = World(params = params, genomes = genomes)
    fitness,history = world.run()

    return fitness

@plotter()
def RAvRA_plot(data, save_dir="./plots/", save_file="RAvRA.pdf"):

    dicts = data.to_dict('records')
    
    agents = []
    l=[]
    for row in dicts:
        agents = ["ReciprocalAgent #%s Reward" % prior for prior in range(2)]
        new_entries = dict(zip(agents,row['return']))
        row.update(new_entries)
        l.append(row)
    ndata = pd.DataFrame(l)
    
    
    #print ndata['SelfishAgent']
    #sns.factorplot(data=ndata,x='SelfishAgent',y='ReciprocalAgent',row='RA_prior', kind='point')
    #reward_ticks = list(set(sum(map(list,[ndata[agent] for agent in agents]))))
    
    #max_tick = int(max(reward_ticks))
    #min_tick = int(min(reward_ticks))
    #ticks = range(min_tick, max_tick,2)#np.linspace(min_tick,max_tick,12)

    for arg_hash in set(ndata['arg_hash']):
        data =  ndata.query('arg_hash == %s' % arg_hash)
        priors = list(data.priors)[0]
        agent_types = list(data.agent_types)[0]
        rewards = sorted(list(set(sum(map(list,data['return']),[]))))
        lims = (min(rewards),max(rewards))
        agents = ["ReciprocalAgent #%s Reward" % prior for prior in range(2)]
        figure = sns.jointplot(agents[0],agents[1], data=data, color="g",kind = 'kde',xlim=lims,ylim=lims,bw=(.75,.75),
                               n_levels = 15
        )
        figure.set_axis_labels(*["RA(prior = %s)" % prior for prior in priors])
        #plt.ylim([min_tick,max_tick])
        #plt.xlim([min_tick,max_tick])
        #print type(figure)
        #figure.set(yticks = ticks,xticks = ticks)
        save_str = "RAvRA(priors = %s, agent_types = %s,trials = %s).pdf" % (priors,agent_types,len(data))
        plt.savefig(save_dir+save_str)

from games import RepeatedDynamicPrisoners
def compare_RA(prior_lst = np.linspace(.25,.75,7)):
    size = [len(prior_lst)]*2
    prior_2_index = dict(map(reversed,enumerate(prior_lst)))
    lookup = lambda p: prior_2_index[p]
    priors = unordered_prior_combinations(prior_lst)
    data = RAvRA(priors = priors, trial = 100, RA_K = 1, agent_types = [(SelfishAgent,ReciprocalAgent)])

    arr = np.empty(size)
    for priors,group in data.groupby('priors'):
        p0,p1 = map(lookup,priors)
        r0,r1 = list(group['return'].mean())
        arr[(p0,p1)] = round(r0,4)
        arr[(p1,p0)] = round(r1,4)

    print [round(n,3) for n in prior_lst]
    print arr
    return arr

from games import RepeatedSequentialBinary
from indirect_reciprocity import NiceReciprocalAgent
def comparison_plotter(prior_list = [.25,.50,.75]):
    priors = unordered_prior_combinations(prior_list)
    print priors
    game = RepeatedSequentialBinary()
    data = RAvRA(priors = priors, trial = 100, RA_K = 2,agent_types = [(SelfishAgent,ReciprocalAgent,AltruisticAgent)],games = game)
    record = []

    rewards = sorted(list(set(sum(map(list,data['return']),[]))))
    for priors,group in data.groupby('priors'):
        p0,p1 = priors
        for r0,r1 in group['return']:
            record.append({'recipient prior':p0, 'opponent prior':p1, 'reward':r0})
            record.append({'recipient prior':p1, 'opponent prior':p0, 'reward':r1})
    data = pd.DataFrame(record)
    
    #data = data.pivot("recipient prior","opponent prior","reward")
    #print data
    meaned = data.groupby(['recipient prior','opponent prior']).mean().unstack()


    def meandrawer(data,**kwargs):
        plt.axvline(data.mean())
    #g = sns.FacetGrid(data,row='recipient prior',col='opponent prior',margin_titles=True)
    #g = (g.map(sns.kdeplot,"reward")
    #    .map(meandrawer,"reward"))
    sns.heatmap(meaned,annot=True,fmt="f")
    #plt.show()
    plt.savefig("./plots/compare.pdf")


@multi_call(unpack = 'record',unordered = ['agent_types'])
def first_impressions(RA_K = 2, RA_prior = .75, preactions = 5, kind = 'seq', agent_type = ReciprocalAgent, agent_types = [(ReciprocalAgent, SelfishAgent)],passive = False,trial = 50):
    condition = locals()
    params = default_params(game = RepeatedPrisonersTournament(10),**condition)
    genomes = [default_genome(**condition)]*2

    BD = BinaryDictator()
    def int_to_actions_string(number):
            """
            given a number return a binary representation
            where 0s are Ds and 1s are Cs
            """
            return ("{0:0"+str(preactions)+"b}").format(number).replace('1','C').replace('0','D')

    if kind == 'seq':
        action_strings = ["C"*preactions+"D"]
        #action_strings = ["D"*3]
    elif kind == 'perm':
        action_strings = [int_to_actions_string(n) for n in range(preactions)]

    action_strings = ["DDD"]

    def char_to_observation(action_char):
        action = "give" if action_char is "C" else "keep"
        return [(BD,[0,1],[0,1],action)]

    def observations(actions_string):
        return map(char_to_observation,actions_string)

    a_ids = [list(islice(cycle(l),0,n)) for l,n in product([(0,1),(1,0)],range(2,RA_K+3))]
    a_ids = [(l[0],l[1:]) for l in a_ids]

    record = []
    #have agents observe the prehistoric observations
    #if second agent gets to react, have them act appropriately
    for action_string in action_strings:
    #make the prehistory and save the initial state of beliefs of agents

        world = World(params, genomes)
        agents = world.agents
        action_len = len(action_string)
        prehistory  = [{
            'round':-action_len,
            'belief': [copy(a.belief) for a in world.agents],
            'prior': RA_prior,
            'players': [deepcopy(agent) for agent in world.agents],
            'action':action_string
        }]

        for r,observation in enumerate(observations(action_string),-(action_len-1)):
            if not passive:
                new_observation = BD.play(world.agents[array([1,0])],world.agents,tremble = 0)[1]
                observation = observation+new_observation
            for agent in world.agents:
                agent.observe(observation)

            prehistory.append({
                'round':r,
                'belief': [copy(a.belief) for a in world.agents],
                'prior': RA_prior,
                'players': [deepcopy(agent) for agent in world.agents],
                'action':action_string,
            })

        fitness,history = world.run(notes = {'action':action_string})

        
        for event in prehistory+history:
            for agent_id,ids in a_ids:
                    agent = event['players'][agent_id]
                    k = len(ids)-1
                    record.append({
                        'K': RA_K,
                        'k':k,
                        'round': event['round'],
                        'actions': event['action'],
                        'belief': agent.k_belief(ids,ReciprocalAgent),
                        'believer':agent_id,
                        'type': "RA",
                        })
    return record

def fi_data_slice(data):

    ret = []
    for record in data.to_dict('records'):
        K = int(record['RA_K'])
        a_ids = [list(islice(cycle(l),0,n)) for l,n in product([(0,1),(1,0)],range(2,K+3))]
        a_ids = [(l[0],l[1:]) for l in a_ids]
        for history in record['return']:
            for event in history:
                for agent_id,ids in a_ids:
                    agent = event['players'][agent_id]
                    k = len(ids)-1
                    ret.append({
                        'K': K,
                        'k':k,
                        'round': event['round'],
                        'actions': event['action'],
                        'belief': agent.k_belief(ids,ReciprocalAgent),
                        'believer':agent_id,
                        'type': "RA",
                        })

    return pd.DataFrame(ret)



def first_impressions_plotter(out_dir = "./plots/"):
    K = 1
    data = first_impressions(trial = 100,kind = 'seq', RA_prior = .75, preactions = 3, RA_K=K,passive=False)
    figure = sns.factorplot('round','belief', hue = 'type',#'actions',
                            col='believer',row='k',data=data, ci=68,legend = False,aspect = 1, size = 4.5, #kind = 'point')
                            kind = 'violin',scale ='area', width=.9,cut = 0,inner = 'point',bw = .2)
    #figure = sns.factorplot('round','belief', hue = None, col='type',row='actions',data=data, ci=68,legend_out = True,aspect = 1.5,size = 5)#kind = 'violin',scale ='count', width=.9,cut = .5,inner = 'box')
    #size = figure.get_size_inches()
    #figure.set_size_inches((size[0]*1,size[1]*1.5))
    figure.set(yticks=[x/10.0 for x in range(11)])
    y_buff = .0
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
        
    #figure.fig.subplots_adjust(top =.80)
    plt.subplots_adjust(top = 0.93)
    figure.fig.suptitle("A and B's beliefs that the other is RA when A's first 3 moves are D")
    plt.show()
    
first_impressions_plotter()
    
    

#compare_RA()
#RAvRA_plot(RAvRA(trial = 1000))
#comparison_plotter(np.linspace(0,1,5))
#RAvRA_plot(RAvRA())
#print RAvRA(trial=1)
