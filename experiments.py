from __future__ import division
import pandas as pd
import seaborn as sns
from experiment_utils import multi_call,experiment,plotter
import numpy as np

from params import default_params,generate_proportional_genomes,default_genome
from indirect_reciprocity import World,ReciprocalAgent,SelfishAgent,AltruisticAgent
from games import RepeatedPrisonersTournament
from collections import defaultdict
from itertools import combinations_with_replacement as combinations
from itertools import product
import matplotlib.pyplot as plt


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

priors_for_RAvRA = map(tuple,map(sorted,combinations(np.linspace(.75,.25,3),2)))
diagonal_priors = [(n,n) for n in np.linspace(.75,.25,3)]

@multi_call(unordered = ['agent_types'],verbose=3)
def RAvRA(priors = priors_for_RAvRA, agent_types = [(ReciprocalAgent,SelfishAgent,AltruisticAgent),(ReciprocalAgent,SelfishAgent)],trial = 100):
    condition = locals()
    params = default_params(**condition)
    genomes = [default_genome(agent_type = ReciprocalAgent,RA_prior = prior,**condition) for prior in priors]
    genome = genomes[0]
    #print genome['prior']
    #print genomes[0]
    world = World(params = params, genomes = genomes)
    fitness,history = world.run()

    return fitness

#print RAvRA()
#print binary_matchup_plot(binary_matchup(rounds=10,cost=1,benefit=3,trial=1000))

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
        figure = sns.jointplot(agents[0],agents[1], data=data, color="g",kind = 'kde', xlim=lims,ylim=lims)
        figure.set_axis_labels(*["RA(prior = %s)" % prior for prior in priors])
        #plt.ylim([min_tick,max_tick])
        #plt.xlim([min_tick,max_tick])
        #print type(figure)
        #figure.set(yticks = ticks,xticks = ticks)
        save_str = "RAvRA(priors = %s, agent_types = %s,trials = %s).pdf" % (priors,agent_types,len(data))
        plt.savefig(save_dir+save_str)

def compare_RA():
    from numpy import array
    prior_lst = [.25,.5,.75]
    prior_2_index = dict(map(reversed,enumerate(prior_lst)))
    lookup = lambda p: prior_2_index[p]
    data = RAvRA(priors = priors_for_RAvRA, trial = 200, agent_types = [(SelfishAgent,ReciprocalAgent)])
    lst = []
    
    arr = np.empty([3,3])
    for priors,group in data.groupby('priors'):
        p0,p1 = map(lookup,priors)
        r0,r1 = group['return'].mean()
        arr[(p0,p1)] = r0
        arr[(p1,p0)] = r1
    print prior_lst
    print arr

compare_RA()
#RAvRA_plot(RAvRA(priors = diagonal_priors,trial = 1000))
#RAvRA_plot(RAvRA())
#print RAvRA(trial=1)
