from __future__ import division
import pandas as pd
import seaborn as sns
from experiment_utils import multi_call,experiment,plotter
import numpy as np
from indirect_reciprocity import World,default_params,generate_proportional_genomes,default_genome
from indirect_reciprocity import ReciprocalAgent,SelfishAgent,AltruisticAgent
from experiment_utils import is_sequency
from games import RepeatedPrisonersTournament
from collections import defaultdict
import matplotlib.pyplot as plt

#print is_sequency(np.linspace(.1,.9,5))

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
    genomes = [default_genome(params,agent_type) for agent_type in agent_types]
    world = World(params,genomes)

    fitness,history = world.run()
    return fitness


@plotter()
def binary_matchup_plot(data=binary_matchup(rounds=10,cost=1,benefit=3,trial=1000), save_dir="./plots/", save_file="binary_matchup.pdf"):
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
        figure = sns.jointplot("ReciprocalAgent","SelfishAgent",data=ndata.query('RA_prior == %s' % RA_prior), color="g",
                               xlim=(min_tick,max_tick),ylim=(min_tick,max_tick), kind = "hex")
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
binary_matchup_plot()
