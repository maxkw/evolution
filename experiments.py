from __future__ import division
import pandas as pd
import seaborn as sns
from experiment_utils import multi_call,experiment,plotter,MultiArg
import numpy as np
from params import default_params,generate_proportional_genomes,default_genome
from indirect_reciprocity import World,ReciprocalAgent,SelfishAgent,AltruisticAgent,NiceReciprocalAgent,RationalAgent
from games import RepeatedPrisonersTournament,BinaryDictator,Repeated,PrivatelyObserved,Symmetric
from collections import defaultdict
from itertools import combinations_with_replacement as combinations
from itertools import permutations
from itertools import product,islice,cycle
import matplotlib.pyplot as plt
from numpy import array
from copy import copy,deepcopy


###multi_call

def justcaps(t):
    return filter(str.isupper,t.__name__)

@multi_call(unordered = ['agent_types'], twinned = ['player_types','priors','Ks'])
@experiment(unpack = 'dict', trials = 100)
def binary_matchup(player_types = (NiceReciprocalAgent,NiceReciprocalAgent), priors = (.75, .75), Ks=(1,1), agent_types = (NiceReciprocalAgent,SelfishAgent),**kwargs):
    condition = dict(locals(),**kwargs)
    params = default_params(**condition)
    genomes = [default_genome(agent_type = t, RA_prior=p ,RA_K = k, **condition) for t,p,k in zip(player_types,priors,Ks)]
    world = World(params,genomes)

    fitness,history = world.run()
    return {'fitness':fitness,
            'history':history}

@plotter(binary_matchup)
def joint_fitness_plot(player_types,priors,data = []):
    agents = []
    for n,(t,p) in enumerate(zip(player_types,priors)):
        if issubclass(t,RationalAgent):
            if player_types[0]==player_types[1] and priors[0]==priors[1]:
                agents.append("%s(prior=%s) #%s"%(str(t),p,n))
            else:
                agents.append("%s(prior=%s)"%(str(t),p))
        else:
            if player_types[0]==player_types[1]:
                agents.append("%s #%s" % (str(t),n))
            else:
                agents.append(str(t))

    record = []
    for rec in data.to_dict('record'):
        record.append(dict(zip(agents,rec['fitness'])))
    data = pd.DataFrame(record)
    bw = .5
    sns.jointplot(agents[0], agents[1], data, kind = 'kde',bw = bw,marginal_kws = {"bw":bw})

def unordered_prior_combinations(prior_list = np.linspace(.75,.25,3)):
    return map(tuple,map(sorted,combinations(prior_list,2)))

def comparison_grid(rational_type, priors = np.linspace(0,1,5),**kwargs):
    player_types = (rational_type,rational_type)
    priors = MultiArg(unordered_prior_combinations(priors))
    condition = dict(kwargs,**locals())
    del condition['kwargs']

    return binary_matchup(**condition)

@plotter(comparison_grid)
def compare_plot(data = []):
    record = []
    for priors,group in data.groupby('priors'):
        p0,p1 = priors
        for r0,r1 in group['fitness']:
            record.append({'recipient prior':p0, 'opponent prior':p1, 'reward':r0})
            record.append({'recipient prior':p1, 'opponent prior':p0, 'reward':r1})
    data = pd.DataFrame(record)
    
    meaned = data.groupby(['recipient prior','opponent prior']).mean().unstack()
    sns.heatmap(meaned,annot=True,fmt="0.2f")

priors_for_RAvRA = map(tuple,map(sorted,combinations(np.linspace(.75,.25,3),2)))
diagonal_priors = [(n,n) for n in np.linspace(.75,.25,3)]


from games import RepeatedSequentialBinary
from indirect_reciprocity import NiceReciprocalAgent

def history_maker(observations,agents,start=0,annotation = {}):
    history = []
    for r,observation in enumerate(observations,start):
        [agent.observe(observation) for agent in agents]
        history.append(dict({
            'round':r,
            'players':deepcopy(agents)},**annotation))
    return history

letter_to_id = dict(map(reversed,enumerate("ABCDEFGHIJK")))
letter_to_action = {"C":'give',"D":'keep'}
@multi_call()
@experiment(unpack = 'record', unordered = ['agent_types'])
def scenarios(RA_K = 1, agent_types = (ReciprocalAgent,SelfishAgent,AltruisticAgent),**kwargs):
    condition = dict(locals(),**kwargs)
    genome = default_genome(**condition)
    game = BinaryDictator()
    def vs(players,action,observers = "ABO"):
        players = [letter_to_id[p] for p in players]
        observers = [letter_to_id.get(p,p) for p in observers]
        print observers
        action = letter_to_action[action]
        return [(game,players,observers,action)]

    scenarios = ["C","D","CD","CC","DD","DC"]
    scenario_dict = {}
    for actions in scenarios:
        scenario_dict[actions] = []
        for action,players in reversed(zip(reversed(actions),["AB","BA"])):
            scenario_dict[actions].append(vs(players,action))

    record = []
    for name, observations in scenario_dict.iteritems():
        observer = RationalAgent(genome = genome ,world_id = "O")
        for observation in observations:
            observer.observe(observation)
        for agent_type in agent_types:
            record.append({
                'scenario':name,
                'belief':observer.belief_that(0,agent_type),
                'type':justcaps(agent_type),
            })
    return record
@plotter(scenarios)
def scene_plot(agent_types = (ReciprocalAgent,SelfishAgent,AltruisticAgent), RA_K = MultiArg([0,1]), data = []):
    sns.factorplot(data = data, x = "RA_K", y = 'belief', col = 'scenario', kind = 'bar', hue = 'type', hue_order = ["RA","AA","SA"])

@multi_call()
@experiment(unordered = ['agent_types'])
def forgiveness(player_types,RA_Ks,RA_priors,defections,**kwargs):
    condition = dict(locals(),**kwargs)

    params = default_params(**condition)
    game = params['games']

    genome_args = doubling_zip(player_types,RA_Ks,RA_priors)
    genomes = [default_genome(agent_type = agent_type, RA_K = RA_K, RA_prior = RA_prior,**condition)
               for agent_type, RA_K, RA_prior in genome_args]

    world = World(params,genomes)
    agents = world.agents

    observations = [[(game,[0,1],[0,1],'keep')]]*defections
    prehistory = history_maker(observations,agents,start = -defections)

    fitness, history = world.run()
    history = prehistory+history

    rec = []
    for event in history:
        for a_id, agent in enumerate(event['players']):
            rec.append({
                'K':genome_args[a_id][1],
                'type':genome_args[a_id][0],
                'round':event['round'],
            })


id_to_letter = dict(enumerate("ABCDEF"))
@plotter(binary_matchup)
def belief_plot(player_types,priors,Ks,believed_type=NiceReciprocalAgent,data=[]):
    K = max(Ks)
    t_ids = [[list(islice(cycle(order),0,k)) for k in range(1,K+2)] for order in [(1,0),(0,1)]]
    
    record = []
    for d in data.to_dict('record'):
        for event in d['history']:
            for a_id, believer in enumerate(event['players']):
                for ids in t_ids[a_id]:
                    k = K-len(ids)+1
                    record.append({
                        "believer":a_id,
                        "k":k,
                        "belief":believer.k_belief(ids,believed_type),
                        "target_id":ids[-1],
                        "round":event['round'],
                        "type":justcaps(believed_type),
                    })
    bdata = pd.DataFrame(record)
    f_grid = sns.factorplot(data = bdata, x = 'round', y = 'belief', row = 'k', col = 'believer', kind = 'violin', hue = 'type',
                   facet_kws = {'ylim':(0,1)})
    f_grid.map(sns.pointplot,'round','belief')
    for a_id,k in product([0,1],range(K+1)):
        ids = t_ids[a_id][k]
        axis = f_grid.facet_axis(k,a_id)
        axis.set(#xlabel='# of interactions',
            ylabel = '$\mathrm{Pr_{%s}( T_{%s} = RA | O_{1:n} )}$'% (k,id_to_letter[ids[-1]]),
            title = ''.join([id_to_letter[l] for l in [a_id]+ids]))
        
letter_2_index = dict(map(reversed,enumerate('ABCDEFG')))
@multi_call()
@experiment(unordered = ['agent_types'], unpack = 'record', memoize = False)
def first_impressions(max_cooperations, agent_types, RA_prior, **kwargs):
    condition = dict(locals(),**kwargs)
    observer_genome = default_genome(**condition)
    observer = RationalAgent(genome = default_genome(**condition), world_id = 'O')
    game = BinaryDictator()

    def vs(action):
        return [(game,[0,1],[0,1,'O'],action)]

    record = []
    for cooperations in range(max_cooperations+1):
        observer = RationalAgent(genome = observer_genome, world_id = 'O')
        observations = [vs('give')*cooperations,
                        vs('keep')]
        for observation in observations:
            observer.observe(observation)
        for agent_type in agent_types:
            record.append({'cooperations': cooperations,
                           'belief': observer.belief_that(0,agent_type),
                           'type': justcaps(agent_type)})

    plt.subplots_adjust(top = 0.93)
    figure.fig.suptitle("A and B's beliefs that the other is RA when A's first 3 moves are D")
    return record

@plotter(first_impressions)
def first_impressions_plot(max_cooperations = 5, agent_types = (NiceReciprocalAgent,AltruisticAgent,SelfishAgent),
                           RA_prior =.75, data = []):
    fplot = sns.factorplot(data = data, x='cooperations', y='belief', col='RA_prior', bw = .1,
                           hue = 'type', hue_order = map(justcaps,agent_types),
                           facet_kws = {'ylim':(0,1)})
    #fplot.set(yticklabels = np.linspace(0,1,5))

#first_impressions_plot()#RA_prior = MultiArg([.25,.5,.75]))
#binary_matchup(player_types = ReciprocalAgent)

#compare_plot(rational_type = ReciprocalAgent, Ks = 0,  agent_types = (ReciprocalAgent,SelfishAgent), trials = 1000)
#joint_fitness_plot(player_types = ReciprocalAgent, priors = .25, Ks =(0,0), agent_types = (ReciprocalAgent,SelfishAgent),trials = 1000)


#belief_plot(priors = (.25,0),Ks = 0)
#belief_plot(priors = (.75,0),Ks = 0)
#belief_plot(priors = (.75,.25))

scene_plot()
