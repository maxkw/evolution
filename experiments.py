from __future__ import division
import pandas as pd
import seaborn as sns
from experiment_utils import multi_call,experiment,plotter,MultiArg,cplotter
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
from utils import softmax_utility


###multi_call

def justcaps(t):
    return filter(str.isupper,t.__name__)

@multi_call(unordered = ['agent_types'], twinned = ['player_types','priors','Ks'], verbose=3)
@experiment(unpack = 'dict', trials = 100, verbose = 3)
def binary_matchup(player_types = (NiceReciprocalAgent,NiceReciprocalAgent), priors = (.75, .75), Ks=(1,1), agent_types = (NiceReciprocalAgent,SelfishAgent), **kwargs):
    condition = dict(locals(),**kwargs)
    params = default_params(**condition)
    genomes = [default_genome(agent_type = t, RA_prior=p ,RA_K = k, **condition) for t,p,k in zip(player_types,priors,Ks)]
    world = World(params,genomes)

    fitness,history = world.run()
    return {'fitness':fitness,
            'history':history}

id_to_letter = dict(enumerate("ABCDEF"))
@cplotter(binary_matchup,plot_args = ['data','believed_type'])
def belief_plot(player_types,priors,Ks,believed_type=NiceReciprocalAgent,data=[]):
    K = max(Ks)
    t_ids = [[list(islice(cycle(order),0,k)) for k in range(1,K+2)] for order in [(1,0),(0,1)]]

    record = []
    for d in data.to_dict('record'):
        for event in d['history']:
            for a_id, believer in enumerate(event['players']):
                a_id = believer.world_id
                for ids in t_ids[a_id]:
                    k = len(ids)-1
                    record.append({
                        "believer":a_id,
                        "k":k,
                        "belief":believer.k_belief(ids,believed_type),
                        "target_id":ids[-1],
                        "round":event['round'],
                        "type":justcaps(believed_type),
                    })
    bdata = pd.DataFrame(record)
    f_grid = sns.factorplot(data = bdata, x = 'round', y = 'belief', row = 'k', col = 'believer', kind = 'violin', hue = 'type', row_order = range(K+1),
                   facet_kws = {'ylim':(0,1)})
    f_grid.map(sns.pointplot,'round','belief')
    for a_id,k in product([0,1],range(K+1)):
        ids = t_ids[a_id][k]
        axis = f_grid.facet_axis(k,a_id)
        axis.set(#xlabel='# of interactions',
            ylabel = '$\mathrm{Pr_{%s}( T_{%s} = RA | O_{1:n} )}$'% (k,id_to_letter[ids[-1]]),
            title = ''.join([id_to_letter[l] for l in [a_id]+ids]))

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
#@apply_to_args(twinning = ['player_types'])
def comparison_grid(player_types, priors = np.linspace(0,1,5),**kwargs):
    priors = MultiArg(unordered_prior_combinations(priors))
    condition = dict(kwargs,**locals())
    del condition['kwargs']

    return binary_matchup(**condition)

@plotter(comparison_grid)
def reward_table(data = []):
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
def scenarios(RA_K = 1, agent_types = (NiceReciprocalAgent,SelfishAgent,AltruisticAgent),**kwargs):
    condition = dict(locals(),**kwargs)
    genome = default_genome(**condition)
    game = BinaryDictator()
    def vs(players,action,observers = "ABO"):
        players = [letter_to_id[p] for p in players]
        observers = [letter_to_id.get(p,p) for p in observers]
        action = letter_to_action[action]
        return [(game,players,observers,action)]

    scenarios = ["C","D","CD","CC","DD","DC"]
    scenario_dict = {}
    for actions in scenarios:
        scenario_dict[actions] = []
        for action,players,times  in reversed(zip(reversed(actions),["AB","BA"],[1,1])):
            scenario_dict[actions].append(vs(players,action)*times)

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

@cplotter(scenarios,plot_args = ['data'])
def scene_plot(agent_types, RA_prior =.75, RA_K = MultiArg([0,1]), data = []):
    sns.set_context("poster",font_scale = 1.5)
    f_grid = sns.factorplot(data = data, x = "RA_K", y = 'belief', col = 'scenario', row = "RA_prior", kind = 'bar', hue = 'type', hue_order = ["NRA","AA","SA"],col_order = ["C","D","DD","DC","CD","CC"],
                            aspect = 1.5,
                            facet_kws = {'ylim': (0,1),
                                         'margin_titles':True})
    def draw_prior(data,**kwargs):
        plt.axhline(data.mean(),linestyle = ":")
    f_grid.map(draw_prior,'RA_prior')
    f_grid.set_xlabels("")
    f_grid.set(yticks=np.linspace(0,1,5))
    f_grid.set_yticklabels(['','0.25','0.50','0.75','1.0'])
    #f_grid.despine(bottom=True)

@multi_call()
@experiment(unordered = ['agent_types'],)
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
    return rec

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
    return record

@cplotter(first_impressions)
def first_impressions_plot(max_cooperations = 5, agent_types = (NiceReciprocalAgent,AltruisticAgent,SelfishAgent),
                           RA_prior =.75, data = None):
    fplot = sns.factorplot(data = data, x='cooperations', y='belief', col='RA_prior', bw = .1,
                           hue = 'type', hue_order = map(justcaps,agent_types),
                           facet_kws = {'ylim':(0,1)})
    fplot.fig.suptitle("A and B's beliefs that the other is RA")

@multi_call(unordered = ['agent_types'], verbose = 3)
@experiment(trials = 100,unpack = 'dict', verbose = 3)
def pop_matchup(player_types = (ReciprocalAgent,SelfishAgent), pop_size = 50, proportion = .5, agent_types = (SelfishAgent,ReciprocalAgent), **kwargs):
    condition = dict(locals(),**kwargs)
    proportions = dict(zip(player_types,[proportion,1-proportion]))
    genomes = generate_proportional_genomes(agent_proportions = proportions, **condition)
    params = default_params(**condition)
    world = World(params,genomes)

    pop_types = [g['type'] for g in genomes]
    fitness, history = world.run()

    return {'type_fitness_pairs':zip(pop_types,fitness)}
    #fitnesses = defaultdict(int)
    #for t,f in zip(pop_types, fitness):
    #    fitnesses[t] += f

    #for t in fitnesses:
    #    fitnesses[t] = fitnesses[t]/pop_types.count(t)

    #fitnesses = softmax_utility(fitnesses,.1)

    #return {'fitness ratio':fitnesses[player_types[0]]}

def pop_fitness_ratios(player_types=(ReciprocalAgent,SelfishAgent),pop_size=50,proportion=.5,**kwargs):
    condition = dict(locals(),**kwargs)
    del condition['kwargs']
    data = pop_matchup(**condition)
    record = []
    for r in data.to_dict('record'):
        fitness = r['type_fitness_pairs']
        types,fits = zip(*fitness)
        fitnesses = defaultdict(int)
        for t,f in fitness:
            fitnesses[t] += f

        for t in fitnesses:
            fitnesses[t] = fitnesses[t]/types.count(t)

        fitnesses = softmax_utility(fitnesses,.1)

        r['fitness ratio']= fitnesses[player_types[0]]

        record.append(r)

    return pd.DataFrame(record)
@cplotter(pop_fitness_ratios, plot_args = ['data'])
def pop_fitness_plot(player_types, proportion = MultiArg([.25,.5,.75]), RA_K = MultiArg([0,1]), data = None):
    #print data
    #ndata = data.groupby(['RA_K','proportion']).mean().unstack()
    #print ndata
    
    sns.pointplot(data = data, x = "proportion", y = "fitness ratio", hue = "RA_K")
    #print locals()
    #fplot.set(yticklabels = np.linspace(0,1,5))

#first_impressions_plot()#RA_prior = MultiArg([.25,.5,.75]))
#binary_matchup(player_types = ReciprocalAgent)

#compare_plot(rational_type = ReciprocalAgent, Ks = 0,  agent_types = (ReciprocalAgent,SelfishAgent), trials = 1000)
#joint_fitness_plot(player_types = ReciprocalAgent, priors = .25, Ks =(0,0), agent_types = (ReciprocalAgent,SelfishAgent),trials = 1000)


#belief_plot(priors = (.25,0),Ks = 0)
#belief_plot(priors = (.75,0),Ks = 0)
#belief_plot(priors = (.8, 0),Ks = 0)
#for k in range(3):
#    belief_plot(priors = (.8,0), Ks = k)
#belief_plot(priors = (.75,.25))
#compare_plot(rational_type = ReciprocalAgent, Ks = 1,beta = 1,trials = 50)

#scene_plot(beta = 4)

for n in range(1,20):
    """
    this outputs a plot for every 10 trials, progressively refining the quality of the plot
    """
    ticks = 9
    props = np.round(np.linspace(0,1,ticks+2)[1:-1],2)
    pop_fitness_plot(
        proportion = MultiArg(props),
        player_types = (ReciprocalAgent,SelfishAgent),
        agent_types = (ReciprocalAgent,SelfishAgent),
        trials = 10*n, pop_size = 10, RA_prior = .8, beta=1)

#belief_plot(priors = (), )

#reward_table(player_types = ReciprocalAgent, Ks = 1, agent_types = (ReciprocalAgent,SelfishAgent), beta = 3)
#reward_table(player_types = ReciprocalAgent, Ks = 0, agent_types = (ReciprocalAgent,SelfishAgent), beta = 1)

