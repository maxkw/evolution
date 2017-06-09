from games import IndirectReciprocity
from world import World
from agents import WeAgent, AllD, ReciprocalAgent
from collections import defaultdict
from params import default_params,default_genome
from experiment_utils import experiment, plotter, multi_call, MultiArg
import numpy as np
import seaborn as sns
import pandas as pd
from utils import softmax_utility




@multi_call(unordered = ['agent_types'], verbose = 3)
def indirect_game_ratios(WA_ratio = .5, pop_size = 30, *args, **kwargs):

    n = pop_size
    c = int(pop_size*WA_ratio)
    #rounds = proportion_of_matches
    type_count= dict([(WeAgent,c), (AllD,n-c)])
    return indirect_game(type_count,*args,**kwargs)

@multi_call(unordered = ['agent_types'], verbose = 3)
@experiment(unpack = 'record', trials = 100,verbose =3)
def indirect_game(type_count_pairs, observability = 1, rounds = 50, tremble= 0, **kwargs):
    types = []
    type_to_count = dict(type_count_pairs)
    for t,count in type_to_count.iteritems():
        types += [t]*count

    params = default_params(**kwargs)
    params.update(kwargs)
    params['games'] = g = IndirectReciprocity(rounds, observability,tremble = tremble, **kwargs)
    genomes = [default_genome(agent_type = t, **params) for t in types]

    world = World(params,genomes)

    fitness, history = world.run()

    fitness_per_type = defaultdict(int)
    for t,f in zip(types,fitness):
        fitness_per_type[t]+=f

    for t in type_to_count:
        try:
            fitness_per_type[t] /= type_to_count[t]
        except ZeroDivisionError:
            assert fitness_per_type[t] == 0
    records =[]
    for t,f in fitness_per_type.iteritems():
        records.append({"type":t,
                        "fitness":f})
    #fitness_ratio = fitness_per_type[WeAgent]/fitness_per_type[AllD]
    #return [{"fitness_ratio":fitness_ratio}]
    return records

def indirect_simulator(types, *args,**kwargs):
    data = indirect_game(*args,**kwargs)
    fitness = {}
    for t,d in data.groupby('type'):
        fitness[t] = d.mean()['fitness']
    return [fitness[t] for t in types]


@plotter(indirect_game_ratios, plot_exclusive_args = ['data'])
def relative_fitness_vs_proportion(data = [], *args, **kwargs):
    print data[data['observability']==.5]

    #records = []
    #for (WA_ratio,observability), df in data.groupby(['WA_ratio','observability']):
    #    fitness_ratio = df[df['type'] == WeAgent].mean()['fitness']/df[df['type'] == AllD].mean()['fitness']
    #    records.append({"WA_ratio": WA_ratio,
    #                    "observability": observability,
    #                    "fitness_ratio": fitness_ratio})
    #data = pd.DataFrame(records)
    data = data[data['type'] == WeAgent]
    g = sns.factorplot(data = data, x = 'WA_ratio', y = 'fitness', hue = 'observability', kind = 'point')

    g.set(ylim = (0,1))

#for t in [n*10 for n in range(1,5)]:
#    relative_fitness_vs_proportion(WA_ratio = MultiArg(round(n,2) for n in np.linspace(0,1,7)[1:-1]), pop_size = 20, observability = MultiArg([0,.5,.75,1]), rounds = 100, tremble = 0, RA_prior = .5, trials = t, beta = 5)


