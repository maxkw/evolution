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
@experiment(unpack = 'record', trials = 100,verbose =3)
def indirect_game(WA_ratio = .5, pop_size = 30, observability = 1, rounds = 50, tremble= 0, **kwargs):
    #print WA_ratio
    n = pop_size
    c = int(pop_size*WA_ratio)
    #rounds = proportion_of_matches
    type_count= dict([(WeAgent,c), (AllD,n-c)])


    types = []
    for t,count in type_count.iteritems():
        types += [t]*count

    params = default_params(agent_types = ('self',AllD), **kwargs)
    params.update(kwargs)
    params['games'] = g = IndirectReciprocity(rounds, observability,tremble = tremble)
    genomes = [default_genome(agent_type = t, **params) for t in types]

    world = World(params,genomes)

    fitness, history = world.run()

    fitness_per_type = defaultdict(int)
    for t,f in zip(types,fitness):
        fitness_per_type[t]+=f

    for t in type_count:
        fitness_per_type[t] /= type_count[t]

    fitness_per_type = softmax_utility(fitness_per_type, beta = 1)

    records =[]
    for t,f in fitness_per_type.iteritems():
        records.append({"type":t,
                        "fitness":f})
    #fitness_ratio = fitness_per_type[WeAgent]/fitness_per_type[AllD]
    #return [{"fitness_ratio":fitness_ratio}]
    return records


@plotter(indirect_game, plot_exclusive_args = ['data'])
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

for t in [n*10 for n in range(1,5)]:
    relative_fitness_vs_proportion(WA_ratio = MultiArg(round(n,2) for n in np.linspace(0,1,7)[1:-1]), pop_size = 20, observability = MultiArg([0,.5,.75,1]), rounds = 100, tremble = 0, RA_prior = .5, trials = t, beta = 5)


