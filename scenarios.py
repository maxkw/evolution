from __future__ import division
from collections import defaultdict
import seaborn as sns
from experiment_utils import multi_call, experiment, plotter, MultiArg, cplotter, memoize, apply_to_args
import numpy as np
from params import default_params, generate_proportional_genomes, default_genome
from indirect_reciprocity import World, ReciprocalAgent, SelfishAgent, AltruisticAgent, NiceReciprocalAgent, RationalAgent, gTFT, AllC, AllD, Pavlov
from games import BinaryDictator
import matplotlib.pyplot as plt
from utils import softmax_utility, justcaps

# letter_to_id = dict(map(reversed, enumerate("ABCDEFGHIJK")))
letter_to_action = {"C": 'give', "D": 'keep'}

@multi_call()
@experiment(unpack='record', unordered=['agent_types'], memoize=False)
def scenarios(agent_types, **kwargs):
    condition = dict(locals(), **kwargs)
    genome = default_genome(agent_type=RationalAgent, **condition)
    game = BinaryDictator()

    scenarios = [
        [["AB"], "C"], 
        [["AB"], "D"],
        # [["BA", "AB"], "CD"],
        # [["BA", "AB"], "CC"],
        [["BA", "AB"], "DD"],
        [["BA", "AB"], "DC"],
        [["AB", "BA", "AB"], "CDC"],
        # [["AB", "BA", "AB"], "CCD"],
    ]
    observers = "ABO"
    scenario_dict = defaultdict(list)
    
    for scenario in scenarios:
        for players, action in zip(*scenario):
            scenario_dict[scenario[1]].append(
                [(game, players, observers, letter_to_action[action])]
            )
    
    record = []
    for name, observations in scenario_dict.iteritems():
        observer = RationalAgent(genome=genome, world_id="O")
        for observation in observations:
            observer.observe(observation)

        for agent_type in agent_types:
            record.append({
                'scenario': name,
                'belief': observer.belief_that('A', agent_type),
                'type': justcaps(agent_type),
            })
    return record

@plotter(scenarios, plot_exclusive_args=['data'])
def scene_plot(agent_types, data = []):
    sns.set_context("poster", font_scale=1.5)
    f_grid = sns.factorplot(data=data, x="type", y='belief', col='scenario', row="RA_K", kind='bar',
                            order = ["RA","AA","SA"],
                            # order = ["RA","AC","AD"],
                            # col_order = ["C","D","DD","DC","CD","CC"],
                            aspect=1.5,
                            facet_kws={'ylim': (0, 1),
                                       'margin_titles': True})

    def draw_prior(data, **kwargs):
        plt.axhline(data.mean(), linestyle=":")
        
    f_grid.map(draw_prior, 'RA_prior')
    f_grid.set_xlabels("")
    f_grid.set(yticks=np.linspace(0, 1, 5))
    f_grid.set_yticklabels(['', '0.25', '0.50', '0.75', '1.0'])
    # f_grid.despine(bottom=True)

if __name__ == '__main__':
    scene_plot(
        agent_types=(ReciprocalAgent, SelfishAgent, AltruisticAgent),
        RA_prior = .75,
        RA_K = MultiArg([0, 1, 2]),
        beta = 5)

