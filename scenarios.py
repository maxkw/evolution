from __future__ import division
from collections import defaultdict, OrderedDict
import seaborn as sns
from experiment_utils import multi_call, experiment, plotter, MultiArg, cplotter, memoize, apply_to_args
import numpy as np
from params import default_params, generate_proportional_genomes, default_genome
from indirect_reciprocity import World, ReciprocalAgent, SelfishAgent, AltruisticAgent, NiceReciprocalAgent, RationalAgent, gTFT, AllC, AllD, Pavlov
from games import BinaryDictator
import matplotlib.pyplot as plt
from utils import softmax_utility, justcaps


letter_to_action = {"C": 'give', "D": 'keep'}
agent_map = {ReciprocalAgent: 'Reciprocal',
             AltruisticAgent: 'Altruistic',
             SelfishAgent: 'Selfish'
}

@multi_call()
@experiment(unpack='record', unordered=['agent_types'], memoize=False)
def scenarios(scenario_func, agent_types, **kwargs):
    condition = dict(locals(), **kwargs)
    genome = default_genome(agent_type=RationalAgent, **condition)
    game = BinaryDictator()

    scenario_dict = scenario_func()
    
    record = []
    for name, observations in scenario_dict.iteritems():
        observer = RationalAgent(genome=genome, world_id="O")
        for observation in observations:
            observer.observe(observation)

        for agent_type in agent_types:
            record.append({
                'scenario': name,
                'belief': observer.belief_that('A', agent_type),
                'type': agent_map[agent_type],
            })
    return record

@plotter(scenarios, plot_exclusive_args=['data'])
def scene_plot(agent_types, data = []):
    sns.set_context("poster", font_scale=1.5)
    if AltruisticAgent in agent_types:
        order = ["Reciprocal","Selfish","Altruistic"]
    else:
        order = ["Reciprocal", "Selfish"]

        
    f_grid = sns.factorplot(data=data, y="type", x='belief', col='scenario', row="RA_K", kind='bar', orient='h', 
                            order = order,
                            aspect=2,
                            size=3,
                            sharex=False, sharey=False
    )

    # def draw_prior(data, **kwargs):
        # plt.axhline(data.mean(), linestyle=":")
        
    (f_grid
     # .map(draw_prior, 'RA_prior')
     .set_xlabels("P(A = type)")
     .set_titles("")
     .set_ylabels("")
     .set(xlim = (0,1),
          xticks=np.linspace(0, 1, 5),
          xticklabels=['0', '', '0.5', '', '1'])
    )
    # plt.tight_layout()

def make_dict_from_scenarios(scenarios, observers, scenario_dict=None):
    if scenario_dict is None:
       scenario_dict = OrderedDict()

    game = BinaryDictator()
    for scenario in scenarios:
        scenario_dict[scenario[1]] = list()
        for players, action in zip(*scenario):
            scenario_dict[scenario[1]].append(
                [(game, players, observers, letter_to_action[action])]
            )

    return scenario_dict
       
def reciprocal_scenarios_0():
    scenarios = [
        [["AB"], "C"], 
        [["AB"], "D"],
    ]

    scenario_dict = OrderedDict()
    # Make the prior a scenario with no observations
    scenario_dict['prior'] = list()
    scenario_dict = make_dict_from_scenarios(scenarios, "ABO", scenario_dict)

    return scenario_dict

def reciprocal_scenarios_1():
    scenarios = [
        [["BA", "AB"], "DD"],
        [["BA", "AB"], "DC"],
        [["BA", "AB"], "CD"],
        # [["AB", "BA", "AB"], "CDC"],
    ]

    scenario_dict = OrderedDict()
    scenario_dict = make_dict_from_scenarios(scenarios, "ABO", scenario_dict)

    return scenario_dict


def false_belief_scenarios():
    game = BinaryDictator()
    scenario_dict = OrderedDict()

    # Make a false belief scenario
    scenario_dict['false'] = [
        [(game, "CB", "BO", letter_to_action['D'])],
        [(game, "BC", "ABCO", letter_to_action['D'])],
        [(game, "AB", "O", letter_to_action['D'])]
    ]

    # Make a true belief scenario
    scenario_dict['true'] = [
        [(game, "CB", "ABCO", letter_to_action['D'])],
        [(game, "BC", "ABCO", letter_to_action['D'])],
        [(game, "AB", "O", letter_to_action['D'])]
    ]

    return scenario_dict

def main():
    scene_plot(
        scenario_func = reciprocal_scenarios_0,
        agent_types=(ReciprocalAgent, SelfishAgent, AltruisticAgent),
        RA_prior = .75,
        RA_K = MultiArg([0, 1]),
        beta = 5,
        file_name = 'scene_reciprocal_0')

    scene_plot(
        scenario_func = reciprocal_scenarios_1,
        agent_types=(ReciprocalAgent, SelfishAgent, AltruisticAgent),
        RA_prior = .75,
        RA_K = MultiArg([0, 1]),
        beta = 5,
        file_name = 'scene_reciprocal_1')
    
    scene_plot(
        scenario_func = false_belief_scenarios,
        agent_types=(ReciprocalAgent, SelfishAgent),
        RA_prior = .75,
        RA_K = MultiArg([0, 1, 2]),
        beta = 5,
        file_name = 'scene_false_belief')

if __name__ == '__main__':
    main()
