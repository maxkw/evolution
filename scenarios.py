from __future__ import division
from collections import defaultdict, OrderedDict, Counter
import seaborn as sns
from experiment_utils import multi_call, experiment, plotter, MultiArg
import numpy as np
from params import default_params, default_genome
from indirect_reciprocity import ReciprocalAgent, SelfishAgent, AltruisticAgent, NiceReciprocalAgent, RationalAgent, gTFT, AllC, AllD, Pavlov
from games import BinaryDictator
import matplotlib.pyplot as plt

agent_to_label = {ReciprocalAgent: 'Reciprocal',
                  AltruisticAgent: 'Altruistic',
                  SelfishAgent: 'Selfish'}

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
                'type': agent_to_label[agent_type],
            })
    return record


@plotter(scenarios, plot_exclusive_args=['data'])
def scene_plot(agent_types, data=[]):
    #print data
    sns.set_context("poster", font_scale=1.5)
    if AltruisticAgent in agent_types:
        order = ["Reciprocal", "Selfish", "Altruistic"]
    else:
        order = ["Reciprocal", "Selfish"]

    f_grid = sns.factorplot(data=data, y="type", x='belief', col='scenario', row="RA_K",
                            kind='bar', orient='h', order=order, aspect=2, size=3, sharex=False, sharey=False)

    # def draw_prior(data, **kwargs):
    # plt.axhline(data.mean(), linestyle=":")
    #f_grid.map(draw_prior, 'RA_prior')

    (f_grid
     .set_xlabels("P(A = type)")
     #.set_titles("")
     .set_ylabels("")
     .set(xlim=(0, 1),
          xticks=np.linspace(0, 1, 5),
          xticklabels=['0', '', '0.5', '', '1'])
     )


def make_dict_from_scenarios(scenarios, observers, scenario_dict=None):
    letter_to_action = {"C": 'give', "D": 'keep'}
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
        [(game, "CB", "BO", 'keep')],
        [(game, "BC", "ABCO", 'keep')],
        [(game, "AB", "O", 'keep')]
    ]

    # Make a true belief scenario
    scenario_dict['true'] = [
        [(game, "CB", "ABCO", 'keep')],
        [(game, "BC", "ABCO", 'keep')],
        [(game, "AB", "O", 'keep')]
    ]

    return scenario_dict


@multi_call()
@experiment(unordered=['agent_types'], unpack='record', memoize=False)
def first_impressions(agent_types, **kwargs):
    condition = dict(locals(), **kwargs)
    genome = default_genome(agent_type=RationalAgent, **condition)
    game = BinaryDictator()

    record = []

    trials = ['D', 'CD', 'CCD', 'CCDD']

    for trial in trials:
        observer = RationalAgent(genome=genome, world_id='B')
        h = Counter(trial)

        observations = [
            [(game, 'AB', 'ABO', 'give'),
             (game, 'BA', 'ABO', 'give')
             ] * h['C']]

        for observation in observations:
            observer.observe(observation)

        before = observer.belief_that('A', ReciprocalAgent)

        observer.observe([(game, 'AB', 'ABO', 'keep')] * h['D'])
        after = observer.belief_that('A', ReciprocalAgent)

        for t, b in zip(['Before', 'After'], [before, after]):
            record.append({'cooperations': trial,
                           'belief': b,
                           'type': t})
    return record


@plotter(first_impressions)
def first_impressions_plot(data=[]):
    sns.set_context("poster", font_scale=1)
    fplot = sns.factorplot(data=data, x='cooperations',
                           y='belief', bw=.1, hue='type', legend=False, aspect=1.3)

    (fplot
     .set(ylim=(0, 1.05),
          ylabel='P(A = Reciprocal)',
          xlabel='',
          yticks=np.linspace(0, 1, 5),
          yticklabels=['0', '0.25', '0.5', '0.75', '1']))
    plt.legend(loc='best')
    plt.tight_layout()


def main():
    scene_plot(
        scenario_func=reciprocal_scenarios_0,
        agent_types=(ReciprocalAgent, SelfishAgent, AltruisticAgent),
        RA_prior=.75,
        RA_K=MultiArg([0, 1]),
        beta=5,
        file_name='scene_reciprocal_0')

    scene_plot(
        scenario_func=reciprocal_scenarios_1,
        agent_types=(ReciprocalAgent, SelfishAgent, AltruisticAgent),
        RA_prior=.75,
        RA_K=MultiArg([0, 1]),
        beta=5,
        file_name='scene_reciprocal_1')

    scene_plot(
        scenario_func=false_belief_scenarios,
        agent_types=(ReciprocalAgent, SelfishAgent),
        RA_prior=.75,
        RA_K=MultiArg([0, 1, 2]),
        beta=5,
        file_name='scene_false_belief')

    first_impressions_plot(
        agent_types=(ReciprocalAgent, SelfishAgent),
        RA_prior=.75,
        beta=5,
        RA_K=1,
        tremble=0.05,
        file_name='first_impressions')


if __name__ == '__main__':
    main()
