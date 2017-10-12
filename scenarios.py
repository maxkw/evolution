from __future__ import division
import scipy as sp
from collections import defaultdict, OrderedDict, Counter
import seaborn as sns
from experiment_utils import multi_call, experiment, plotter, MultiArg
import numpy as np
from params import default_params, default_genome
from agents import ReciprocalAgent, SelfishAgent, AltruisticAgent, RationalAgent, gTFT, AllC, AllD, Pavlov, WeAgent, RandomAgent
from games import BinaryDictator, GradatedBinaryDictator, SocialDictator
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
def forgive(agent_types, **kwargs):
    condition = dict(locals(), **kwargs)
    genome = default_genome(agent_type=WeAgent, **condition)
    game = BinaryDictator()

    record = []

    trials = ['D', 'CD', 'CCD', 'CCDDD']

    for trial in trials:
        observer = WeAgent(genome=genome, world_id='B')
        h = Counter(trial)

        # observations = [
        #     [(game, 'AB', 'ABO', 'give'),
        #      (game, 'BA', 'ABO', 'give')
        #     ]] * h['C']

        # for observation in observations:
        #     observer.observe(observation)

        for _ in xrange(h['C']):
            observer.observe([(game, 'AB', 'ABO', 'give'),
                              (game, 'BA', 'ABO', 'give')])

        before = observer.belief_that('A', WeAgent)
        
        for _ in xrange(h['D']):
            observer.observe([(game, 'AB', 'ABO', 'keep')])
        
        after = observer.decide_likelihood(game, 'BA', )[0]

        # after = observer.belief_that('A', WeAgent)
        # after = after - before
        
        # for t, b in zip(['Before', 'After'], [before, after]):
        for t, b in zip(['After'], [after]):
            record.append({'cooperations': trial,
                           'belief': b,
                           'type': t})

        print record
    return record




@multi_call()
@experiment(unordered=['agent_types'], unpack='record', memoize=False)
def first_impressions(agent_types, **kwargs):
    condition = dict(locals(), **kwargs)
    genome = default_genome(agent_type=WeAgent, **condition)
    game = BinaryDictator()

    record = []

    trials = ['', 'D', 'CD', 'CCD', 'CCDDD']

    for trial in trials:
        observer = WeAgent(genome=genome, world_id='B')
        h = Counter(trial)

        # observations = [
        #     [(game, 'AB', 'ABO', 'give'),
        #      (game, 'BA', 'ABO', 'give')
        #     ]] * h['C']

        # for observation in observations:
        #     observer.observe(observation)

        for _ in xrange(h['C']):
            observer.observe([(game, 'AB', 'ABO', 'give'),
                              (game, 'BA', 'ABO', 'give')])

        before = observer.belief_that('A', WeAgent)
        
        for _ in xrange(h['D']):
            observer.observe([(game, 'AB', 'ABO', 'keep')])
        
        # after = observer.decide_likelihood(game, 'BA', )[0]

        after = observer.belief_that('A', WeAgent)
        # after = after - before
        
        # for t, b in zip(['Before', 'After'], [before, after]):
        for t, b in zip(['After'], [after]):
            record.append({'cooperations': trial,
                           'belief': b,
                           'type': t})

        print record
    return record

@plotter(first_impressions)
def forgive_plot(data=[]):
    sns.set_context("poster", font_scale=1)
    fplot = sns.factorplot(data=data, x='cooperations',
                           y='belief',
                           # bw=.1,
                           kind = 'bar',
                           hue='type', legend=False, aspect=1.3)

    fplot.axes[0][0].axhline(.33, 0, 2, color='black')
    fplot.set(
         # ylim=(0, 1.05),
          # ylabel='P(A = Reciprocal)',
          ylabel='P(Cooperate)',
          xlabel='',
          # yticks=np.linspace(0, 1, 5),
          # yticklabels=['0', '0.25', '0.5', '0.75', '1']
    )
    # plt.legend(loc='best')
    plt.tight_layout()


@plotter(first_impressions)
def first_impressions_plot(data=[]):
    sns.set_context("poster", font_scale=1)
    fplot = sns.factorplot(data=data, x='cooperations',
                           y='belief',
                           # bw=.1,
                           kind = 'bar',
                           hue='type', legend=False, aspect=1.3)

    fplot.set(
         # ylim=(0, 1.05),
          ylabel='P(A = Reciprocal)',
          # ylabel='',
          xlabel='',
          # yticks=np.linspace(0, 1, 5),
          # yticklabels=['0', '0.25', '0.5', '0.75', '1']
    )
    # plt.legend(loc='best')
    plt.tight_layout()

@multi_call()
@experiment(unordered=['agent_types'], unpack='record', memoize=False)
def n_action_info(agent_types, **kwargs):
    condition = dict(locals(), **kwargs)
    genome = default_genome(agent_type=WeAgent, **condition)

    record = []
    for args in [
            dict(cost = 5, benefit = 15, tremble = 0),
    ]:
        
        for i in [2, 4, 8]:
            
            for game in [
                    SocialDictator(intervals = i, **args),
                    # GradatedBinaryDictator(intervals = i, **args)
            ]:
                for t in agent_types:
                    E_KL = 0
                    if t == 'self':
                        actor = WeAgent(genome=genome, world_id='O')
                    else:
                        actor = t(genome=genome)
                        # continue 

                    for a_id, likelihood in enumerate(actor.decide_likelihood(game, 'AB')):
                        observer = WeAgent(genome=genome, world_id='O')
                        q = observer.belief['A']
                        observer.observe([(game, 'AB', 'ABO', game.actions[a_id])])
                        p = observer.belief['A']
                        E_KL += likelihood * sp.stats.entropy(p, q, base = 2)
                        
                    # import pdb; pdb.set_trace()
                    
                    record.append({
                        '# Actions': int(i),
                        'bits': E_KL,
                        # 'args': 'Tremble=%0.2f C=%d B=%d' % (args['tremble'], args['cost'], args['benefit']),
                        'args': 'C=%d B=%d' % (args['cost'], args['benefit']),
                        # 'args': '%0.2f' % (args['tremble']),
                        'game':  'Social' in str(game.name),
                        'actor': str(actor.__class__)
                    })

    return record

@plotter(n_action_info)
def n_action_plot(data=[]):
    data['# Actions'] = data['# Actions'].astype(int)
    sns.set_context("poster", font_scale=1)
    fplot = sns.factorplot(data=data,
                           x='# Actions',
                           y='bits',
                           # hue = 'args',
                           hue = 'actor',
                           # bw=.1,
                           kind = 'bar',
                           # col = 'actor', 
                           aspect=1.3).set_titles("{col_name}")

    

    fplot.set(
         # ylim=(0, 2),
          ylabel='bits',
          xlabel='# Actions',
    )
    # plt.legend(loc='best')
    # plt.tight_layout()


def main(prior = 0.75, beta = 5, **kwargs):
    first_impressions_plot(
        agent_types=(ReciprocalAgent, SelfishAgent),
        RA_prior=prior,
        beta=beta,
        RA_K=1,
        tremble=0.05,
        file_name='first_impressions', **kwargs)

    scene_plot(
        scenario_func=reciprocal_scenarios_0,
        agent_types=(ReciprocalAgent, SelfishAgent, AltruisticAgent),
        RA_prior=prior,
        RA_K=MultiArg([0, 1]),
        beta=beta,
        file_name='scene_reciprocal_0', **kwargs)

    scene_plot(
        scenario_func=reciprocal_scenarios_1,
        agent_types=(ReciprocalAgent, SelfishAgent, AltruisticAgent),
        RA_prior=prior,
        RA_K=MultiArg([0, 1]),
        beta=beta,
        file_name='scene_reciprocal_1', **kwargs)

    scene_plot(
        scenario_func=false_belief_scenarios,
        agent_types=(ReciprocalAgent, SelfishAgent),
        RA_prior=prior,
        RA_K=MultiArg([0, 1, 2]),
        beta=beta,
        file_name='scene_false_belief', **kwargs)

if __name__ == '__main__':
    
    forgive_plot(agent_types = (SelfishAgent, AltruisticAgent, 'self'))
    first_impressions_plot(agent_types = (SelfishAgent, AltruisticAgent, 'self'))
    
    # n_action_plot(
    #     agent_types= ('self', SelfishAgent, AltruisticAgent),
    #     beta = 5,
    #     RA_prior = 0.5
    # )
    # main()
