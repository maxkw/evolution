
import scipy as sp
from collections import Counter, OrderedDict
import seaborn as sns
from experiment_utils import multi_call, experiment, plotter, MultiArg
import numpy as np
from params import default_genome
from agents import AltruisticAgent, SelfishAgent, WeAgent
from games import BinaryDictator
import matplotlib.pyplot as plt
from itertools import product

def lookup_agent(a):
    a = str(a)
    if 'WeAgent' in a or 'self' in a: return 'Reciprocal'
    if 'AltruisticAgent' in a: return 'Altruistic'
    if 'SelfishAgent' in a: return 'Selfish'
    raise 'String not defined for agent %s'

agent_to_label = {
            # ReciprocalAgent: 'Reciprocal',
            AltruisticAgent: 'Altruistic',
            SelfishAgent: 'Selfish',
            WeAgent: 'Reciprocal',
}

@multi_call()
@experiment(unordered=['agent_types'], memoize=False)
def scenarios(scenarios, agent_types, **kwargs):
    condition = dict(locals(), **kwargs)

    record = []
    
    for name, observations in enumerate(scenarios):
    # for name, observations in scenario_dict.iteritems():
        og = default_genome(agent_type=kwargs['observer'], **kwargs['observer'].genome)
        observer = og['type'](og, "O")

        # observer = WeAgent(genome=default_genome(agent_type = WeAgent(**condition), **condition), world_id="O")
        
        
        for observation in observations:
            observer.observe([observation])

        for agent_type in  observer._type_to_index:
            record.append({
                'player_types': None, 
                'scenario': name,
                'belief': observer.belief_that('A', agent_type),
                'type': lookup_agent(agent_type),
            })
             
    return record

@plotter(scenarios, plot_exclusive_args=['data', 'color', 'graph_kwargs'])
def scene_plot(agent_types, titles=None,  data=[], color = sns.color_palette(['C5', 'C0', 'C1']), graph_kwargs={}):
    sns.set_context("talk", font_scale=.8)

    order = ["Reciprocal", "Altruistic", "Selfish"]
    f_grid = sns.catplot(data=data, y="type", x='belief', col='scenario', 
                            kind='bar', orient='h', order=order,
                            palette=color,
                            aspect=1.2, height=1.8, 
                            sharex=False, sharey=False,
                            **graph_kwargs
    )
    
    if titles is not None  and 'Prior' in titles[0]:
        f_grid.set_xlabels(" ")
    else:
        f_grid.set_xlabels("Belief")
        
        
    (f_grid
     .set_titles("")
     .set_ylabels("")
     .set(xlim=(0, 1),
          xticks=np.linspace(0, 1, 5),
          xticklabels=['0', '', '0.5', '', '1'],
          yticklabels=[])
     )

    if titles is not None:
        for i, (t, ax) in enumerate(zip(titles, f_grid.axes[0])):
            if i > 0:
                ax.set_yticklabels([])
            # ax.set_title(t, fontsize=14, loc='right')
            ax.set_title(t, loc='right')

    plt.tight_layout()

def make_observations_from_scenario(scenario, **kwargs):
    '''
    Scenario is a dict with keys `actions` and `title`
    '''
    letter_to_action = {"C": 'give', "D": 'keep'}
    observations = list()

    game = BinaryDictator(**kwargs)
    for players, action in zip(*scenario):
        observations.append({
            'game': game,
            'action': letter_to_action[action],
            'participant_ids': players,
            'observer_ids': 'ABO',
        })

    return observations

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
@experiment(unordered=['agent_types'], memoize=False)
def first_impressions(agent_types, **kwargs):
    condition = dict(locals(), **kwargs)
    genome = default_genome(agent_type=WeAgent, **condition)
    game = BinaryDictator()

    record = []

    trials = ['Prior', 'D', 'CD', 'CCD', 'CCDDD']

    for trial in trials:
        observer = WeAgent(genome=genome, world_id='B')
        h = Counter(trial)

        for _ in range(h['C']):
            observer.observe([(game, 'AB', 'ABO', 'give'),
                              (game, 'BA', 'ABO', 'give')])

        before = observer.belief_that('A', WeAgent)
        
        for _ in range(h['D']):
            observer.observe([(game, 'AB', 'ABO', 'keep')])
        
        # after = observer.decide_likelihood(game, 'BA', )[0]
        after_b = observer.belief_that('A', WeAgent)
        after_l = observer.decide_likelihood(game, 'BA', )[0]
        # after = after - before
        
        # for t, b in zip(['Before', 'After'], [before, after]):
        for t, b, d in zip(['After'], [after_b], [after_l]):
            record.append({'cooperations': trial,
                           'belief': b,
                           'decision': d,
                           'type': t})

        print(record)
    return record

@plotter(first_impressions)
def forgive_plot(p, data=[], label=''):
    sns.set_context("poster", font_scale=1)
    fplot = sns.factorplot(data=data,
                           y='cooperations',
                           x=p,
                           orient = 'h',
                           # bw=.1,
                           palette={'k'},
                           kind = 'bar',
                           hue='type', legend=False, aspect=1)

    # fplot.axes[0][0].axhline(.33, 0, 2, color='black')
    fplot.set(
        # ylim=(0, 1.05),
        xlabel = label,
        ylabel = '',
        xticklabels=['0', '0.5', '1']
    )

    if p=='decision':
        fplot.set(yticklabels=[])
        
    plt.tight_layout()

# @experiment(unordered=['agent_types'], memoize=False)
# def n_action_info(agent_types, **kwargs):
#     condition = dict(locals(), **kwargs)
#     genome = default_genome(agent_type=WeAgent, **condition)
#     trials = 100
#     record = []

#     # build args
#     args = []
#     # for b in [2,3,4, 10]:
#     #     args.append(dict(cost = 1, benefit = b, tremble = 0))
#     for t in np.linspace(0, .15, 4):
#         args.append(dict(cost = 1, benefit = 3, tremble = t))
        
#     for arg in args:
#         for i in range(2, 17, 2):
#             for k in xrange(trials):
#                 game = RandomDictator(cost = arg['cost'], benefit = arg['benefit'], tremble = arg['tremble'], intervals = i)

#                 for t in agent_types:
#                     if t == 'self':
#                         actor = WeAgent(genome=genome, world_id='A')
#                     else:
#                         actor = t(genome=genome, world_id = 'A')
                        
#                     E_KL = 0
                     
#                     for a_id, likelihood in enumerate(actor.decide_likelihood(game, 'AB')):
#                         observer = WeAgent(genome=genome, world_id='O')
#                         q = observer.belief['A']
#                         observer.observe([(game, 'AB', 'ABO', game.actions[a_id])])
#                         p = observer.belief['A']
#                         E_KL += likelihood * sp.stats.entropy(p, q, base = 2)
                        
#                     record.append({
#                         '# Actions': i,
#                         'bits': E_KL,
#                         'tremble': arg['tremble'],
#                         'b/c': arg['benefit'] / arg['cost'],
#                         'type': lookup_agent(actor)
#                     })

#     return record

# @plotter(n_action_info, plot_exclusive_args=['data', 'color', 'graph_kwargs', 'titles'])
# def n_action_plot(data=[], color=sns.color_palette(['C5', 'C0', 'C1'])):
#     sns.set_context('notebook')

#     fig, axs = plt.subplots(3, 1, figsize = (3.5, 9), sharex=True, sharey=False)
#     means = data.groupby(['# Actions', 'type']).mean().unstack()['bits']
#     means = means.reindex(['Reciprocal', 'Altruistic', 'Selfish'], axis=1)
#     means.plot(ax=axs[0], color=color, legend = True)
    
#     means = data[data['tremble'] == 0].groupby(['# Actions', 'b/c']).mean().unstack()['bits']
#     means.plot(ax=axs[1], color= sns.color_palette("Blues"), legend = True)

#     means = data[data['b/c'] == 3].groupby(['# Actions', 'tremble']).mean().unstack()['bits']
#     means.plot(ax=axs[2], color=sns.color_palette("Blues_r"), legend = True)

#     for ax in axs:
#         ax.set_ylabel('bits')
    
#     plt.xlabel('# of Options')
#     plt.xticks(2**np.arange(1, np.log2(max(means.index))+1))

#     sns.despine()
#     plt.tight_layout()

def main(prior = 0.5, beta = 5, **kwargs):
    # first_impressions_plot(
    #     agent_types=(ReciprocalAgent, SelfishAgent),
    #     RA_prior=prior,
    #     beta=beta,
    #     RA_K=1,
    #     tremble=0.05,
    #     file_name='first_impressions', **kwargs)

    # scene_plot(
    #     scenario_func=false_belief_scenarios,
    #     agent_types=(ReciprocalAgent, SelfishAgent),
    #     RA_prior=prior,
    #     RA_K=MultiArg([0, 1, 2]),
    #     beta=beta,
    #     file_name='scene_false_belief', **kwargs)
    pass

if __name__ == '__main__':
    main()
    
    # forgive_plot(agent_types = (SelfishAgent, AltruisticAgent, 'self'))
    
    # n_action_plot(
    #     agent_types= ('self', SelfishAgent, AltruisticAgent),
    #     beta = 5,
    #     RA_prior = 0.5
    # )
