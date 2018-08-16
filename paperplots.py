import inspect
import numpy as np
import seaborn as sns
from itertools import product

import agents as ag
from evolve import limit_param_plot, complete_sim_plot
from experiment_utils import MultiArg
from experiments import plot_beliefs, population_beliefs
from utils import splits
import matplotlib.pyplot as plt

from evolve import param_v_rounds_heat, ssd_v_xy, ssd_param_search, ssd_v_params

PLOT_DIR = "./plots/"+inspect.stack()[0][1][:-3]+"/"
BETA = np.Inf
PRIOR = 0.5
MIN_TREMBLE = 0.01
TREMBLE_RANGE = lambda ticks: np.round(np.geomspace(MIN_TREMBLE, .26, ticks)- MIN_TREMBLE, 3)

def color_list(agent_list, sort = True):
    '''takes a list of agent types `agent_list` and returns the correctly
    ordered color mapping for plots
    '''
    def lookup(a):
        a = str(a)
        if 'WeAgent' in a: return 'C0'
        if 'AltruisticAgent' in a or 'AllC' in a: return 'C2'
        if 'SelfishAgent' in a or 'AllD' in a: return 'C3'
        if 'WSLS' in a: return 'C1'
        if 'GTFT' in a: return 'C4'
        if 'TFT' in a: return 'C5'
        raise('Color not defined for agent %s' % a)
    
    if sort:
        return sns.color_palette([lookup(a) for a in sorted(agent_list, key=str)])
    else: 
        return sns.color_palette([lookup(a) for a in agent_list])

'''
Evolution of Cooperation in the Game Engine:
1. As a function of repeated interactions
2. As a function of tremble
3. TODO what about b/c, what about # of actions?
'''
def game_engine():
    TRIALS = 200
    
    opponents = (ag.SelfishAgent(beta = BETA), ag.AltruisticAgent(beta = BETA))
    ToM = ('self',) + opponents
    agents = (ag.WeAgent(prior = PRIOR, beta = BETA, agent_types = ToM),) + opponents

    common_params = dict(
        game = "game_engine",
        player_types = agents,
        s = .5,
        pop_size = 10,
        trials = TRIALS,
        stacked = True,
        benefit = 10,
        plot_dir = PLOT_DIR,
        observability = 0,
        overmind = True,
        graph_kwargs = {'color' : color_list(agents)},
    )

    ticks = 3
    
    # from evolve import params_heat
    # params = {'expected_interactions': np.round(np.linspace(1, 4, ticks)),
    #           'observability': np.round(np.linspace(0, 1, ticks), 2)}
    
    # # Heatmap based on gamma vs. observability
    # params_heat(params,
    #             tremble = 0,
    #             file_name = 'game_engine_indirect_direct',
    #             **common_params)
    
    # Expected number of interactions
    def gamma_plot():
        limit_param_plot(
            param = 'expected_interactions',
            param_vals = np.round(np.linspace(1, 10, ticks), 2),
            tremble = MIN_TREMBLE, 
            analysis_type = 'limit',
            file_name = 'game_engine_gamma',
            **common_params)

    def tremble_plot():
        # Vary tremble
        limit_param_plot(
            param = 'tremble',
            param_vals = TREMBLE_RANGE(ticks),
            expected_interactions = 10,
            analysis_type = 'limit',
            file_name = 'game_engine_tremble',
            **common_params)

    def observe_plot():
        # Vary observability
        limit_param_plot(
            param = 'observability',
            param_vals = np.round(np.linspace(0, 1, ticks), 2),
            expected_interactions = 1,
            analysis_type = 'complete',
            file_name = 'game_engine_observability',
            **common_params)

    

    
    # # Agent Sim Plots

    # sim_  params = dict(
    #     generations = 60000,
    #     mu = 0.001,
    #     start_pop = (0, 10, 0),
    #     tremble = 0,
    #     **common_params)
    
    # complete_sim_plot(
    #     expected_interactions = 10,
    #     observability = 0,
    #     file_name = 'game_engine_sim_direct',
    #     seed = 0,
    #     **sim_params)

    # complete_sim_plot(
    #     expected_interactions = 1,
    #     observability = 1,
    #     file_name = 'game_engine_sim_indirect',
    #     seed = 0,
    #     **sim_params)

    gamma_plot()
    #tremble_plot()
    #observe_plot()
def ipd():
    TRIALS = 100
    
    old_pop = (ag.AllC,ag.AllD,ag.GTFT,ag.TFT,ag.WSLS)
    ToM = ('self',) + old_pop
    new_pop = old_pop + (ag.WeAgent(prior = PRIOR, beta = BETA, agent_types = ToM),)

    common_params = dict(
        game = 'direct',
        s = .5,
        benefit = 3,
        cost = 1,
        pop_size = 100,
        analysis_type = 'limit',
        plot_dir = PLOT_DIR,
        trials = TRIALS,
        stacked = True,
    )

    from experiments import matchup_matrix_per_round, payoff_heatmap
    # payoffs = matchup_matrix_per_round(player_types = new_pop,
    #                                    max_rounds = 50,
    #                                    game = 'direct',
    #                                    tremble = 0,
    #                                    benefit = 3,
    #                                    trials = TRIALS,
    #                                    cost = 1)

    payoffs = payoff_heatmap(player_types = new_pop,
                             max_rounds = 50,
                             game = 'direct',
                             tremble = .1,
                             benefit = 3,
                             trials = TRIALS,
                             cost = 1,
                             plot_dir = PLOT_DIR,
                             file_name = "ipd_payoffs_tremble_%d" % 0)
    
    
    import ipdb; ipdb.set_trace()
    
    for label, player_types in zip(['wRA', 'woRA'], [new_pop, old_pop]):
        # By expected rounds
        limit_param_plot(
            param = 'rounds',
            rounds = 50,
            tremble = MIN_TREMBLE,
            player_types = player_types,
            file_name = "ipd_rounds_%s" % label,
            graph_kwargs = {'color' : color_list(player_types)},
            **common_params)

        # Tremble
        limit_param_plot(
            param = 'tremble',
            param_vals = TREMBLE_RANGE(20),
            rounds = 10,
            player_types = player_types,
            file_name = "ipd_tremble_%s" % label,
            graph_kwargs = {'color' : color_list(player_types)},
            **common_params)

    # Cognitive Costs
    def cog_cost_graph(ax):
        vals = ax.get_xticks()
        surplus = common_params['benefit'] - common_params['cost']
        print vals, surplus, ['{:3.0f}%'.format(x / surplus * 100) for x in vals]
        ax.set_xticklabels(['{:3.0f}%'.format(x / surplus * 100) for x in vals])
        ax.set_xlabel('Cognitive Cost \n% of (b-c)')
    
    limit_param_plot(param = 'cog_cost',
                     # tremble = .15,
                     tremble = .01,
                     rounds = 50,
                     param_vals = np.linspace(0, .5, 50),
                     file_name = "ipd_cogcosts",
                     player_types = new_pop,
                     graph_funcs = cog_cost_graph,
                     graph_kwargs = {'color' : color_list(new_pop)},
                     **common_params)

def agent():
    opponents = (ag.AltruisticAgent(beta = BETA), ag.SelfishAgent(beta = BETA))
    ToM = ('self',) + opponents
    agents = (ag.WeAgent(prior = PRIOR, beta = BETA, agent_types = ToM),) + opponents

    common_params = dict(
        player_types = agents,
        agent_types = ToM,
        plot_dir = PLOT_DIR,
        color = color_list(agents, sort=False)
    )

    game_params = dict(
        tremble = MIN_TREMBLE,
        benefit = 3,
        cost = 1
    )

    # SCENARIO PLOTS
    from scenarios import scene_plot, make_observations_from_scenario

    reciprocal_scenarios_0 = [[["AB"], "C"], [["AB"], "D"]]
    titles_0 = ['Prior' + "\n ",
              r'$j$ cooperates w/ $k$' + "\n ",
              r'$j$ defects on $k$' + "\n "]
    
    reciprocal_scenarios_0 = [make_observations_from_scenario(s, **game_params) for s in reciprocal_scenarios_0]
    reciprocal_scenarios_0.insert(0, [])

    scene_plot(
        scenarios=reciprocal_scenarios_0,
        # titles = titles_0,
        file_name='scene_reciprocal_0',
        **common_params)

    reciprocal_scenarios_1 = [
        [["BA", "AB"], "DD"],
        [["BA", "AB"], "DC"],
        [["BA", "AB"], "CD"],
    ]
    titles_1 = [r'$k$ defects on $j$' + "\n" + r'$j$ defects on $k$',
                r'$k$ defects on $j$' + "\n" + r'$j$ cooperates w/ $k$', 
                r'$k$ cooperates w/ $j$' + "\n" + r'$j$ defects on $k$']

    reciprocal_scenarios_1 = [make_observations_from_scenario(s, **game_params) for s in reciprocal_scenarios_1]
    scene_plot(
        scenarios=reciprocal_scenarios_1,
        # titles = titles_1,
        file_name='scene_reciprocal_1',
        **common_params)

    # # FORGIVENESS AND REPUTATION
    # from scenarios import forgive_plot
    # forgive_plot(
    #     p = 'belief',
    #     file_name = 'forgive_belief',
    #     label = 'Belief',
    #     **common_params
    # )

    # forgive_plot(
    #     p = 'decision',
    #     file_name = 'forgive_act',
    #     label = 'Probability of Cooperate',
    #     **common_params
    # )
    
    # from scenarios import n_action_plot
    # n_action_plot(
    #     file_name = 'n_action_info', 
    #     **common_params
    # )

def belief():
    everyone = (ag.AltruisticAgent(beta = BETA), ag.SelfishAgent(beta = BETA))
    agent = ag.WeAgent(prior = PRIOR, beta = BETA, agent_types = ('self',) + everyone)

    common_params = dict(
        believer = agent,
        opponent_types = (agent,)+everyone,
        believed_types = (agent,)+everyone,

        tremble = MIN_TREMBLE,
        plot_dir = PLOT_DIR,
        deterministic = True,
        game = 'belief_game',

        traces = 0,
        trials = 200,
        overmind = True, 
        colors = color_list((agent,)+everyone, sort = False)
    )

    population = [4, 3, 3]
    
    # Private Interactions. Not we do not set the experiment or
    # population for this experiment.
    plot_beliefs(
        observability = 0,
        # Below computes makes it so the number of interactions are
        # equivalent in both cases.
        rounds = sum(population)*(sum(population)-1) / 2 / 3,
        file_name = "intra_gen_belief_private",
        **common_params)

    # Public Interactions. We must set the experiment and the
    # population for this experiment.
    plot_beliefs(
        experiment = population_beliefs,
        population = population,
        rounds = 1,
        observability = 1,
        file_name = "intra_gen_belief_public",
        **common_params)
    
def explore_param_dict():
    opponents = (ag.SelfishAgent,ag.AltruisticAgent)
    ToM = ('self',)+opponents
    WE = ag.WeAgent(agent_types=ToM)
    agents = (WE,)+opponents
    common_params = dict(s=.5,
                         f = ssd_param_search,
                         #game='dynamic',
                         game = "game_engine",
                         #game = 'direct',
                         #game = 'dd_ci_va',
                         player_types = agents,
                         #analysis_type = 'limit',
                         #beta = BETA,
                         pop_size = 10,
                         #benefit = 10,
                         tremble = 0,
                         observability = 0,
                         row_param = "expected_interactions", row_vals = np.linspace(1,5,splits(1)),
                         trials = 100,
                         benefit = 10,
                         param = "observability",
                         param_lim = [0,1],
                         target = WE,
                         param_tol = .05,
                         mean_tol = .1)

    return common_params

def heatmaps():
    from explore import param_sweep, grid_v_param, grid_param_plot
    plot_dir = "./plots/heatmaps/"
    
    WA = ag.WeAgent
    SA = ag.SelfishAgent
    AA = ag.AltruisticAgent
    everyone = (SA, AA)
    ToM = ('self',) + everyone
    ra = WA(#prior = .5,
            agent_types = ToM)
    player_types= (ra,)+everyone

    res = 2
    y_param = 'observability'
    y_vals = np.linspace(0, 1, splits(res))
    x_vals = np.linspace(1,5,splits(res))
    x_param = 'expected_interactions'

    common_params = dict(player_types = player_types,

                         game = 'game_engine',
                         benefit = 10,
                         analysis_type = 'limit',
                         s = .5,
                         pop_size = 10,
                         observability = 0.0,
                         trials = 200,
                         tremble = 0,
                         #agent_types = ToM,
                         #parallelized = False,
                         #deterministic = True,
    )

    heat_params = dict(y_param = y_param,
                       #y_vals = np.linspace(0, 1, splits(1)),
                       x_param = x_param,
                       x_vals = x_vals)

    search_params = dict(f = ssd_param_search,
                         param = y_param,
                         params = {x_param: x_vals},
                         #row_param = x_param, row_vals = x_vals,
                         #row_param = "expected_interactions", row_vals = np.linspace(1,5,splits(0)),
                         param_lim = [0,1],
                         target = WA,
                         param_tol = .05,
                         mean_tol = .1)

    hmd = ssd_v_params(params = {y_param:y_vals, x_param:x_vals},**common_params)
    hmd_ = hmd[hmd['type'] == ra.short_name('agent_types')]


    #hmd_ = hmd
    ####import pdb;pdb.set_trace()
    #seaborn.jointplot(x = '')
    d = hmd_.pivot(index = y_param, columns = x_param, values = 'proportion')
    #print d
    ax = sns.heatmap(d,cbar = False,# square=True,
                     vmin = 0,
                     vmax = 1,
                     square = True,
                     #vmax=d['proportion'].max(),
                     cmap = plt.cm.gray_r,)
    ax.invert_yaxis()

    
    
    #sd = param_sweep(**dict(common_params, **search_params))
    #sd = grid_v_param(**dict(common_params, **search_params))
    #sd = grid_v_param(**explore_param_dict())
    #sd, ax = grid_param_plot(**explore_param_dict())

    #sd.plot(#ax = ax,
    #        x = x_param, y = y_param)
    #plt.ylim([0,1])

    file_name = "topology"

    plt.savefig(plot_dir+file_name)

    #colors = color_list((agent,)+everyone, sort = False))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--belief", action="store_true")
    parser.add_argument("--ipd", action="store_true")
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--engine", action="store_true")
    parser.add_argument("--heatmaps", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        game_engine()
        belief()
        ipd()
        agent()
        heatmaps()
    
    if args.belief:
        belief()

    if args.ipd:
        ipd()

    if args.agent:
        agent()

    if args.engine:
        game_engine()

    if args.heatmaps:
        heatmaps()
    
if __name__ == '__main__':
    main()
