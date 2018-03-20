import inspect
import numpy as np
import seaborn as sns

import agents as ag
from evolve import limit_param_plot, complete_sim_plot
from scenarios import scene_plot, reciprocal_scenarios_0, reciprocal_scenarios_1
from experiment_utils import MultiArg

PLOT_DIR = "./plots/"+inspect.stack()[0][1][:-3]+"/"
TRIALS = 10
BETA = 5
PRIOR = 0.5

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
    opponents = (ag.SelfishAgent(beta = BETA), ag.AltruisticAgent(beta = BETA))
    ToM = ('self',) + opponents
    agents = (ag.WeAgent(prior = prior, beta = BETA, agent_types = ToM),) + opponents

    common_params = dict(
        game = "game_engine",
        player_types = agents,
        s = .5,
        pop_size = 10,
        trials = TRIALS,
        stacked = True,
        plot_dir = PLOT_DIR,
        graph_kwargs = {'color' : color_list(agents)},
    )

    ticks = 50
    
    # from evolve import params_heat
    # params = {'expected_interactions': np.round(np.linspace(1, 4, ticks)),
    #           'observability': np.round(np.linspace(0, 1, ticks), 2)}
    
    # # Heatmap based on gamma vs. observability
    # params_heat(params,
    #             tremble = 0,
    #             file_name = 'game_engine_indirect_direct',
    #             **common_params)
    
    # Expected number of interactions
    limit_param_plot(
        param = 'expected_interactions',
        param_vals = np.round(np.linspace(1, 10, ticks), 2),
        tremble = 0.0, 
        analysis_type = 'limit',
        file_name = 'game_engine_gamma',
        **common_params)

    # # Vary tremble
    # limit_param_plot(
    #     param = 'tremble',
    #     param_vals = np.round(np.linspace(0, 1, ticks), 2),
    #     expected_interactions = 10,
    #     analysis_type = 'limit',
    #     file_name = 'game_engine_tremble',
    #     **common_params)

    # # Agent Sim Plots

    # sim_params = dict(
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

    
def ipd():
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

    for label, player_types in zip(['wRA', 'woRA'], [old_pop, new_pop]):
        # By expected rounds
        limit_param_plot(
            param = 'rounds',
            rounds = 50,
            tremble = 0.0,
            player_types = player_types,
            file_name = "ipd_rounds_%s" % label,
            graph_kwargs = {'color' : color_list(player_types)},
            **common_params)

        # Tremble
        limit_param_plot(
            param = 'tremble',
            param_vals = np.round(np.linspace(0, 0.4, 11), 2),
            rounds = 10,
            player_types = player_types,
            file_name = "ipd_tremble_%s" % label,
            graph_kwargs = {'color' : color_list(player_types)},
            **common_params)

    # Cognitive Costs
    def cog_cost_graph(ax):
        vals = ax.get_xticks()
        surplus = common_params['benefit'] - common_params['cost']
        ax.set_xticklabels(['{:3.0f}%'.format(x / surplus * 100) for x in vals])
        ax.set_xlabel('Cognitive Cost \n% of (b-c)')
    
    limit_param_plot(param = 'cog_cost',
                     tremble = 0,
                     rounds = 50,
                     param_vals = np.linspace(0, .3, 50),
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
        agent_types = ToM,
        plot_dir = PLOT_DIR,
        color = color_list(agents, sort=False)
    )

    # SCENARIO PLOTS
    scene_plot(
        scenario_func=reciprocal_scenarios_0,
        titles = ['Prior' + "\n ",
                  r'$j$ cooperates w/ $k$' + "\n ",
                  r'$j$ defects on $k$' + "\n "],
        file_name='scene_reciprocal_0',
        **common_params)

    scene_plot(
        scenario_func=reciprocal_scenarios_1,
        titles = [r'$k$ defects on $j$' + "\n" + r'$j$ defects on $k$',
                  r'$k$ defects on $j$' + "\n" + r'$j$ cooperates w/ $k$', 
                  r'$k$ cooperates w/ $j$' + "\n" + r'$j$ defects on $k$'],
        file_name='scene_reciprocal_1',
        **common_params)

    

if __name__ == '__main__':
    # game_engine()
    # ipd()
    agent()










# plot_coop_prob(file_name = 'coop_prob', **params)
# indirect.ToM_indirect(**params)
    
