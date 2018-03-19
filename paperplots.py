import inspect
import numpy as np
import seaborn as sns

import agents as ag
from evolve import limit_param_plot


PLOT_DIR = "./plots/"+inspect.stack()[0][1][:-3]+"/"
TRIALS = 10
POP_SIZE = 10

def color_list(agent_list):
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

    return sns.color_palette([lookup(a) for a in sorted(agent_list, key=str)])

'''
Evolution of Cooperation in the Game Engine:
1. As a function of repeated interactions
2. As a function of tremble
3. TODO what about b/c, what about # of actions?
'''
def game_engine():
    beta = 5
    prior = 0.5
    opponents = (ag.SelfishAgent(beta = beta), ag.AltruisticAgent(beta = beta))
    ToM = ('self',) + opponents
    agents = (ag.WeAgent(prior = prior, beta = beta, agent_types = ToM),) + opponents

    common_params = dict(
        game = "game_engine",
        player_types = agents,
        observability = 0,
        analysis_type = 'complete',
        s = .5,
        pop_size = POP_SIZE,
        trials = TRIALS,
        stacked = True,
        plot_dir = PLOT_DIR,
        graph_kwargs = {'color' : color_list(agents)},
    )

    from evolve import params_heat
    ticks = 2
    params = {'expected_interactions': np.round(np.linspace(1, 10, ticks)),
              'observability': np.round(np.linspace(0, 1, ticks), 2)}
    
    # Heatmap based on gamma vs. observability
    params_heat(params,
                tremble = 0,
                file_name = 'game_engine_indirect_direct',
                **common_params)
    
    # # Vary expected number of repetitions
    # limit_param_plot(
    #     param = 'expected_interactions',
    #     param_vals = np.round(np.linspace(1, 10, 10), 2),
    #     tremble = 0.0, 
    #     file_name = 'game_engine_gamma',
    #     **common_params)

    # # Vary tremble
    # limit_param_plot(
    #     param = 'tremble',
    #     param_vals = np.round(np.linspace(0, 1, 10), 2),
    #     expected_interactions = 5,
    #     file_name = 'game_engine_tremble',
    #     **common_params)
    
    # complete_sim_plot(
    #     generations = 1500,
    #     # param = 'rounds',
    #     mu = .001,
    #     tremble = 0,
    #     expected_interactions = 5,
    #     start_pop = (0,10,0),
    #     file_name = 'game_engine_sim',
    #     # seed = 0,
    #     **common_params)

def ipd():
    old_pop = (ag.AllC,ag.AllD,ag.GTFT,ag.TFT,ag.WSLS)
    ToM = ('self',) + old_pop
    new_pop = old_pop + (ag.WeAgent(prior = .5, beta = 5, agent_types = ToM),)

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

    
    # Adding cognitive costs
    limit_param_plot(param = 'cog_cost',
                     tremble = 0.0,
                     rounds = 10,
                     param_vals = np.linspace(0,.3, 50),
                     file_name = "ipd_cogcosts",
                     player_types = new_pop,
                     graph_kwargs = {'color' : color_list(new_pop)},
                     **common_params)

    
    
if __name__ == '__main__':
    game_engine()
    # ipd()










# plot_coop_prob(file_name = 'coop_prob', **params)
# scenarios.main(**params)
# indirect.ToM_indirect(**params)
    

# # Currently broken
# evolve.AllC_AllD_race()
# evolve.Pavlov_gTFT_race()




# condition = dict(trials=50,
#                  plot_dir=plot_dir,
#                  beta=5,
# )



# belief_plot(believed_type=ReciprocalAgent,
#             player_types=ReciprocalAgent, priors=.75, Ks=1, **condition)
# belief_plot(experiment=forgiveness, believed_type=ReciprocalAgent,
#             player_types=ReciprocalAgent, priors=.75, Ks=1, **condition)
# joint_fitness_plot(**condition)
# reward_table(player_types=ReciprocalAgent, size=11, Ks=0, **condition)
# reward_table(player_types=ReciprocalAgent, size=11, Ks=1, **condition)


# scene_plot(RA_prior=.75, RA_K=MultiArg([0, 1]))

# pop_fitness_plot((ReciprocalAgent, SelfishAgent), proportion=MultiArg([float(i) / 10 for i in range(10)[1:]]), Ks=MultiArg(
#     range(3)), plot_dir=plot_dir, trials=500, agent_types=(ReciprocalAgent, SelfishAgent), min_pop_size=50, beta=1)
