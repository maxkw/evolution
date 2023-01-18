import inspect
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import product
import agents as ag
from evolve import limit_param_plot, params_heat, bc_plot
from experiments import plot_beliefs, population_beliefs, payoff_heatmap
import matplotlib.pyplot as plt
import params

PLOT_DIR = "./plots/"
WE_BETA = 3
# NOTE: Setting OTHER_BETA=inf makes tremble=0 blow up
OTHER_BETA = np.inf
PRIOR = 0.5
MIN_TREMBLE = 0.01
# TREMBLE_RANGE = lambda ticks: np.round(
#     np.geomspace(MIN_TREMBLE, 0.25, ticks),
#     3
# )
# TREMBLE_EXP = [.0, .01, .02, .04, .08, .16, .32, .64]
TREMBLE_EXP = np.round(np.linspace(0,.6,13),2)
Extort2 = ag.ZDAgent(B=3, C=1, chi=2, phi="midpoint", subtype_name="Extort2")
old_pop = (
    ag.AllC,
    ag.AllD,
    ag.GTFT,
    ag.WSLS,
    ag.TFT,
    Extort2,
)

def color_list(agent_list, sort=True):
    """takes a list of agent types `agent_list` and returns the correctly
    ordered color mapping for plots
    """

    def lookup(a):
        a = str(a)
        if "WeAgent" in a:
            return "C0"
        if "AltruisticAgent" in a or "AllC" in a:
            return "C2"
        if "SelfishAgent" in a or "AllD" in a:
            return "C3"
        if "WSLS" in a:
            return "C1"
        if "GTFT" in a:
            return "C4"
        if "TFT" in a:
            return "C5"
        if "Extort2" in a:
            return "C6"
        raise "Color not defined for agent %s"

    if sort:
        return sns.color_palette([lookup(a) for a in sorted(agent_list, key=str)])
    else:
        return sns.color_palette([lookup(a) for a in agent_list])

def game_engine():
    # 100 was good enough for the search_bc plot
    TRIALS = 100
    heat_ticks = 5

    opponents = (ag.SelfishAgent(beta=OTHER_BETA), ag.AltruisticAgent(beta=OTHER_BETA))
    ToM = ("self",) + (ag.SelfishAgent(beta=WE_BETA), ag.AltruisticAgent(beta=WE_BETA))
    agents = (ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=ToM),) + opponents

    common_params = dict(
        game="game_engine",
        player_types=agents,
        s=1,
        pop_size=10,
        trials=TRIALS,
        stacked=True,
        benefit=10,
        cost=1,
        plot_dir=PLOT_DIR,
        memoized=params.memoized,
        graph_kwargs={"color": color_list(agents)},
    )

    ticks = 11
    max_expected_interactions = 5
    # max_expected_interactions = (1+ticks)/2
    # Expected number of interactions
    def gamma_plot():
        print('Running gamma plot')
        limit_param_plot(
            param_dict={"expected_interactions": np.round(np.linspace(1, max_expected_interactions, ticks), 2)},
            tremble=MIN_TREMBLE,
            observability=0,
            legend=True,
            analysis_type="limit",
            file_name="game_engine_gamma",
            **common_params,
        )

    def tremble_plot():
        print('Running tremble plot')
        # NOTE: Tremble = 0, beta = inf, and max_players = 3 will cause an error
        # because between three-player subsets there might be some partial
        # observability.
        limit_param_plot(
            param_dict={"tremble": TREMBLE_EXP},
            observability=0,
            expected_interactions=max_expected_interactions,
            analysis_type="limit",
            file_name="game_engine_tremble",
            **common_params,
        )

    def observe_plot():
        print('Running observability plot')
        limit_param_plot(
            param_dict={"observability": np.round(np.linspace(0, 1, ticks), 2)},
            tremble=MIN_TREMBLE,
            expected_interactions=1,
            analysis_type="complete",
            file_name="game_engine_observability",
            **common_params,
        )
        
    def observe_tremble_plot():
        print('Running tremble plot with observability')
        limit_param_plot(
            param_dict={"tremble": TREMBLE_EXP},
            observability=1,
            expected_interactions=1,
            analysis_type="complete",
            file_name="game_engine_tremble_public",
            **common_params,
        )        
        
    def heat_map_gamma_omega():
        print('Running heat map gamma vs. omega')
        param_dict = dict(
            expected_interactions=np.round(np.linspace(1, 3, heat_ticks), 2),
            observability=np.round(np.linspace(0, 1, heat_ticks), 2),
        )

        heat_graph_kwargs = dict(
            xlabel = 'Probability of observation',
            ylabel = '# of Interactions',
            xy = ("observability", "expected_interactions"),
            onlyRA = True,
        )
        common_params_heat = common_params.copy()
        common_params_heat['graph_kwargs'] = {
            **common_params_heat['graph_kwargs'],
            **heat_graph_kwargs
        }

        params_heat(
            param_dict=param_dict,
            tremble=MIN_TREMBLE,
            analysis_type="complete",
            line=True,
            file_name="game_engine_gamma_omega",
            **common_params_heat
        )

    def heat_map_gamma_tremble():
        print('Running heat map gamma vs. tremble')
        param_dict = dict(
            expected_interactions=np.round(np.linspace(1, 3, heat_ticks), 2),
            tremble=TREMBLE_EXP,
        )

        heat_graph_kwargs = dict(
            xlabel = 'Probability of tremble',
            ylabel = '# of Interactions',
            xy = ("tremble", "expected_interactions"),
            onlyRA = True,
        )
        common_params_heat = common_params.copy()
        common_params_heat['graph_kwargs'] = {
            **common_params_heat['graph_kwargs'],
            **heat_graph_kwargs
        }

        params_heat(
            param_dict=param_dict,
            tremble=MIN_TREMBLE,
            analysis_type="limit",
            line=True,
            observability=0,
            file_name="game_engine_gamma_tremble",
            **common_params_heat
        )

    def heat_map_omega_tremble():
        print('Running heat map omega vs. tremble')
        param_dict = dict(
            tremble=TREMBLE_EXP,
            observability=np.round(np.linspace(0, 1, heat_ticks), 2),
        )

        heat_graph_kwargs = dict(
            xlabel = 'Probability of tremble',
            ylabel = 'Probability of observation',
            xy = ("tremble", "observability"),
            onlyRA = True,
        )
        common_params_heat = common_params.copy()
        common_params_heat['graph_kwargs'] = {
            **common_params_heat['graph_kwargs'],
            **heat_graph_kwargs
        }

        params_heat(
            param_dict=param_dict,
            tremble=MIN_TREMBLE,
            analysis_type="complete",
            line=True,
            expected_interactions=1,            file_name="game_engine_omega_tremble",
            **common_params_heat
        )

    def search_bc():
        print('Running search bc')
        bcs = {
            (15, 1): 3,
            (10, 1): 4,
            (5, 1): 6,
            # (2, 1): 15,
        }

        bc_plot(
            ei_stop=bcs,
            observe_param=np.round(np.linspace(1, 0, heat_ticks), 2),
            delta=0.025,
            tremble=MIN_TREMBLE,
            analysis_type="complete",
            file_name="bc_plot",
            **common_params)

    # gamma_plot()
    # tremble_plot()
    # observe_plot()
    # observe_tremble_plot()
    heat_map_gamma_omega()
    # heat_map_gamma_tremble()
    # heat_map_omega_tremble()
    # search_bc()


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


def ipd():
    BENEFIT = 3
    COST = 1
    # 10000 puts SEM around 0.001-0.002
    TRIALS = 1000
    # TRIALS = 100

    new_pop = old_pop + (
        ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=("self",) + old_pop),
    )

    common_params = dict(
        game="direct",
        s=1,
        benefit=BENEFIT,
        cost=COST,
        pop_size=100,
        analysis_type="limit",
        plot_dir=PLOT_DIR,
        trials=TRIALS,
        stacked=True,
        return_rounds=True,
        per_round=True,            
    )
    
    n_rounds = 25

    selfpayoff_params = common_params.copy()
    selfpayoff_params.update(
        stacked=False,
        var='selfpayoff',
    )
    
    for label, player_types in zip(["wRA", "woRA"], [new_pop, old_pop]):
        print("Running Expected Rounds with", label)
        limit_param_plot(
            param_dict=dict(rounds=[n_rounds]),
            tremble=MIN_TREMBLE,
            player_types=player_types,
            legend=True,
            file_name="ipd_rounds_%s" % label,
            graph_kwargs={"color": color_list(player_types),
                          "xlabel": "# Pairwise Interactions"},
            **common_params,
        )

        print("Running Pop Size with", label)
        limit_param_plot(
            param_dict=dict(pop_size=[2,4,8,16,32,64,128]),
            rounds=n_rounds,
            tremble=MIN_TREMBLE,
            player_types=player_types,
            file_name="ipd_popsize_%s" % label,
            graph_kwargs={"color": color_list(player_types),
                          "xlabel": "Population Size"},
            **common_params,
        )

        print("Running Tremble with", label, TREMBLE_EXP)
        limit_param_plot(
            param_dict=dict(tremble=TREMBLE_EXP),
            rounds=n_rounds,
            player_types=player_types,
            file_name="ipd_tremble_%s" % label,
            graph_kwargs={"color": color_list(player_types)},
            **common_params,
        )

    print('Running Self Payoff')
    limit_param_plot(
        param_dict=dict(rounds=[n_rounds]),
        tremble=MIN_TREMBLE,
        player_types=new_pop,
        file_name="ipd_rounds_selfpay",
        graph_kwargs={"color": color_list(new_pop),
                        "xlabel": "# Pairwise Interactions"},
        **selfpayoff_params,
    )
    limit_param_plot(
        param_dict=dict(tremble=TREMBLE_EXP),
        rounds=n_rounds,
        tremble=MIN_TREMBLE,
        player_types=new_pop,
        file_name="ipd_tremble_selfpay",
        graph_kwargs={"color": color_list(new_pop)},
        **selfpayoff_params,
    )
   
    print('Running Payoff Heatmap')
    for r in [5, n_rounds]:
        payoff_heatmap(
            rounds=r,
            player_types=new_pop,
            tremble=MIN_TREMBLE,
            sem=False,
            file_name="ipd_payoffs_rounds_{}".format(r),
            **common_params
        )    

    cog_cost_params = np.linspace(0, 0.6, 11)
    def cog_cost_graph(ax):
        vals = cog_cost_params
        surplus = common_params["benefit"] - common_params["cost"]
        percents = ["{:3.0f}%".format(x / surplus * 100) for x in vals]
        percents = [percents[i] for i in ax.get_xticks()]
        ax.set_xticklabels(percents)
        # ax.set_xlabel("Cognitive Cost \n% of (b-c)")
        ax.set_xlabel("Cognitive Cost")

    print("Running Cog Cost")
    limit_param_plot(
        param_dict=dict(cog_cost=cog_cost_params),
        tremble=MIN_TREMBLE,
        rounds=n_rounds,
        file_name="ipd_cogcosts",
        player_types=new_pop,
        graph_funcs=cog_cost_graph,
        graph_kwargs={"color": color_list(new_pop)},
        **common_params,
    )
    
    print("Running Beta IPD")
    limit_param_plot(
        param_dict=dict(beta=np.append(np.linspace(1,6.5,12), np.inf)),
        tremble=MIN_TREMBLE,
        rounds=5,
        file_name="ipd_beta",
        player_types=new_pop,
        graph_kwargs={"color": color_list(new_pop),
                      "xlabel": r"Softmax ($\beta$)"},
        **common_params,        
    )         
    
    def beta_heat_map():
        print('Running Beat Heat Map')
        heat_graph_kwargs = dict(
                xlabel = r'Softmax ($\beta$)',
                ylabel = '# Pairwise Interactions',
                xy = ("beta", "rounds"),
                onlyRA = True)
        
        param_dict = dict(
            beta=np.append(np.linspace(1,7.5,14), np.inf),
            rounds=[10],
        )
        
        params_heat(
            param_dict=param_dict,
            player_types=new_pop,
            tremble=MIN_TREMBLE,
            file_name="ipd_heat_beta",
            line = True,
            graph_kwargs=heat_graph_kwargs,
            **common_params
        )
    
    def tremble_heat_map():
        print('Running Tremble Heat Map')
        heat_graph_kwargs = dict(
                xlabel = r"Prob. of action error ($\epsilon$)",
                ylabel = '# Pairwise Interactions',
                xy = ("tremble", "rounds"),
                onlyRA = True)
        
        param_dict = dict(
            tremble=TREMBLE_EXP,
            rounds=[n_rounds],
        )
        
        params_heat(
            param_dict=param_dict,
            player_types=new_pop,
            file_name="ipd_heat_tremble",
            line = True,
            graph_kwargs=heat_graph_kwargs,
            **common_params
        )
        
    tremble_heat_map()
    beta_heat_map()    
  

def agent():
    opponents = (ag.AltruisticAgent(beta=OTHER_BETA), ag.SelfishAgent(beta=OTHER_BETA))
    ToM = ("self",) + (ag.AltruisticAgent(beta=WE_BETA), ag.SelfishAgent(beta=WE_BETA))
    agents = (ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=ToM),) + opponents
    
    
    observer = ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=ToM)
    common_params = dict(
        player_types=agents,
        agent_types=ToM,
        plot_dir=PLOT_DIR,
        trials=1,
        color=color_list(agents, sort=False),
    )

    game_params = dict(tremble=MIN_TREMBLE, benefit=3, cost=1)

    # SCENARIO PLOTS
    from scenarios import scene_plot, make_observations_from_scenario

    scenarios = [
        ('Alice', [["AB"], "C"]),
        ('Alice', [["AB"], "D"]),  
        ('Bob', [["BA", "AB"], "DD"]),
        ('Bob', [["BA", "AB"], "DC"]),
        ('Bob', [["BA", "AB"], "CC"]),
        ('Bob', [["BA", "AB"], "CD"]),
        ('Alice', [["prior"], "prior"]),
        ('Bob', [["prior"], "prior"])
    ]

    for xlabel, s in scenarios:
        if s[1] == "prior":
            obs = []
        else:
            obs = make_observations_from_scenario(s, **game_params)
            
        scene_plot(
            observer=observer,
            scenarios=[obs],
            xlabel="$P(U_{" + xlabel + "})$",
            file_name="scene_reciprocal_{}_{}".format(s[1], xlabel),
            **common_params,
        )

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
    everyone = (ag.AltruisticAgent(beta=OTHER_BETA), ag.SelfishAgent(beta=OTHER_BETA))
    ToM = ("self",) + (ag.AltruisticAgent(beta=WE_BETA), ag.SelfishAgent(beta=WE_BETA))
    agent = ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=ToM)

    common_params = dict(
        believer=agent,
        opponent_types=(agent,) + everyone,
        believed_types=(agent,) + ToM[1:],
        tremble=MIN_TREMBLE,
        plot_dir=PLOT_DIR,
        deterministic=True,
        game="belief_game",
        benefit=3,
        cost=1,
        traces=5,
        trials=1000,
        memoized=params.memoized,
        colors=color_list((agent,) + everyone, sort=False),
    )

    population = [4, 3, 3]

    # Private Interactions. Not we do not set the experiment or
    # population for this experiment.
    plot_beliefs(
        observability=0,
        # Below computes makes it so the number of interactions are
        # equivalent in both cases. 
        rounds=int(sum(population) * (sum(population) - 1) / 2 / 3),
        xlabbel="# Observations",
        file_name="intra_gen_belief_private",
        **common_params,
    )

    # Public Interactions. We must set the experiment and the
    # population for this experiment.
    plot_beliefs(
        experiment=population_beliefs,
        population=population,
        rounds=1,
        observability=1,
        file_name="intra_gen_belief_public",
        xlabel = "# Obersevations",
        **common_params,
    )

    # FSA agents
    ToM = ("self",) + old_pop
    agent = ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=ToM)
    common_params.update(
        dict(
            believer=agent,
            opponent_types=(agent,) + old_pop,
            believed_types=(agent,) + old_pop,
            game="direct",
            colors=color_list((agent,) + old_pop, sort=False),
        )
    )

    plot_beliefs(
        observability=0,
        rounds=15,
        file_name="fsa_belief",
        xlabel="# Pairwise Interactions",
        **common_params,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--belief", action="store_true")
    parser.add_argument("--ipd", action="store_true")
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--engine", action="store_true")
    parser.add_argument("--heatmaps", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.all:
        game_engine()
        belief()
        ipd()
        agent()

    if args.belief:
        belief()

    if args.ipd:
        ipd()

    if args.agent:
        agent()

    if args.engine:
        game_engine()

    if args.heatmaps:
        pass

    if args.debug:
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
