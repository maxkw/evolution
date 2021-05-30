import inspect
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import product
import agents as ag
from evolve import beta_heat, limit_param_plot, params_heat
from experiments import plot_beliefs, population_beliefs
import matplotlib.pyplot as plt
from evolve import (
    param_v_rounds_heat,
    ssd_v_xy,
    ssd_param_search,
    ssd_v_params,
    ssd_bc,
    bc_plot,
    beta_heat
)

import params

PLOT_DIR = "./plots/" + inspect.stack()[0][1][:-3] + "/"
WE_BETA = 5
# NOTE: Setting OTHER_BETA=inf makes tremble=0 blow up
OTHER_BETA = np.inf
PRIOR = 0.5
MIN_TREMBLE = 0.01
TREMBLE_RANGE = lambda ticks: np.round(
    np.geomspace(MIN_TREMBLE, 0.25, ticks),
    3
)
TREMBLE_EXP = [.00, .01, .02, .04, .08, .16, .32, .64]
ZD = ag.ZDAgent(B=3, C=1, chi=3, phi="midpoint",)

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
        if "ZD" in a:
            return "C6"
        raise "Color not defined for agent %s"

    if sort:
        return sns.color_palette([lookup(a) for a in sorted(agent_list, key=str)])
    else:
        return sns.color_palette([lookup(a) for a in agent_list])


"""
Evolution of Cooperation in the Game Engine:
1. As a function of repeated interactions
2. As a function of tremble
3. TODO what about b/c, what about # of actions?
"""


def game_engine():
    # 100 was good enough for the search_bc plot
    TRIALS = 100
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
        observability=0,
        memoized=params.memoized,
        graph_kwargs={"color": color_list(agents)},
    )

    ticks = 11
    max_expected_interactions = 5
    # max_expected_interactions = (1+ticks)/2
    # Expected number of interactions
    def gamma_plot():
        limit_param_plot(
            param="expected_interactions",
            param_vals=np.round(np.linspace(1, max_expected_interactions, ticks), 2),
            tremble=MIN_TREMBLE,
            legend=True,
            analysis_type="limit",
            file_name="game_engine_gamma",
            **common_params,
        )

    def tremble_plot():
        # Vary tremble

        # NOTE: Tremble = 0, beta = inf, and max_players = 3 will cause an error
        # because between three-player subsets there might be some partial
        # observability.

        limit_param_plot(
            param="tremble",
            # param_vals=TREMBLE_RANGE(ticks),
            param_vals=TREMBLE_EXP,
            expected_interactions=max_expected_interactions,
            legend=False,
            analysis_type="limit",
            file_name="game_engine_tremble",
            **common_params,
        )

    def observe_plot():
        # Vary observability
        limit_param_plot(
            param="observability",
            param_vals=np.round(np.linspace(0, 1, ticks), 2),
            tremble=MIN_TREMBLE,
            expected_interactions=1,
            analysis_type="complete",
            file_name="game_engine_observability",
            **common_params,
        )
        
    heat_ticks = 5
    def heat_map():
        # Heatmap based on gamma vs. observability
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
            file_name="game_engine_indirect_direct",
            **common_params_heat
        )

    def search_bc():
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

    gamma_plot()
    tremble_plot()
    observe_plot()
    heat_map()
    search_bc()


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

    old_pop = (
        ag.AllC,
        ag.AllD,
        ag.GTFT,
        ag.WSLS,
        ag.TFT,
    )
    new_pop = old_pop + (
        ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=("self",) + old_pop),
    )
    ZD_pop = new_pop + (ZD, )

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
    )
    # tremble_ticks = 5
    n_rounds = 25
    
    # from experiments import matchup_matrix_per_round, payoff_heatmap
    # tremble = MIN_TREMBLE
    # payoffs = payoff_heatmap(
    #     player_types=new_pop,
    #     max_rounds=20,
    #     game="direct",
    #     tremble=MIN_TREMBLE,
    #     benefit=BENEFIT,
    #     trials=TRIALS,
    #     cost=COST,
    #     memoized=False,
    #     plot_dir=PLOT_DIR,
    #     per_round = True,
    #     file_name="ipd_payoffs_tremble_%0.2f" % tremble,
    # )
    # assert 0
    


    # for label, player_types in zip(["wRA", "woRA", "wRAZD"], [new_pop, old_pop, ZD_pop]):
    for label, player_types in zip(["wRA", "woRA"], [new_pop, old_pop]):
        print("Running Expected Rounds with", label)
        limit_param_plot(
            param="rounds",
            rounds=n_rounds,
            tremble=MIN_TREMBLE,
            player_types=player_types,
            legend=True,
            file_name="ipd_rounds_%s" % label,
            graph_kwargs={"color": color_list(player_types)},
            **common_params,
        )

        print("Running Tremble with", label)
        limit_param_plot(
            param="tremble",
            # param_vals=TREMBLE_RANGE(tremble_ticks),
            param_vals=TREMBLE_EXP,
            rounds=10,
            player_types=player_types,
            legend=False,
            file_name="ipd_tremble_%s" % label,
            graph_kwargs={"color": color_list(player_types)},
            **common_params,
        )

    # Cognitive Costs
    cog_cost_params = np.linspace(0, 0.6, 11)
    def cog_cost_graph(ax):
        vals = cog_cost_params
        surplus = common_params["benefit"] - common_params["cost"]
        percents = ["{:3.0f}%".format(x / surplus * 100) for x in vals]
        percents = [percents[i] for i in ax.get_xticks()]
        # print(vals, surplus, percents)
        ax.set_xticklabels(percents)
        # ax.set_xlabel("Cognitive Cost \n% of (b-c)")
        ax.set_xlabel("Cognitive Cost")

    print("Running Cog Cost")
    limit_param_plot(
        param="cog_cost",
        tremble=MIN_TREMBLE,
        rounds=n_rounds,
        param_vals=cog_cost_params,
        file_name="ipd_cogcosts",
        player_types=new_pop,
        graph_funcs=cog_cost_graph,
        legend=False,
        graph_kwargs={"color": color_list(new_pop)},
        **common_params,
    )
    # # Commented out because the # of rounds should be calibrated to be right on the edge. 
    # print("Running Beta IPD")
    # limit_param_plot(
    #     param="beta",
    #     tremble=MIN_TREMBLE,
    #     rounds=5,
    #     param_vals=np.linspace(1,7,13),
    #     file_name="ipd_beta",
    #     player_types=new_pop,
    #     legend=False,
    #     graph_kwargs={"color": color_list(new_pop)},
    #     **common_params,        
    # )         

    def beta_heat_map():
        # Heatmap based on beta vs. rounds
        common_params_heat = common_params.copy()
        common_params_heat['graph_kwargs'] = dict(
                xlabel = 'Beta',
                ylabel = 'Mean Pairwise Interactions',
                xy = ("beta", "rounds"),
                onlyRA = True)
        # common_params_heat['trials'] = 20

        beta_heat(
            param_vals=np.linspace(1,7,13),
            param="beta",
            rounds=13,
            player_types=new_pop,
            tremble=MIN_TREMBLE,
            file_name="heat_beta",
            return_rounds=True,
            **common_params_heat
        )
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
        [["AB"], "C"],
        [["AB"], "D"],  
        [["BA", "AB"], "DD"],
        [["BA", "AB"], "DC"],
        [["BA", "AB"], "CC"],
        [["BA", "AB"], "CD"],
        [["prior"], "prior"]
    ]

    for s in scenarios:
        if s[1] == "prior":
            obs = []
        else:
            obs = make_observations_from_scenario(s, **game_params)
            
        scene_plot(
            observer=observer,
            scenarios=[obs],
            file_name="scene_reciprocal_" + s[1],
            **common_params,
        )


    # reciprocal_scenarios_0 = [[["AB"], "C"], [["AB"], "D"]]
    # titles_0 = [
    #     "Prior" + "\n ",
    #     r"$j$ cooperates w/ $k$" + "\n ",
    #     r"$j$ defects on $k$" + "\n ",
    # ]

    # reciprocal_scenarios_0 = [
    #     make_observations_from_scenario(s, **game_params)
    #     for s in reciprocal_scenarios_0
    # ]
    # reciprocal_scenarios_0.insert(0, [])

    # scene_plot(
    #     observer=observer,
    #     scenarios=reciprocal_scenarios_0,
    #     # titles = titles_0,
    #     file_name="scene_reciprocal_0",
    #     **common_params,
    # )

    # reciprocal_scenarios_1 = [
    #     [["BA", "AB"], "DD"],
    #     [["BA", "AB"], "DC"],
    #     [["BA", "AB"], "CD"],
    # ]
    # titles_1 = [
    #     r"$k$ defects on $j$" + "\n" + r"$j$ defects on $k$",
    #     r"$k$ defects on $j$" + "\n" + r"$j$ cooperates w/ $k$",
    #     r"repeat",
    # ]

    # reciprocal_scenarios_1 = [
    #     make_observations_from_scenario(s, **game_params)
    #     for s in reciprocal_scenarios_1
    # ]
    # scene_plot(
    #     observer=observer,
    #     scenarios=reciprocal_scenarios_1,
    #     # titles = titles_1,
    #     file_name="scene_reciprocal_1",
    #     **common_params,
    # )
    
    
    # reciprocal_scenarios_2 = [
    #     [["BA", "AB"], "DD"],
    #     [["BA", "AB"], "CC"],
    #     [["BA", "AB"], "CD"],
    # ]
    # titles_2 = [
    #     r"repeat",
    #     r"$k$ cooperates w/ $j$" + "\n" + r"$j$ cooperates w/ $k$",
    #     r"$k$ cooperates w/ $j$" + "\n" + r"$j$ defects on $k$",
    # ]

    # reciprocal_scenarios_2 = [
    #     make_observations_from_scenario(s, **game_params)
    #     for s in reciprocal_scenarios_2
    # ]
    # scene_plot(
    #     observer=observer,
    #     scenarios=reciprocal_scenarios_2,
    #     # titles = titles_2,
    #     file_name="scene_reciprocal_2",
    #     **common_params,
    # )

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
        **common_params,
    )

    # Use the FSA agents as comparison agents too
    old_pop = (
        ag.AllC,
        ag.AllD,
        ag.TFT,
        ag.WSLS,
        ag.GTFT,
        # ZD,
    )
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

    if args.debug:
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
