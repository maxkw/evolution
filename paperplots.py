import inspect
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import product
import agents as ag
from evolve import limit_param_plot, params_heat
from experiments import plot_beliefs, population_beliefs
import matplotlib.pyplot as plt
from evolve import param_v_rounds_heat, ssd_v_xy, ssd_param_search, ssd_v_params, ssd_bc

import params

PLOT_DIR = "./plots/" + inspect.stack()[0][1][:-3] + "/"
BETA = np.Inf
PRIOR = 0.5
MIN_TREMBLE = 0.01
TREMBLE_RANGE = lambda ticks: np.round(
    np.geomspace(MIN_TREMBLE, 0.26, ticks) - MIN_TREMBLE, 3
    # np.geomspace(MIN_TREMBLE, 0.26, ticks), 3
)

ZD = ag.ZDAgent(
    B=3,
    C=1,
    chi=3,
    phi='midpoint',
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
    TRIALS = 5

    opponents = (
        ag.SelfishAgent(beta=BETA),
        ag.AltruisticAgent(beta=BETA)
    )
    ToM = ("self",) + opponents
    agents = (ag.WeAgent(prior=PRIOR, beta=BETA, agent_types=ToM),) + opponents

    common_params = dict(
        game="game_engine",
        player_types=agents,
        s=.5,
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

    ticks = 5

    # Expected number of interactions
    def gamma_plot():
        limit_param_plot(
            param="expected_interactions",
            param_vals=np.round(np.linspace(1, 10, ticks), 2),
            tremble=MIN_TREMBLE,
            legend=True,
            analysis_type="limit",
            file_name="game_engine_gamma_b_%d" % common_params["benefit"],
            **common_params
        )

    def tremble_plot():
        # Vary tremble
        limit_param_plot(
            param="tremble",
            param_vals=TREMBLE_RANGE(ticks),
            expected_interactions=10,
            legend=False,
            analysis_type="limit",
            file_name="game_engine_tremble",
            **common_params
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
            **common_params
        )

    def heat_map():
        # Heatmap based on gamma vs. observability
        param_dict = dict(
            expected_interactions=np.round(np.linspace(1, 5, ticks)),
            observability=np.round(np.linspace(0, 1, ticks), 2),
        )
        params_heat(param_dict = param_dict,
                    tremble = MIN_TREMBLE,
                    analysis_type="complete",
                    file_name = 'game_engine_indirect_direct',
                    **common_params)

    def search_bc():
        bcs = {
            (16, 1): dict(expected_interactions=np.round(np.linspace(1, 5, ticks)), observability=np.round(np.linspace(0, 1, ticks), 2)),
            (8, 1) : dict(expected_interactions=np.round(np.linspace(1, 5, ticks)), observability=np.round(np.linspace(0, 1, ticks), 2)),
            (4, 1) : dict(expected_interactions=np.round(np.linspace(1, 5, ticks)), observability=np.round(np.linspace(0, 1, ticks), 2)),
            (2, 1) : dict(expected_interactions=np.round(np.linspace(1, 5, ticks)), observability=np.round(np.linspace(0, 1, ticks), 2)),
        }
        ssds = list()

        for b,c in bcs:
            bc_params = dict(common_params)
            bc_params.update(
                benefit = b,
                cost = c,
                param_dict = bcs[(b,c)],
                tremble = MIN_TREMBLE,
                analysis_type="complete",
                
            )
            
            # print(bc_params)
            # ssds.append(ssd_v_params(**bc_params))

            # ssd = ssd_v_params(**bc_params)
            ssd = ssd_bc(**bc_params)
            import pdb; pdb.set_trace()
            # params_heat(file_name = 'game_engine_heat%d%d' % (b,c),
                        # **bc_params)

            
            # import pdb; pdb.set_trace()
            

    # gamma_plot()
    # tremble_plot()
    # observe_plot()
    # heat_map()
    
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
    TRIALS = 100

    old_pop = (
        ag.AllC,
        ag.AllD,
        ag.GTFT,
        ag.WSLS,
        ag.TFT,
        ZD,
    )
    new_pop = old_pop + (ag.WeAgent(prior=PRIOR, beta=BETA, agent_types=("self",) + old_pop),)

    common_params = dict(
        game="direct",
        s=.5,
        benefit=BENEFIT,
        cost=COST,
        pop_size=100,
        analysis_type="limit",
        plot_dir=PLOT_DIR,
        trials=TRIALS,
        stacked=True,
    )
    tremble_ticks = 10

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
    
    for label, player_types in zip(["wRA", "woRA"], [new_pop, old_pop]):
        # Tremble
        limit_param_plot(
            param="tremble",
            param_vals=TREMBLE_RANGE(tremble_ticks),
            rounds=50,
            player_types=player_types,
            legend=False,
            file_name="ipd_tremble_%s" % label,
            graph_kwargs={"color": color_list(player_types)},
            **common_params
        )

        # By expected rounds
        limit_param_plot(
            param="rounds",
            rounds=50,
            tremble=MIN_TREMBLE,
            player_types=player_types,
            legend=True,
            file_name="ipd_rounds_%s" % label,
            graph_kwargs={"color": color_list(player_types)},
            **common_params
        )
        
    # Cognitive Costs
    def cog_cost_graph(ax):
        vals = ax.get_xticks()
        surplus = common_params["benefit"] - common_params["cost"]
        print(vals, surplus, ["{:3.0f}%".format(x / surplus * 100) for x in vals])
        ax.set_xticklabels(["{:3.0f}%".format(x / surplus * 100) for x in vals])
        ax.set_xlabel("Cognitive Cost \n% of (b-c)")


    # Pick the right tremble value to hurt WSLS. 
    limit_param_plot(
        param="cog_cost",
        tremble=0.2,
        rounds=50,
        param_vals=np.linspace(0, 0.5, 50),
        file_name="ipd_cogcosts",
        player_types=new_pop,
        graph_funcs=cog_cost_graph,
        legend=False,
        graph_kwargs={"color": color_list(new_pop)},
        **common_params
    )


def agent():
    opponents = (ag.AltruisticAgent(beta=BETA), ag.SelfishAgent(beta=BETA))
    ToM = ("self",) + opponents
    agents = (ag.WeAgent(prior=PRIOR, beta=BETA, agent_types=ToM),) + opponents

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

    reciprocal_scenarios_0 = [[["AB"], "C"], [["AB"], "D"]]
    titles_0 = [
        "Prior" + "\n ",
        r"$j$ cooperates w/ $k$" + "\n ",
        r"$j$ defects on $k$" + "\n ",
    ]

    reciprocal_scenarios_0 = [
        make_observations_from_scenario(s, **game_params)
        for s in reciprocal_scenarios_0
    ]
    reciprocal_scenarios_0.insert(0, [])

    scene_plot(
        scenarios=reciprocal_scenarios_0,
        # titles = titles_0,
        file_name="scene_reciprocal_0",
        **common_params
    )

    reciprocal_scenarios_1 = [
        [["BA", "AB"], "DD"],
        [["BA", "AB"], "DC"],
        [["BA", "AB"], "CD"],
    ]
    titles_1 = [
        r"$k$ defects on $j$" + "\n" + r"$j$ defects on $k$",
        r"$k$ defects on $j$" + "\n" + r"$j$ cooperates w/ $k$",
        r"$k$ cooperates w/ $j$" + "\n" + r"$j$ defects on $k$",
    ]

    reciprocal_scenarios_1 = [
        make_observations_from_scenario(s, **game_params)
        for s in reciprocal_scenarios_1
    ]
    scene_plot(
        scenarios=reciprocal_scenarios_1,
        # titles = titles_1,
        file_name="scene_reciprocal_1",
        **common_params
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
    everyone = (ag.AltruisticAgent(beta=BETA), ag.SelfishAgent(beta=BETA))
    agent = ag.WeAgent(prior=PRIOR, beta=BETA, agent_types=("self",) + everyone)
    
    common_params = dict(
        believer=agent,
        opponent_types=(agent,) + everyone,
        believed_types=(agent,) + everyone,
        tremble=MIN_TREMBLE,
        plot_dir=PLOT_DIR,
        deterministic=True,
        game="belief_game",
        benefit=3,
        cost=1,
        traces=0,
        trials=50,
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
        **common_params
    )

    # Public Interactions. We must set the experiment and the
    # population for this experiment.
    plot_beliefs(
        experiment=population_beliefs,
        population=population,
        rounds=1,
        observability=1,
        file_name="intra_gen_belief_public",
        **common_params
    )
    
    # Use the FSA agents as comparison agents too
    old_pop = (
        ag.AllC,
        ag.AllD,
        ag.TFT,
        ag.WSLS,
        ag.GTFT,
        ZD,
    )
    ToM = ("self",) + old_pop
    agent = ag.WeAgent(prior=PRIOR, beta=BETA, agent_types=ToM)
    common_params.update(dict(
        believer=agent,
        opponent_types=(agent,) + old_pop,
        believed_types=(agent,) + old_pop,
        game='direct',
        colors=color_list((agent,) + old_pop, sort=False),
    ))
    
    plot_beliefs(
        observability=0,
        # rounds=int(sum(population) * (sum(population) - 1) / 2 / 3),
        rounds=50,
        file_name="fsa_belief",
        **common_params
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
    # parser.add_argument("--nomemoize", action="store_true")
    # parser.add_argument("--cpus", default=1, type=int)

    args = parser.parse_args()
    # params.n_jobs = args.cpus
    # params.n_jobs = 2
    # params.memoized = args.nomemoize == False

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
