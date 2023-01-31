import argparse
import numpy as np
import seaborn as sns
import agents as ag
from evolve import limit_param_plot, params_heat, bc_plot
from experiments import plot_beliefs, population_beliefs, payoff_heatmap
from scenarios import scene_plot, make_observations_from_scenario, decision_plot, forgive_plot
from utils import excluding_keys
import params
import os


PLOT_DIR = "./plots/"
WE_BETA = 3
# NOTE: Setting OTHER_BETA=inf makes tremble=0 blow up
OTHER_BETA = np.inf
PRIOR = 0.5
# TREMBLE_EXP = [.0, .01, .02, .04, .08, .16, .32, .64]
TREMBLE_EXP = np.round(np.linspace(0,.3,13),3)
MIN_TREMBLE = TREMBLE_EXP[1]
# MIN_TREMBLE = .01
Extort2 = ag.ZDAgent(B=3, C=1, chi=2, phi="midpoint", subtype_name="Extort2")

old_pop = (
    ag.AllC,
    ag.AllD,
    ag.GTFT,
    ag.WSLS,
    ag.TFT,
    ag.Forgiver,
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
        if "Forgiver" in a:
            return "C7"        
        raise "Color not defined for agent %s"

    if sort:
        return sns.color_palette([lookup(a) for a in sorted(agent_list, key=str)])
    else:
        return sns.color_palette([lookup(a) for a in agent_list])

def game_engine():
    # 100 was good enough for the search_bc plot
    TRIALS = 200
    heat_ticks = 5

    opponents = (ag.SelfishAgent(beta=OTHER_BETA), ag.AltruisticAgent(beta=OTHER_BETA))
    ToM = ("self",) + (ag.SelfishAgent(beta=WE_BETA), ag.AltruisticAgent(beta=WE_BETA))
    agents = (ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=ToM),) + opponents

    common_params = dict(
        game="game_engine",
        # game="PublicDonationGame",
        player_types=agents,
        s=1,
        pop_size=10,
        trials=TRIALS,
        stacked=True,
        benefit=10,
        cost=1,
        nactions=2,
        plot_dir=os.path.join(PLOT_DIR, "engine/"),
        memoized=params.memoized,
        deterministic=True,
        graph_kwargs={"color": color_list(agents)},
    )
    
    payoff_params = common_params.copy()
    payoff_params.update(
        stacked=False,
        var='total_payoff',
    )    
    
    ticks = 11
    max_expected_interactions = 9 
    # max_expected_interactions = (1+ticks)/2
    # Expected number of interactions
    def gamma_plot():
        print('Running gamma plot')
        limit_param_plot(
            param_dict={"rounds": range(1, max_expected_interactions+1)},
            tremble=MIN_TREMBLE,
            observability=0,
            legend=True,
            analysis_type="complete",
            file_name="gamma",
            **common_params,
        )
        limit_param_plot(
            param_dict={"rounds": range(1, max_expected_interactions+1)},
            tremble=MIN_TREMBLE,
            observability=0,
            analysis_type="complete",
            file_name="gamma_payoff",
            **payoff_params,
        )        

    def tremble_plot():
        print('Running tremble plot')
        # NOTE: Tremble = 0, beta = inf, and max_players = 3 will cause an error
        # because between three-player subsets there might be some partial
        # observability.
        limit_param_plot(
            param_dict={"tremble": TREMBLE_EXP},
            observability=0,
            rounds=max_expected_interactions,
            analysis_type="complete",
            file_name="tremble",
            **common_params,
        )
        limit_param_plot(
            param_dict={"tremble": TREMBLE_EXP},
            observability=0,
            rounds=max_expected_interactions,
            analysis_type="complete",
            file_name="tremble_payoff",
            **payoff_params,
        )        

    def observe_plot():
        print('Running observability plot')
        limit_param_plot(
            param_dict={"observability": np.round(np.linspace(0, 1, ticks), 2)},
            tremble=MIN_TREMBLE,
            rounds=1,
            analysis_type="complete",
            file_name="observability",
            **common_params,
        )
        limit_param_plot(
            param_dict={"observability": np.round(np.linspace(0, 1, ticks), 2)},
            tremble=MIN_TREMBLE,
            rounds=1,
            analysis_type="complete",
            file_name="observability_payoff",
            **payoff_params,
        )        
        
    def observe_tremble_plot():
        print('Running perception errors')
        limit_param_plot(
            param_dict={"observation_error": TREMBLE_EXP},
            observability=1,
            tremble=0,
            rounds=1,
            analysis_type="complete",
            file_name="percepterror",
            **common_params,
        )
        limit_param_plot(
            param_dict={"observation_error": TREMBLE_EXP},
            observability=1,
            tremble=0,
            rounds=1,
            analysis_type="complete",
            file_name="percepterror_payoff",
            **payoff_params,
        )                    
        
    def heat_map_gamma_omega():
        print('Running heat map gamma vs. omega')
        param_dict = dict(
            rounds=range(1, heat_ticks+1),
            observability=np.round(np.linspace(0, 1, heat_ticks), 2),
        )

        heat_graph_kwargs = dict(
            xlabel = r"Prob. of observation ($\omega$)",
            ylabel = '# of Interactions',
            xy = ("observability", "rounds"),
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
            file_name="gamma_omega",
            **common_params_heat
        )

    def heat_map_gamma_tremble():
        print('Running heat map gamma vs. tremble')
        param_dict = dict(
            rounds=range(1, max_expected_interactions+1, 2),
            tremble=TREMBLE_EXP[::2],
        )

        heat_graph_kwargs = dict(
            xlabel = r"Prob. of action error ($\epsilon$)",
            ylabel = "# Pairwise Interactions",
            xy = ("tremble", "rounds"),
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
            observability=0,
            file_name="gamma_tremble",
            **common_params_heat
        )

    def heat_map_omega_tremble():
        print('Running heat map omega vs. tremble')
        param_dict = dict(
            tremble=TREMBLE_EXP[::2][:-1],
            observability=np.round(np.linspace(0, 1, heat_ticks), 2),
        )

        heat_graph_kwargs = dict(
            xlabel = r"Prob. of action error ($\epsilon$)",
            ylabel = r"Prob. of observation ($\omega$)",
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
            rounds=1,
            file_name="omega_tremble",
            **common_params_heat
        )

    def search_bc():
        print('Running search bc')
        bcs = {
            (15, 1): 4,
            # (10, 1): 4,
            (5, 1): 6,
            (3, 1): 15,
        }

        bc_plot(
            ei_stop=bcs,
            observe_param=np.round(np.linspace(1, 0, heat_ticks), 2),
            delta=1,
            tremble=MIN_TREMBLE,
            analysis_type="complete",
            file_name="bc_plot",
            **common_params)

    # beta_plot()
    gamma_plot()
    tremble_plot()
    observe_plot()
    observe_tremble_plot()
    heat_map_gamma_omega()
    heat_map_gamma_tremble()
    heat_map_omega_tremble()
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


def ipd(game):
    BENEFIT = 3
    
    COST = 1
    # 10000 puts SEM around 0.001-0.002
    TRIALS = 1000
    # TRIALS = 100

    # ToM = ("self",) + (ag.SelfishAgent(beta=WE_BETA), ag.AltruisticAgent(beta=WE_BETA))


    new_pop = old_pop + (
        ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=("self",) + old_pop),
        # ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=ToM),
    )

    common_params = dict(
        game=game,
        s=1,
        benefit=BENEFIT,
        cost=COST,
        pop_size=100,
        analysis_type="limit",
        plot_dir=os.path.join(PLOT_DIR, 'ipd_' + game + '/'),
        trials=TRIALS,
        stacked=True,
        return_rounds=True,
        per_round=True,          
        memoized=params.memoized,  
    )
    
    n_rounds = 25

    payoff_params = common_params.copy()
    payoff_params.update(
        stacked=False,
    )
    # MIN_TREMBLE =0
    print('Running Payoff Heatmap')
    for r in [5, 
              n_rounds]:
        payoff_heatmap(
            rounds=r,
            player_types=new_pop,
            tremble=MIN_TREMBLE,
            sem=False,
            file_name=("payoffs_rounds_{}".format(r)),
            **common_params
        )    
    
    for label, player_types in zip(["wRA", "woRA"], [new_pop, old_pop]):
        print("Running Expected Rounds with", label)
        limit_param_plot(
            param_dict=dict(rounds=[n_rounds]),
            tremble=MIN_TREMBLE,
            player_types=player_types,
            legend=True,
            file_name="rounds_%s_%s" % (game, label),
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
            file_name="popsize_%s_%s" % (game, label),
            graph_kwargs={"color": color_list(player_types),
                        "xlabel": "Population Size"},
            **common_params,
        )

        print("Running Tremble with", label, TREMBLE_EXP)
        limit_param_plot(
            param_dict=dict(tremble=TREMBLE_EXP),
            rounds=n_rounds,
            player_types=player_types,
            file_name="tremble_%s_%s" % (game, label),
            graph_kwargs={"color": color_list(player_types)},
            **common_params,
        )
        
        # print("Running Benefit with", label)
        # limit_param_plot(
        #     param_dict=dict(benefit=[1.1,1.5,2,2.5,3,6]),
        #     rounds=n_rounds,
        #     tremble=MIN_TREMBLE,
        #     player_types=player_types,
        #     file_name="benefit_%s_%s" % (game, label),
        #     graph_kwargs={"color": color_list(player_types)},
        #     **common_params,
        # )
        
    for label, player_types in zip(["wRA", "woRA"], [new_pop, old_pop]):
        print("Running Expected Rounds with", label)
        limit_param_plot(
            param_dict=dict(rounds=[n_rounds]),
            tremble=MIN_TREMBLE,
            var='total_payoff',
            player_types=player_types,
            legend=False,
            file_name="rounds_totalpayoff_%s_%s" % (game, label),
            graph_kwargs={"color": color_list(player_types),
                        "xlabel": "# Pairwise Interactions"},
            **payoff_params,
        )

        print("Running Tremble with", label, TREMBLE_EXP)
        limit_param_plot(
            param_dict=dict(tremble=TREMBLE_EXP),
            rounds=n_rounds,
            var='total_payoff',
            legend=False,
            player_types=player_types,
            file_name="tremble_totalpayoff_%s_%s" % (game, label),
            graph_kwargs={"color": color_list(player_types)},
            **payoff_params,
        ) 

    print('Running Payoffs')
    limit_param_plot(
        param_dict=dict(rounds=[n_rounds]),
        tremble=MIN_TREMBLE,
        player_types=new_pop,
        file_name="%s_rounds_wepay" % game,
        var='wepayoff',
        graph_kwargs={"color": color_list(new_pop),
                      "xlabel": "# Pairwise Interactions"},
        **payoff_params,
    )
    limit_param_plot(
        param_dict=dict(tremble=TREMBLE_EXP),
        rounds=n_rounds,
        player_types=new_pop,
        file_name="%s_tremble_selfpay" % game,
        graph_kwargs={"color": color_list(new_pop)},
        var='selfpayoff',
        **payoff_params,
    )
    limit_param_plot(
        param_dict=dict(beta=np.append(np.linspace(1,6.5,12), np.inf)),
        rounds=n_rounds,
        tremble=MIN_TREMBLE,
        player_types=new_pop,
        file_name="%s_beta_wepay" % game,
        graph_kwargs={"color": color_list(new_pop)},
        var='wepayoff',
        **payoff_params
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
        file_name="%s_cogcosts" % game,
        player_types=new_pop,
        graph_funcs=cog_cost_graph,
        graph_kwargs={"color": color_list(new_pop)},
        **common_params,
    )
       
    # print("Running Beta IPD")
    # limit_param_plot(
    #     param_dict=dict(beta=np.append(np.round(np.linspace(1,6.5,12),1), np.inf)),
    #     tremble=MIN_TREMBLE,
    #     rounds=5,
    #     file_name="%s_beta" % game,
    #     player_types=new_pop,
    #     graph_kwargs={"color": color_list(new_pop),
    #                   "xlabel": r"Softmax ($\beta$)"},
    #     **common_params,        
    # )         
    
    def beta_heat_map():
        print('Running Beat Heat Map')
        heat_graph_kwargs = dict(
                xlabel = r'Softmax ($\beta$)',
                ylabel = '# Pairwise Interactions',
                xy = ("beta", "rounds"),
                onlyRA = True)
        
        param_dict = dict(
            beta=np.append(np.round(np.linspace(1,6.5,12),1), np.inf),
            rounds=[10],
        )
        
        params_heat(
            param_dict=param_dict,
            player_types=new_pop,
            tremble=MIN_TREMBLE,
            file_name="%s_heat_beta" % game,
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
            file_name="heat_tremble",
            line = True,
            graph_kwargs=heat_graph_kwargs,
            **common_params
        )
        
    def benefit_heat_map():
        print('Running benefit Heat Map')
        heat_graph_kwargs = dict(
                xlabel = r"B",
                ylabel = '# Pairwise Interactions',
                xy = ("benefit", "rounds"),
                onlyRA = True)
        
        param_dict = dict(
            benefit=[3,5,7,9],
            rounds=[n_rounds],
        )
        
        params_heat(
            param_dict=param_dict,
            tremble=MIN_TREMBLE,
            player_types=new_pop,
            file_name="heat_benefit",
            line = True,
            graph_kwargs=heat_graph_kwargs,
            **excluding_keys(common_params, "benefit")
        )
        
    def benefit_beta_map():
        print('Running benefit Heat Map')
        heat_graph_kwargs = dict(
                xlabel = r"B",
                ylabel = 'Beta',
                xy = ("benefit", "beta"),
                onlyRA = True)
        
        param_dict = dict(
            benefit=[1.5,3,4,5,6],
            beta=[1,2,3],
        )
        
        params_heat(
            param_dict=param_dict,
            rounds=n_rounds,
            tremble=MIN_TREMBLE,
            player_types=new_pop,
            file_name="heat_benefit_beta",
            line = True,
            graph_kwargs=heat_graph_kwargs,
            **excluding_keys(common_params, "benefit")
        )        
            
    benefit_heat_map()        
    tremble_heat_map()
    beta_heat_map()    
    benefit_beta_map()
  

def agent():
    opponents = (ag.AltruisticAgent(beta=OTHER_BETA), ag.SelfishAgent(beta=OTHER_BETA))
    ToM = ("self",) + (ag.AltruisticAgent(beta=WE_BETA), ag.SelfishAgent(beta=WE_BETA))
    agents = (ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=ToM),) + opponents
    
    observer = ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=ToM)
    
    common_params = dict(
        observer=observer,
        plot_dir=os.path.join(PLOT_DIR, "scenarios/"),
        trials=1,
        color=color_list(agents, sort=False),
    )

    game_params = dict(tremble=MIN_TREMBLE, benefit=3, cost=1)

    # SCENARIO PLOTS

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
            scenarios=[obs],
            xlabel="$P(U_{" + xlabel + "})$",
            file_name="scene_reciprocal_{}_{}".format(s[1], xlabel),
            **common_params,
        )

    scenarios = ["Prior", "D", "CD", "CCD", "CCCD", "CCCCD"]
    
    forgive_plot(
        scenarios=scenarios,
        p = 'belief',
        game_params = game_params,
        file_name = 'forgive_belief',
        label = 'Belief',
        **common_params
    )

    forgive_plot(
        scenarios=scenarios,
        p = 'decision',
        game_params = game_params,
        file_name = 'forgive_act',
        label = 'Probability of Cooperate',
        **common_params
    )
    
    decision_plot(
        betas = [3],
        game_params = game_params,
        file_name = 'decision_beta',
        **common_params        
    )


def belief():
    everyone = (ag.AltruisticAgent(beta=OTHER_BETA), ag.SelfishAgent(beta=OTHER_BETA))
    ToM = ("self",) + (ag.AltruisticAgent(beta=WE_BETA), ag.SelfishAgent(beta=WE_BETA))
    agent = ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=ToM)

    common_params = dict(
        believer=agent,
        opponent_types=(agent,) + everyone,
        believed_types=(agent,) + ToM[1:],
        tremble=MIN_TREMBLE,
        plot_dir=os.path.join(PLOT_DIR, 'belief/'),
        deterministic=True,
        game="game_engine",
        # Max Players = 2 so we can more easily interpret the results
        max_players=2,
        benefit=3,
        cost=1,
        traces=5,
        trials=100,
        memoized=params.memoized,
        colors=color_list((agent,) + everyone, sort=False),
    )

    population = np.array([4, 3, 3])

    # Private Interactions. Not we do not set the experiment or
    # population for this experiment.
    print('Running private interactions')
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
    print('Running public interactions')
    plot_beliefs(
        experiment=population_beliefs,
        population=population,
        rounds=1,
        observability=1,
        file_name="intra_gen_belief_public",
        xlabel = "# Obersevations",
        **common_params,
    )

    for g in ['direct', 'direct_seq']:
        # FSA agents
        print('Running FSA agents in game %s' % g)
        ToM = ("self",) + old_pop
        agent = ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=ToM)
        common_params.update(
            dict(
                believer=agent,
                opponent_types=(agent,) + old_pop,
                believed_types=(agent,) + old_pop,
                game=g,
                colors=color_list((agent,) + old_pop, sort=False),
            )
        )

        plot_beliefs(
            observability=0,
            rounds=15,
            file_name="%s_belief" % g,
            xlabel="# Pairwise Interactions",
            **common_params,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--belief", action="store_true")
    parser.add_argument("--ipd", action="store_true")
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--engine", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.all:
        game_engine()
        belief()
        ipd('direct')
        ipd('direct_seq')
        agent()

    if args.belief:
        belief()

    if args.ipd:
        ipd('direct')
        ipd('direct_seq')

    if args.agent:
        agent()

    if args.engine:
        game_engine()

    if args.debug:
        import pdb; pdb.set_trace()
        
if __name__ == "__main__":
    main()
