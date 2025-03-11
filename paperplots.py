import argparse
import numpy as np
import seaborn as sns
import agents as ag
from evolve import limit_param_plot, params_heat, bc_plot, payoff_plot, params_dom_heat
from experiments import plot_beliefs, population_beliefs
from scenarios import scene_plot, make_observations_from_scenario, forgive_plot
from utils import excluding_keys
import params
import os
from icecream import install
install()

sns.set_context("paper", font_scale=1.5)
PLOT_DIR = "./plots/"
WE_BETA = np.inf
# NOTE: Setting OTHER_BETA=inf makes tremble=0 blow up
OTHER_BETA = np.inf
PRIOR = 0.5
TREMBLE_EXP = np.round(np.linspace(0.025, 0.4, 16), 3)
BENEFIT_EXP = np.round(np.linspace(1.1, 3, 20),2)
MIN_TREMBLE = TREMBLE_EXP[0]
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

def color_list(agent_list, sort=True, error_on_unknown=True):
    """takes a list of agent types `agent_list` and returns the correctly
    ordered color mapping for plots
    """
    increment = 8
    
    def lookup(a):
        nonlocal increment
        
        a = str(a)
        if "WeAgent" in a:
            return "C0"
        if "AltruisticAgent" in a or "AllC" in a:
            return "C2"
        if "SelfishAgent" in a or "AllD" in a:
            return "C3"
        if "FSAgent" in a:
            return "C4"
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
        if "Grim" in a:
            return "C8"
        
        if error_on_unknown:
            raise "Color not defined for agent %s"
        else:
            increment += 1
            return "C" + str(increment)

    if sort:
        return sns.color_palette([lookup(a) for a in sorted(agent_list, key=str)])
    else:
        return sns.color_palette([lookup(a) for a in agent_list])

def game_engine():
    TRIALS = 200
    heat_ticks = 5

    opponents = (ag.SelfishAgent(beta=OTHER_BETA), ag.AltruisticAgent(beta=OTHER_BETA))
    ToM = ("self",) + (ag.SelfishAgent(beta=WE_BETA), ag.AltruisticAgent(beta=WE_BETA))
    agents = (ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=ToM),) + opponents

    common_params = dict(
        game="game_engine",
        player_types=agents,
        s=2,
        pop_size=10,
        trials=TRIALS,
        stacked=True,
        benefit=5,
        cost=1,
        nactions=2,
        plot_dir=os.path.join(PLOT_DIR, "engine/"),
        memoized=params.memoized,
        deterministic=True,
        analysis_type="complete",
        add_payoff=False,
        graph_kwargs={"color": color_list(agents)},
    )
    
    payoff_params = common_params.copy()
    payoff_params.update(
        stacked=False,
        var='total_payoff',
    )    
    
    ticks = 11
    max_expected_interactions = 9 

    # def FS_gamma():
    #     print('Running FS gamma')
    #     common_params_FS = common_params.copy()
    #     agents = (ag.SelfishAgent(beta=OTHER_BETA), ag.FSAgent(beta=OTHER_BETA, w_aia=0.5, w_dia=0.5))
    #     common_params_FS['player_types'] = agents 
    #     common_params_FS['graph_kwargs']['color'] = color_list(agents)

    #     limit_param_plot(
    #         param_dict={"rounds": range(1, max_expected_interactions+1)},
    #         tremble=MIN_TREMBLE,
    #         observability=0,
    #         file_name="FS_gamma",
    #         **common_params_FS,
    #     )
    # FS_gamma()

    # Expected number of interactions
    def gamma_plot():
        print('Running gamma plot')
        limit_param_plot(
            param_dict={"rounds": range(1, max_expected_interactions+1)},
            tremble=MIN_TREMBLE,
            observability=0,
            # legend='center right',
            file_name="gamma",
            **common_params,
        )

    def prior_plot():
        print('Running prior plot')

        # limit_param_plot(
        #     param_dict={"prior": np.round(np.linspace(0, 1, ticks), 2)[1:-1]},
        #     observability=0,
        #     rounds=max_expected_interactions,
        #     tremble=MIN_TREMBLE,
        #     file_name="prior",
        #     **common_params,
        # )
        graph_kwargs = common_params['graph_kwargs'].copy()
        graph_kwargs['xlabel'] = "Prior Belief"
        limit_param_plot(
            param_dict={"prior": np.round(np.linspace(0, 1, ticks), 2)[1:-1]},
            observability=1,
            rounds=1,
            tremble=MIN_TREMBLE,
            file_name="prior_indirect",
            graph_kwargs=graph_kwargs,
            **excluding_keys(common_params, 'graph_kwargs'),
        )      
        payoff_plot(
            param_dict={"prior": np.round(np.linspace(0, 1, ticks), 2)[1:-1]},
            observability=1,
            tremble=MIN_TREMBLE,
            rounds=1,
            var='total_payoff',
            file_name="totalpayoff_prior",
            graph_kwargs={"xlabel": r"Prior Belief",
                          "ylabel": 'Payoff',
                          "xlim": (0,1.02),
                          "ylim": (0,2.5),
                          "xticks": [0, .2, .4, .6, .8, 1],
                          "yticks": [0, .5, 1, 1.5, 2, 2.5],},
            **excluding_keys(common_params, 'graph_kwargs'),            
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
            file_name="tremble",
            **common_params,
        )          

    def observe_plot():
        print('Running observability plot')
        limit_param_plot(
            param_dict={"observability": np.round(np.linspace(0, 1, ticks), 2)},
            tremble=MIN_TREMBLE,
            rounds=1,
            file_name="observability",
            **common_params,
        )
        payoff_plot(
            param_dict={"observability": np.round(np.linspace(0, 1, ticks), 2)},
            tremble=MIN_TREMBLE,
            rounds=1,
            var='total_payoff',
            file_name="totalpayoff_observe",
            graph_kwargs={"xlabel": r"Prob. observation ($\omega$)",
                          "ylabel": 'Payoff',
                          "xlim": (0,1.02),
                          "ylim": (0,2.5),
                          "xticks": [0, .2, .4, .6, .8, 1],
                          "yticks": [0, .5, 1, 1.5, 2, 2.5],},
            **excluding_keys(common_params, 'graph_kwargs'),            
        )
        
    def observe_tremble_plot():
        print('Running perception errors')
        limit_param_plot(
            param_dict={"observation_error": np.append(0,TREMBLE_EXP[1::2]/2)},
            observability=1,
            tremble=MIN_TREMBLE,
            rounds=1,
            file_name="percepterror",
            **common_params,
        )
        payoff_plot(
            param_dict={"observation_error": np.append(0,TREMBLE_EXP[1::2]/2)},
            tremble=MIN_TREMBLE,
            observability=1,
            rounds=1,
            var='total_payoff',
            file_name="totalpayoff_percepterror",
            graph_kwargs={"xlabel": r"Prob. observation error",
                          "ylabel": 'Payoff',
                          "xlim": (0,TREMBLE_EXP[1::2][-1]/2+.02),
                          "ylim": (0,2.5),
                          "xticks": [0, .05, .1, .15, .2],
                          "yticks": [0, .5, 1, 1.5, 2, 2.5],},
            **excluding_keys(common_params, 'graph_kwargs'),            
        )
        
    def heat_map_gamma_omega(var):
        print('Running heat map gamma vs. omega')
        param_dict = dict(
            rounds=range(1, heat_ticks+1),
            observability=np.round(np.linspace(0, 1, heat_ticks), 2),
        )

        heat_graph_kwargs = dict(
            xlabel = r"Prob. observation ($\omega$)",
            ylabel = 'Game length',
            xy = ("observability", "rounds"),
            var=var,
            who = 'RA',
        )
        common_params_heat = common_params.copy()
        common_params_heat['graph_kwargs'] = {
            **common_params_heat['graph_kwargs'],
            **heat_graph_kwargs
        }

        params_heat(
            param_dict=param_dict,
            tremble=MIN_TREMBLE,
            var=var,
            file_name="gamma_omega_%s" % (var),
            **common_params_heat
        )
        
    def heat_map_gamma_prior(var):
        print('Running heat map gamma vs. prior')
        param_dict = dict(
            rounds=range(1, heat_ticks+1),
            prior=np.round(np.linspace(0, 1, ticks), 2)[1:-1],
        )

        heat_graph_kwargs = dict(
            xlabel = r"Prior Belief",
            ylabel = 'Game length',
            xy = ("prior", "rounds"),
            var=var,
            who = 'RA',
        )
        common_params_heat = common_params.copy()
        common_params_heat['graph_kwargs'] = {
            **common_params_heat['graph_kwargs'],
            **heat_graph_kwargs
        }

        params_heat(
            param_dict=param_dict,
            tremble=MIN_TREMBLE,
            observability=0,
            var=var,
            file_name="gamma_prior_%s" % (var),
            **common_params_heat
        )
        

    def heat_map_FS(var, with_RA=False):
        print('Running heat map FS')
        param_dict = dict(
            w_aia=np.round(np.linspace(0, 1, heat_ticks), 2),
            w_dia=np.round(np.linspace(0, 1, heat_ticks), 2),
        )

        heat_graph_kwargs = dict(
            xlabel = r"DIA ($\alpha$)",
            ylabel = r"AIA ($\beta$)",
            xy = ("w_aia", "w_dia"),        
            var=var,
            who = 'all',
        )

        if with_RA:
            opponents = (
                ag.SelfishAgent(beta=OTHER_BETA), 
                ag.AltruisticAgent(beta=OTHER_BETA), 
                ag.FSAgent(beta=OTHER_BETA, w_aia=None, w_dia=None)
                )
            ToM = ("self",) + (ag.SelfishAgent(beta=WE_BETA), 
                               ag.AltruisticAgent(beta=WE_BETA), 
                               ag.FSAgent(beta=WE_BETA, w_aia=None, w_dia=None))
            
            agents = (ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=ToM),) + opponents
        else:
            agents = (ag.SelfishAgent(beta=OTHER_BETA), ag.AltruisticAgent(beta=OTHER_BETA), ag.FSAgent(beta=OTHER_BETA, w_aia=None, w_dia=None))

        common_params_heat = common_params.copy()
        common_params_heat['graph_kwargs'] = {
            **common_params_heat['graph_kwargs'],
            **heat_graph_kwargs
        }
        common_params_heat['player_types'] = agents
                
        params_heat(
            param_dict=param_dict,
            tremble=MIN_TREMBLE,
            observability=0,
            rounds=max_expected_interactions,
            file_name="FS_%s_%s" % (var, with_RA),
            **common_params_heat
        )


    def heat_map_gamma_tremble(var):
        print('Running heat map gamma vs. tremble')
        param_dict = dict(
            rounds=range(1, max_expected_interactions+1),
            tremble=TREMBLE_EXP[::2],
        )

        heat_graph_kwargs = dict(
            xlabel = r"Prob. action error ($\epsilon$)",
            ylabel = "Game length",
            xy = ("tremble", "rounds"),        
            var=var,
            who = 'RA',
        )
        common_params_heat = common_params.copy()
        common_params_heat['graph_kwargs'] = {
            **common_params_heat['graph_kwargs'],
            **heat_graph_kwargs
        }

        params_heat(
            param_dict=param_dict,
            tremble=MIN_TREMBLE,
            observability=0,
            file_name="gamma_tremble_%s" % (var),
            **common_params_heat
        )

    def heat_map_omega_tremble(var):
        print('Running heat map omega vs. tremble')
        param_dict = dict(
            tremble=TREMBLE_EXP[::2][:-1],
            observability=np.round(np.linspace(.6, 1, heat_ticks), 2),
        )

        heat_graph_kwargs = dict(
            xlabel = r"Prob. action error ($\epsilon$)",
            ylabel = r"Prob. observation ($\omega$)",
            xy = ("tremble", "observability"),
            var=var,
            who = 'RA',
        )
        common_params_heat = common_params.copy()
        common_params_heat['graph_kwargs'] = {
            **common_params_heat['graph_kwargs'],
            **heat_graph_kwargs
        }

        params_heat(
            param_dict=param_dict,
            tremble=MIN_TREMBLE,
            rounds=1,
            file_name="omega_tremble_%s" % (var),
            **common_params_heat
        )

    def search_bc():
        print('Running search bc')
        bcs = {
            (10, 1): 4,
            (5, 1): 6,
            (3, 1): 6,
        }

        bc_plot(
            ei_stop=bcs,
            observe_param=np.round(np.linspace(1, 0, heat_ticks), 2),
            delta=1,
            tremble=MIN_TREMBLE,
            file_name="bc_plot",
            **common_params)


    
    prior_plot()
    for var in ['proportion', 'total_payoff']:
        heat_map_gamma_prior(var)
        heat_map_FS(var, with_RA=True)
        heat_map_FS(var, with_RA=False)
        heat_map_gamma_tremble(var)
        heat_map_gamma_omega(var)
        heat_map_omega_tremble(var)

    observe_plot()
    gamma_plot()
    tremble_plot()
    observe_tremble_plot()
    search_bc()
    
def ipd(game):
    BENEFIT = 3
    COST = 1

    TRIALS = 1000

    new_pop = old_pop + (
        ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=("self",) + old_pop),
    )
    mem1_autonoma = tuple(ag.all_automata)
    mem1_pop = mem1_autonoma + (ag.WeAgent(prior=PRIOR, beta=WE_BETA, agent_types=("self",) + mem1_autonoma),)    

    common_params = dict(
        game=game,
        s=.3,
        pop_size=100,   
        benefit=BENEFIT,
        add_payoff=False,
        cost=COST,
        analysis_type="limit",
        plot_dir=os.path.join(PLOT_DIR, 'ipd_' + game + '/'),
        trials=TRIALS,
        stacked=True,
        return_rounds=True,
        per_round=True,          
        memoized=params.memoized,  
    )
    n_rounds = 49
    
    # from experiments import payoff_heatmap
    # def FS_heatmap():
    #     player_types = old_pop

    #     payoff_heatmap(player_types, 
    #                    1, 
    #                    file_name="FS_heatmap",
    #                    tremble=MIN_TREMBLE,
    #                    **common_params) 
    # FS_heatmap()    
    
    # def FS_ipd():
    #     print('Running FS gamma')
    #     player_types = (ag.AllD, ag.FSAgent(beta=OTHER_BETA, w_aia=0.5, w_dia=0.5))
    #     limit_param_plot(
    #         param_dict=dict(rounds=[n_rounds]),
    #         tremble=MIN_TREMBLE,
    #         player_types=player_types,
    #         # legend='center right',
    #         file_name="FS",
    #         graph_kwargs={"color": color_list(player_types, error_on_unknown=False),
    #                     "xlabel": "Game length"},
    #         **common_params,    
    #     ) 
    # # FS_ipd()    
    
    # for label, player_types in zip(["wRA", "woRA"], [new_pop, old_pop]):
    #     print("Running Expected Rounds with", label)
    #     limit_param_plot(
    #         param_dict=dict(rounds=[n_rounds]),
    #         tremble=MIN_TREMBLE,
    #         player_types=player_types,
    #         # legend='center right',
    #         file_name="rounds_%s" % (label),
    #         graph_kwargs={"color": color_list(player_types, error_on_unknown=False),
    #                     "xlabel": "Game length"},
    #         **common_params,    
    #     ) t
    
    def pop_heat_map(player_types = ('new_pop', new_pop)):
        print('Running Pop Heat Map')
        heat_graph_kwargs = dict(
                xlabel = r"Population Size",
                ylabel = 'Game length',
                xy = ("pop_size", "rounds"))
        
        param_dict = dict(
            rounds=[n_rounds],
            pop_size=list(range(2,51,4))
        )
        if player_types[0] == 'old_pop' or player_types[0] == 'new_pop':
            heat_graph_kwargs['color'] = color_list(player_types[1])
            params_dom_heat(
                param_dict=param_dict,
                player_types=player_types[1],
                tremble = MIN_TREMBLE,
                file_name="pop_size_%s" % (player_types[0]),
                graph_kwargs=heat_graph_kwargs,
                **common_params
            )

        for var in ['proportion', 'total_payoff']:
            heat_graph_kwargs['var'] = var
            heat_graph_kwargs['who'] = 'nice'
        
            params_heat(
                param_dict=param_dict,
                player_types=player_types[1],
                line=False,
                tremble=MIN_TREMBLE,
                file_name="RAheat_popsize_%s_%s" % (player_types[0], var),
                graph_kwargs=heat_graph_kwargs,
                **common_params
            )        
            
    
    def tremble_heat_map(player_types = ('new_pop', new_pop)):
        print('Running Tremble Heat Map')
        heat_graph_kwargs = dict(
                xlabel = r"Prob. action error ($\epsilon$)",
                ylabel = 'Game length',
                xy = ("tremble", "rounds"))
        
        param_dict = dict(
            tremble=TREMBLE_EXP,
            rounds=[n_rounds],
        )
        if player_types[0] == 'old_pop' or player_types[0] == 'new_pop':
            heat_graph_kwargs['color'] = color_list(player_types[1])
            params_dom_heat(
                param_dict=param_dict,
                player_types=player_types[1],
                file_name="heat_tremble_%s" % (player_types[0]),
                graph_kwargs=heat_graph_kwargs,
                **common_params
            )

        for var in ['proportion', 'total_payoff']:
            heat_graph_kwargs['var'] = var
            heat_graph_kwargs['who'] = 'nice'
        
            params_heat(
                param_dict=param_dict,
                player_types=player_types[1],
                line=False,
                file_name="RAheat_tremble_%s_%s" % (player_types[0], var),
                graph_kwargs=heat_graph_kwargs,
                **common_params
            )        
        
        
        
    def benefit_heat_map(player_types = ('new_pop', new_pop)):
        print('Running benefit Heat Map')
        heat_graph_kwargs = dict(
                xlabel = r"benefit / cost",
                ylabel = 'Game length',
                xy = ("benefit", "rounds"))
        
        param_dict = dict(
            benefit=BENEFIT_EXP,
            rounds=[n_rounds],
        )
        if player_types[0] == 'old_pop' or player_types[0] == 'new_pop':
            heat_graph_kwargs['color'] = color_list(player_types[1])
            params_dom_heat(
                param_dict=param_dict,
                tremble=MIN_TREMBLE,
                player_types=player_types[1],
                file_name="heat_benefit_%s" % (player_types[0]),
                graph_kwargs=heat_graph_kwargs,
                **excluding_keys(common_params, "benefit")
            )
            
        for var in ['proportion', 'total_payoff']:
            if 'old' in player_types[0] and var=='proportion': 
                pass                    
            heat_graph_kwargs['var'] = var
            heat_graph_kwargs['who'] = 'nice'
            
            params_heat(
                param_dict=param_dict,
                tremble=MIN_TREMBLE,
                player_types=player_types[1],
                line=False,
                file_name="RAheat_benefit_%s_%s" % (player_types[0], var),
                graph_kwargs=heat_graph_kwargs,
                **excluding_keys(common_params, "benefit")
            )                

    # pop_heat_map(('new_pop', new_pop))
    # pop_heat_map(('old_pop', old_pop))
    # pop_heat_map(('old_mem1', mem1_autonoma))
    # pop_heat_map(('mem1', mem1_pop))

    benefit_heat_map(('new_pop', new_pop))
    benefit_heat_map(('old_pop', old_pop))
    tremble_heat_map(('new_pop', new_pop))
    tremble_heat_map(('old_pop', old_pop))   
    benefit_heat_map(('old_mem1', mem1_autonoma)) 
    tremble_heat_map(('old_mem1', mem1_autonoma))          
    tremble_heat_map(('mem1', mem1_pop))
    benefit_heat_map(('mem1', mem1_pop))          
 


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
    # import pdb; pdb.set_trace()
    game_params = dict(tremble=MIN_TREMBLE, benefit=3, cost=1)

    # SCENARIO PLOTS

    scenarios = [
        ('Alice', [["AB"], "C"]),
        ('Alice', [["AB"], "D"]),  
        ('Alice', [["BA", "AB"], "DD"]),
        ('Alice', [["BA", "AB"], "DC"]),
        ('Alice', [["BA", "AB"], "CC"]),
        ('Alice', [["BA", "AB"], "CD"]),
        ('Alice', [["prior"], "prior"]),
        # ('Bob', [["prior"], "prior"])
    ]
        
    # solo_legend()   

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

    scenarios = ["Prior", "D", "CD", "CCD", "CCCD"]
    
    forgive_plot(
        scenarios=scenarios,
        p = 'belief',
        game_params = game_params,
        file_name = 'forgive_belief',
        label = 'Belief',
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
        show_legend=False,
        game="game_engine",
        # Max Players = 2 so we can more easily interpret the results
        max_players=2,
        cost=1,
        traces=20,
        trials=1000,
        memoized=params.memoized,
        colors=color_list((agent,) + everyone, sort=False),
    )

    population = np.array([4, 4, 4])

    # Private Interactions. Not we do not set the experiment or
    # population for this experiment.
    print('Running private interactions')
    plot_beliefs(
        observability=0,
        # Below computes makes it so the number of interactions are
        # equivalent in both cases. 
        rounds=int(sum(population) * (sum(population) - 1) / 2 / 3),
        xlabel="Interaction #",
        benefit=10,
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
        benefit=10,
        legend=False,
        file_name="intra_gen_belief_public",
        xlabel = "Observation #",
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
            benefit=3,
            rounds=20,
            file_name="%s_belief" % g,
            xlabel="Game length",
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
        import pdb
        pdb.set_trace()
        
if __name__ == "__main__":
    main()
