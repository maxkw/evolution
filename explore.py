import inspect
plot_dir = "./plots/"+inspect.stack()[0][1][:-3]+"/"
from utils import splits
from evolve import evo_analysis
from itertools import product
import pandas as pd
import seaborn as sns
import agents as ag
import numpy as np
from games import AnnotatedGame, IndefiniteMatchup, AllNoneObserve, literal
from evolve import ssd_param_search
from experiment_utils import plotter
import matplotlib.pyplot as plt


def grid_v_param(f, row_param, row_vals,
                 #col_param, col_vals,
                 #hue_param, hue_vals,
                 **kwargs):
    dfs = []
    #for row_val, col_val, hue_val in product(row_vals, col_vals, hue_vals):
    for row_val in row_vals:
        dfs.append(f(**dict(kwargs,**{
            row_param       :row_val,
            #col_param       :col_val,
            #hue_param       :hue_val
            })))
    return pd.DataFrame(dfs)

def param_sweep(f,params,**kwargs):
    dfs = []
    keys = params.keys()
    for vals in product(*[params[k] for k in keys]):
        dfs.append(f(**dict(kwargs,**dict(zip(keys,vals)))))
    return pd.DataFrame(dfs)


@plotter(grid_v_param, plot_exclusive_args = ['data', 'stacked'])
def grid_param_plot(row_param,# col_param,
                    #hue_param,
                    data = [], **kwargs):
    #d = data.query('"WeAgent()" == type')
    #sns.factorplot(y = 'proportion', hue = hue_param, x = param, col = col_param, row = row_param, data = d)
    ax = sns.factorplot(y = "observability", x = "expected_interactions", hue = 'type', data = data)
    #ax = data.plot(y = "observability", x = "expected_interactions")
    plt.ylim([0,1])
    print data
    import pdb;pdb.set_trace()
    return data, ax

@literal
def dd_ci_sa(expected_interactions = 1, observability = 0, cost = 1, benefit = 3, intervals = 1, tremble = .1, variance = 1, **kwargs):
    assert intervals>=0

    def Gen():
        N_actions = intervals-1
        N_players = np.random.choice([2,3])
        t = tremble#np.random.beta(tremble, 10)
        # initialize set of choices with the zero-action

        costs = sorted([np.random.poisson(cost) for _ in range(intervals-1)])
        weights = sorted([np.random.exponential(benefit) for _ in range(intervals-1)])

        choices = [np.zeros(N_players)]
        for c,w in zip(costs,weights):
            for p in xrange(1,N_players):
                choice = np.zeros(N_players)
                choice[0] = -c
                choice[p] = w*c
                choices.append(copy(choice))
        decision = Decision(OrderedDict((str(p),p) for p in choices))
        decision.tremble = t

        return decision

    dictator = Dynamic(Gen)
    dictator.name = "dd_ci_va"
    gamma = 1-1/expected_interactions
    game = AnnotatedGame(IndefiniteMatchup(gamma, AllNoneObserve(observability, dictator)))
    return game

@literal
def dd_ci_pa(expected_interactions = 1, observability = 0, cost = 1, benefit = 3, intervals = 1, tremble = .1, **kwargs):
    assert intervals>=0

    def Gen():
        N_players = np.random.choice([2,3])
        c = np.random.poisson(cost)
        b = np.random.exponential(benefit)*c
        t = tremble#np.random.beta(tremble, 10)

        max_d = benefit - cost
        max_r = cost / benefit

        ratios = np.linspace(0, max_r, intervals)
        differences = np.linspace(0, max_d, intervals)

        def new_cost(r,d):
            return d/(1-r)-d
        def new_benefit(r,d):
            return d/(1-r)

        payoffs = []
        for i,(d,r) in enumerate(zip(differences,ratios)):
            if i == 0:
                payoffs.append(np.zeros(N_players))
            else:
                nc = new_cost(r,d)
                nb = new_benefit(r,d)*nc
                for p in xrange(1,N_players):
                    choice = np.zeros(N_players)
                    choice[0] = -nc
                    choice[p] = nb*nc
                    payoffs.append(copy(choice))

        decision = Decision(OrderedDict((str(p),p) for p in payoffs))
        decision.tremble = t
        return decision

    dictator = Dynamic(Gen)
    dictator.name = "dd_ci_va"
    gamma = 1-1/expected_interactions
    game = AnnotatedGame(IndefiniteMatchup(gamma, AllNoneObserve(observability, dictator)))
    return game

@literal
def dd_ci_pas(expected_interactions = 1, observability = 0, cost = 1, benefit = 3, intervals = 1, tremble = .1, **kwargs):
    assert intervals>=0

    def Gen():
        N_players = np.random.choice([2,3])
        c = cost#np.random.poisson(cost)
        b = benefit*c#np.random.exponential(benefit)*c
        t = tremble#np.random.beta(tremble, 10)

        max_d = benefit - cost
        max_r = cost / benefit

        ratios = np.linspace(0, max_r, intervals)
        differences = np.linspace(0, max_d, intervals)

        def new_cost(r,d):
            return d/(1-r)-d
        def new_benefit(r,d):
            return d/(1-r)

        payoffs = []
        for i,(d,r) in enumerate(zip(differences,ratios)):
            if i == 0:
                payoffs.append(np.zeros(N_players))
            else:
                nc = np.random.poisson(new_cost(r,d))
                nb = np.random.exponential(new_benefit(r,d))*nc
                for p in xrange(1,N_players):
                    choice = np.zeros(N_players)
                    choice[0] = -nc
                    choice[p] = nb
                    payoffs.append(copy(choice))

        decision = Decision(OrderedDict((str(p),p) for p in payoffs))
        decision.tremble = t
        return decision

    dictator = Dynamic(Gen)
    dictator.name = "dd_ci_va"
    gamma = 1-1/expected_interactions
    game = AnnotatedGame(IndefiniteMatchup(gamma, AllNoneObserve(observability, dictator)))
    return game

def compare_slopes():
    opponents = (ag.SelfishAgent,ag.AltruisticAgent)
    ToM = ('self',)+opponents
    WE = ag.WeAgent(agent_types=ToM)
    agents = (WE,)+opponents
    common_params = dict(s=.5,
                         #game='dynamic',
                         game = "game_engine",
                         #game = 'direct',
                         #game = 'dd_ci_va',
                         player_types = agents,
                         #analysis_type = 'limit',
                         #beta = 5,
                         pop_size = 10,
                         #benefit = 10,
                         tremble = 0,
                         observability = 0,
                         trials = 100,
                         benefit = 10,
                         param = "observability",
                         param_lim = [0,1],
                         expected_interactions = 1,
                         target = WE,
                         param_tol = .05,
                         #parallelized = False,
                         mean_tol = .1)
    d = ssd_param_search(**common_params)
    print d
    import pdb; pdb.set_trace()

    name = "topographic"
    grid_param_plot(
        f = ssd_param_search,
        row_param = "expected_interactions", row_vals = np.linspace(1,5,splits(1)),
        file_name = name,
        extension = '.png',
        plot_dir = plot_dir,
        **common_params)

def main():
    compare_slopes()


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
BETA = 5
PRIOR = 0.5


def color_list(agent_list, sort = True):
    '''takes a list of agent types `agent_list` and returns the correctly
    ordered color mapping for plots
    '''
    def lookup(a):
        a = str(a)
        if 'WeAgent' in a: return 'C0'
        if 'ReciprocalAgent' in a: return 'C0'
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

def belief():
    everyone = (ag.AltruisticAgent(beta = BETA), ag.SelfishAgent(beta = BETA))
    A = ag.WeAgent
    A = ag.ReciprocalAgent
    agent = A(prior = PRIOR, beta = BETA, agent_types = ('self',) + everyone)

    plot_beliefs(agent, (agent,)+everyone, (agent,)+everyone,
                 #population = [3,3,3],
                 #population = [
                 #    3,3,3
                 #    #5,5,5
                 #],
                 #experiment = population_beliefs,
                 tremble = 0.0,
                 plot_dir = PLOT_DIR,
                 #deterministic = True,
                 game = 'belief_game',
                 # benefit = 10,
                 rounds = 1,
                 observability = 1,
                 file_name = "intra_gen_belief",
                 #deterministic = True,
                 extension = '.png',
                 traces = 0,
                 trials = 100,
                 colors = color_list((agent,)+everyone, sort = False))

def game_engine():
    TRIALS = 10#200
    
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
        #benefit = 3,
        plot_dir = PLOT_DIR,
        observability = 0,
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
    def gamma_plot():
        limit_param_plot(
            param = 'expected_interactions',
            param_vals = np.round(np.linspace(1, 10, splits(4)), 2),
            tremble = 0.0, 
            analysis_type = 'limit',
            file_name = 'game_engine_gamma',
            **common_params)

    def tremble_plot():
        # Vary tremble
        limit_param_plot(
            param = 'tremble',
            param_vals = np.round(np.linspace(0, 1, ticks), 2),
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

    gamma_plot()

def gaussian_test():
    from scipy.ndimage.filters import gaussian_filter1d as gfilter
    a = np.array([[0.0,1,0],
                  [0,1,0],
                  [1,0,0],
                  [0,0,1]])
    print a
    print np.round(gfilter(a,.5,0),2)


def compare():
    from games import BinaryDictator
    bd = BinaryDictator(cost = 1, benefit = 10)
    def observation(participant_ids, action, game = bd, observer_ids = None, tremble = 0):
        if observer_ids == None:
            observer_ids = participant_ids
        return dict(locals(),**dict(observer_ids = frozenset(observer_ids)))

    def PD_obs(actions, participants = "AB", tremble = 0):
        return [observation(participants, actions[0], bd, participants, tremble),
                observation(list(reversed(participants)), actions[1], bd, participants, tremble)]

    tft_obs = [PD_obs(["keep","give"]), PD_obs(["give","keep"]), PD_obs(["give","give"])]

    def test_agent(RA):
        genome = dict(type = RA, agent_types = [RA, ag.SelfishAgent, ag.TFT], prior = .33, beta = 5)
        agent = RA(genome, "A")
        print "\n\n\n", RA
        print agent.belief['B']
        for o in tft_obs:
            for os in o:
                #print os
                #print "l", agent.likelihood_of(**os)
                pass
            agent.observe(o)
            print agent.belief['B']

    RAs = [ag.WeAgent, ag.ReciprocalAgent]

    for RA in RAs:
        test_agent(RA)

if __name__ == "__main__":
    #compare()
    belief()
    #game_engine()
    #gaussian_test()
    #a = foo("a","ab")
    #print a.name
    #print a.me
    #print [i.me for i in a.lattice]
    #print bar().like()
    #main()
    #test()


