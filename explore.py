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
    #sns.factorplot(y = "observability", x = "expected_interactions", hue = hue_param, data = data)
    ax = data.plot(y = "observability", x = "expected_interactions")
    plt.ylim([0,1])
    return ax

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
                         #stacked = True,
                         #parallelized = False,
                         #expected_interactions = 1
    #                     )
                         #col_param = "beta", col_vals = (5,),
                         row_param = "expected_interactions", row_vals = np.linspace(1,5,splits(1)),
    #search_params = dict(
                         trials = 25,
                         benefit = 10,
                         param = "observability",
                         param_lim = [0,1],
                         target = WE,
                         param_tol = .05,
                         mean_tol = .1)

    trials = [
        25,
        #50,
        #75,
        #100,
        #125,
        #150,
        #175,
        #200,
    ]

    sp = [
        #dict(hue_param = 'benefit', hue_vals = np.linspace(3,10, splits(2))),
        dict(#hue_param = 'tremble', #hue_vals = [0,#0,.02,
                                                #.04,
                                                #.06,
                                                #.08,#.1
             #]
             trials = 100)
    ]

    for search_params in sp:
        name = "topographic"# % search_params['hue_param']
        #print ssd_param_search(trials = t,**dict(search_params, **common_params))
        #assert 0
        grid_param_plot(
            f = ssd_param_search,
            #row_param = "benefit", row_vals =[10],
            file_name = name,
            extension = '.png',
            plot_dir = plot_dir,
            **dict(common_params, **search_params))

def main():
    compare_slopes()

if __name__ == "__main__":
    main()
