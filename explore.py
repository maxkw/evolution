import inspect
plot_dir = "./plots/"+inspect.stack()[0][1][:-3]+"/"
from evolve import grid_param_plot, splits, splits
import agents as ag
import numpy as np
from games import AnnotatedGame, IndefiniteMatchup, AllNoneObserve, literal

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
    agents = (ag.WeAgent(agent_types=ToM),)+opponents
    common_params = dict(s=.5,
                         #game='dynamic',
                         #game = "dynamic_sim",
                         #game = 'direct',
                         #game = 'dd_ci_va',
                         player_types = agents,
                         analysis_type = 'limit',
                         beta = 5,
                         pop_size = 10,
                         plot_dir = plot_dir,
                         stacked = True,
                         #parallelized = False,
                         extension = '.png',
                         benefit = 3,
                         expected_interactions = 1
                         )

    games = [
        #'dd_ci_sa',
        'dd_ci_va',
        #'dd_ci_pa',
        #'dd_ci_pas'
    ]

    trials = [
        #25,
        #50,
        #75,
        #100,
        125,
        150,
        175,
        200,
    ]

    for t in trials:
        name = "compare_slopes(games=%s, trials = %s)" % (games,t)
        grid_param_plot(
            #game = g,
            trials = t,
            prior = .5,
            file_name = name,
            param = "observability", param_vals = np.round(np.linspace(0,1,splits(2)),2),#(0,.25,.5,.75,1),
            #col_param = "prior", col_vals = (.5,),#(.25,.5,.75),
            col_param = "beta", col_vals = (5,),
            #col_param = "variance", col_vals
            row_param = "game", row_vals = games,
            hue_param = "intervals", hue_vals = (
                2,
                3,
                4,
                5,
                6,
                #7,
                #8,
                #9,
                #10
            ), **common_params)



def main():
    compare_slopes()

if __name__ == "__main__":
    main()
