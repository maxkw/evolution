import inspect
plot_dir = "./plots/"+inspect.stack()[0][1][:-3]+"/"

from evolve import limit_param_plot, splits
import agents as ag
import numpy as np
import seaborn as sns

TRIALS = 100

def dynamic_dilemma_plot(trials = TRIALS):
    opponents = (ag.SelfishAgent(beta=5),ag.AltruisticAgent(beta=5))
    ToM = ('self',)+opponents
    agents = (ag.WeAgent(prior=.5,beta=5,agent_types=ToM),)+opponents
    scenario = {
        'omega':{
            'param':'observability',
            'param_vals': np.round(np.linspace(0, 1, 21), 2),
            'expected_interactions':1,
            'trials':50,
            "graph_kwargs": {'color' : sns.color_palette(['C0', 'C1', 'C5']),
                            'xlim': [0,1],
                            'xticks':np.linspace(0,1,5)},
        },
        'gamma':{
            'param':'expected_interactions',
            'param_vals': np.round(np.linspace(1, 10, 10), 2),
            'observability':0,
            'trials':100,
            "graph_kwargs":{'color' : sns.color_palette(['C0', 'C1', 'C5']),
                            'xlim': [1,10]},
        }
    }

    common_params = dict(s=.5,
                         game='dynamic',
                         player_types = agents,
                         analysis_type = 'limit',
                         pop_size = 10,
                         plot_dir = plot_dir,
                         stacked = True,
                         #parallelized = False,
                         )

    for scene_name in [
            'omega',
            'gamma'
    ]:
        scene_params = scenario[scene_name]
        file_name = scene_name+"_plot"
        limit_param_plot(file_name = file_name,
                         **dict(common_params,**scene_params))

def fig4(trials = 200):
    common_params = dict(game = 'direct',
                         benefit = 3,
                         cost = 1,
                         pop_size = 100,
                         analysis_type = 'limit',
                         s = .5,
                         plot_dir = plot_dir,
                         trials = trials,
                         stacked = True,
                         rounds = 10
                         )

    old_pop = (ag.AllC,ag.AllD,ag.GTFT,ag.TFT,ag.WSLS)
    ToM = ('self',)+old_pop
    new_pop= old_pop +(ag.WeAgent(prior = .5, beta = 5, agent_types = ToM),)
    conditions = dict(top = dict(param = 'rounds', rounds = 50, tremble = 0, graph_kwargs = {'xlim':[1,50]}),
                      bottom = dict(param = 'tremble',
                                    param_vals = np.round(np.linspace(0, 0.4, 41),2),
                                    graph_kwargs = {'xlim':[0,.4]}),
                      left = dict(player_types = old_pop),
                      right = dict(player_types = new_pop))

    scenarios = dict(a=dict(conditions['top'],**conditions['left']),
                     b=dict(conditions['top'],**conditions['right']),
                     c=dict(conditions['bottom'],**conditions['left']),
                     d=dict(conditions['bottom'],**conditions['right']))
    for letter in [
            'a',
            'b',
            'c',
            'd',
    ]:
        scene_params = scenarios[letter]
        limit_param_plot(file_name = "fig4"+letter,
                         **dict(common_params,**scene_params))
def main():
    #dynamic_dilemma_plot()
    fig4()

if __name__ == "__main__":
    main()
