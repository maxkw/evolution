import inspect
plot_dir = "./plots/"+inspect.stack()[0][1][:-3]+"/"

from evolve import limit_param_plot
import agents as ag



def dynamic_dilemma_plot():
    opponents = (ag.SelfishAgent(beta=5),ag.AltruisticAgent(beta=5))
    ToM = ('self',)+opponents
    agents = (ag.WeAgent(prior=.5,beta=5,agent_types=ToM),)+opponents
    scenario = {
        'omega':{
            'param':'observability',
            'expected_interactions':1,
        },
        'gamma':{
            'param':'expected_interactions',
            'observability':0
        }
    }

    common_params = dict(s=.5,
                         game='dynamic',
                         player_types = agents,
                         analysis_type = 'limit',
                         trials = 50,
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

def fig4():
    common_params = dict(game = 'direct',
                         benefit = 3,
                         cost = 1,

                         pop_size = 100,
                         analysis_type = 'limit',
                         s = .5,
                         plot_dir = plot_dir,
                         trials = 50,
                         stacked = True,
                         rounds = 50
                         )

    old_pop = (ag.AllC,ag.AllD,ag.GTFT,ag.TFT,ag.Pavlov)
    ToM = ('self',)+old_pop
    new_pop= old_pop +(ag.WeAgent(prior = .5, beta = 5, agent_types = ToM),)
    conditions = dict(top = dict(param = 'rounds', tremble = 0),
                      bottom = dict(param = 'tremble'),
                      left = dict(player_types = old_pop),
                      right = dict(player_types = new_pop))

    scenarios = dict(a=dict(conditions['top'],**conditions['left']),
                     b=dict(conditions['top'],**conditions['right']),
                     c=dict(conditions['bottom'],**conditions['left']),
                     d=dict(conditions['bottom'],**conditions['right']))
    for letter, scene_params in scenarios.iteritems():
        limit_param_plot(file_name = "fig4"+letter,
                         **dict(common_params,**scene_params))
def main():
    dynamic_dilemma_plot()
    fig4()

if __name__ == "__main__":
    main()
