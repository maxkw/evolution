from agents import SelfishAgent, WeAgent
from evolve import limit_param_plot
from itertools import product

bugs_dir = "./plots/bugs/"

def variable_plots():
    """this is due to prefabs not being stored correctly"""
    opponents = (
        SelfishAgent(beta=5),
    )
    ToM = ('self', ) + opponents
    
    
    W = WeAgent(agent_types = ToM, beta = 10, prior = .5,)
    pop = (W,)+opponents


    background_params = dict(
        trials = 20,
        #game = 'direct', direct = False,rounds = 10,
        game = 'dynamic', observability = 0, expected_interactions = 10,
        tremble = 0,
        

        player_types = pop,
        analysis_type = 'limit',
        beta = 5,
        pop_size = 10,
        s = .5, 

        #parallelized = False,
    )
    
    file_name = 'variable_plots - bugged if different every time'
    limit_param_plot("rounds",
                     file_name = file_name,
                     plot_dir = bugs_dir,
                     ##stacked = True,
                     **background_params)

def analyses():
    opponents = (
        SelfishAgent,
    )
    ToM = ('self', ) + opponents
    
    
    W = WeAgent(agent_types = ToM, prior = .5,)
    pop = (W,)+opponents


    background_params = dict(
        trials = 20,
        game = 'direct', rounds = 10,
        tremble = 0,
        

        player_types = pop,
        beta = 5,
        pop_size = 10,
        s = .5, 
    )

    for analysis,direct in product(['limit','complete'],[True,False]):
        file_name = "ssd_v_rounds(analysis_type = %s, analytic = %s)" % (analysis,direct)
        limit_param_plot("rounds",
                         file_name = file_name,
                         analysis_type = analysis,
                         direct = direct,
                         plot_dir = bugs_dir+"analyses - should all look the same/",
                         **background_params)

def main():
    variable_plots()
    analyses()

if __name__ == "__main__":
    main()
