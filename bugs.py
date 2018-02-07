from agents import SelfishAgent,AltruisticAgent,WeAgent
from evolve import limit_param_plot

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
        game = 'dynamic', observability = 0, gamma = 0.9,
        tremble = 0,
        

        player_types = pop,
        analysis_type = 'limit',
        beta = 5,
        pop_size = 10,
        s = .5, 

        #parallelized = False,
    )
    
    file_name = 'bugged if different every time'
    limit_param_plot("rounds",
                     file_name = file_name,
                     plot_dir = bugs_dir,
                     ##stacked = True,
                     **background_params)

def main():
    variable_plots()

if __name__ == "__main__":
    main()
