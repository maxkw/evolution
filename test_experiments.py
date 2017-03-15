from indirect_reciprocity import ReciprocalAgent,AltruisticAgent,NiceReciprocalAgent,RationalAgent,SelfishAgent
from experiments import joint_fitness_plot,reward_table,scene_plot,belief_plot
from experiment_utils import MultiArg

default_plots_dir = './default_plots/'
belief_plot(believed_type=ReciprocalAgent, player_types = ReciprocalAgent, agent_types = (ReciprocalAgent,SelfishAgent), priors = .75, Ks = 1,trials = 500, plot_dir = default_plots_dir)


reward_table(plot_dir = "./default_plots/",player_types = ReciprocalAgent, size = 11, Ks = 0, agent_types = (ReciprocalAgent,SelfishAgent), beta = 1,)
reward_table(plot_dir = "./default_plots/",player_types = ReciprocalAgent, size = 11, Ks = 1, agent_types = (ReciprocalAgent,SelfishAgent), beta = 1)

#scene_plot(RA_prior = .75, RA_K = MultiArg([0,1]), )
