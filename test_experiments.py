from indirect_reciprocity import ReciprocalAgent,AltruisticAgent,NiceReciprocalAgent,RationalAgent,SelfishAgent
from experiments import joint_fitness_plot,reward_table,scene_plot,belief_plot,first_impressions_plot,pop_fitness_plot,forgiveness
from experiment_utils import MultiArg

def kwarg_to_dict(**kwargs):
    return kwargs

plot_dir = './default_plots/'
condition = kwarg_to_dict(trials = 50,
                          plot_dir = plot_dir,
                          #player_types = (ReciprocalAgent,SelfishAgent),
                          agent_types = (ReciprocalAgent,SelfishAgent),
                          beta = 1)

belief_plot(believed_type = ReciprocalAgent, player_types = ReciprocalAgent, priors = .75, Ks = 1, **condition)
belief_plot(experiment = forgiveness, believed_type = ReciprocalAgent, player_types = ReciprocalAgent, priors = .75, Ks = 1, **condition)
joint_fitness_plot(**condition)
reward_table(player_types = ReciprocalAgent, size = 11, Ks = 0, **condition)
reward_table(player_types = ReciprocalAgent, size = 11, Ks = 1, **condition)


scene_plot(RA_prior = .75, RA_K = MultiArg([0,1]))

first_impressions_plot(max_cooperations = 5, RA_prior = MultiArg([float(i)/4 for i in range(1,4)]), **condition)
pop_fitness_plot((ReciprocalAgent,SelfishAgent), proportion = MultiArg([float(i)/10 for i in range(10)[1:]]), Ks = MultiArg(range(3)), plot_dir = plot_dir, trials = 500, agent_types = (ReciprocalAgent,SelfishAgent), min_pop_size = 50, beta = 1)
