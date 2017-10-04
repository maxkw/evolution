import numpy as np
# from agents import ReciprocalAgent, AltruisticAgent, RationalAgent, SelfishAgent
# from experiment_utils import MultiArg
import scenarios
from experiments import plot_coop_prob
import evolve
import direct

if __name__ == '__main__':
    plot_dir = './figures/'

    params = dict(
        beta = 5,
        plot_dir = plot_dir
    )

    plot_coop_prob(file_name = 'coop_prob', **params)
    scenarios.main(**params)
    direct.ToM_indirect(**params)
    direct.Compare_Old(**params)

# # Currently broken
# evolve.AllC_AllD_race()
# evolve.Pavlov_gTFT_race()


assert 0 


# condition = dict(trials=50,
#                  plot_dir=plot_dir,
#                  beta=5,
# )



# belief_plot(believed_type=ReciprocalAgent,
#             player_types=ReciprocalAgent, priors=.75, Ks=1, **condition)
# belief_plot(experiment=forgiveness, believed_type=ReciprocalAgent,
#             player_types=ReciprocalAgent, priors=.75, Ks=1, **condition)
# joint_fitness_plot(**condition)
# reward_table(player_types=ReciprocalAgent, size=11, Ks=0, **condition)
# reward_table(player_types=ReciprocalAgent, size=11, Ks=1, **condition)


# scene_plot(RA_prior=.75, RA_K=MultiArg([0, 1]))

# pop_fitness_plot((ReciprocalAgent, SelfishAgent), proportion=MultiArg([float(i) / 10 for i in range(10)[1:]]), Ks=MultiArg(
#     range(3)), plot_dir=plot_dir, trials=500, agent_types=(ReciprocalAgent, SelfishAgent), min_pop_size=50, beta=1)
