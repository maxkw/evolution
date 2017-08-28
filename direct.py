def AllC_AllD_race():
    today = "./plots/"+date.today().isoformat()+"/"
    
    ToM = ('self', AllC, AllD)
    opponents = (AllC, AllD)
    pop = (WeAgent(agent_types = ToM), ag.TFT)
    
    for t in [#0,
              0.05]:
        background_params = dict(
            experiment = compare_ssd_v_param,
            direct = True,
            RA_prior = 0.5,
            beta = 5,
            player_types = pop,
            opponent_types = opponents,
            agent_types = ToM,
            tremble = t,
            pop_size = 100, 
            plot_dir = today
        )
        
        # limit_param_plot('s', rounds = 100, file_name = 'contest_s_rounds=100_tremble=%0.2f' % t, **background_params)
        # limit_param_plot('s', rounds = 10, file_name = 'contest_s_rounds=10_tremble=%0.2f' % t, **background_params)
        # limit_param_plot("rounds", rounds = 100, s=1, file_name = 'contest_rounds_tremble=%0.2f' % t, **background_params)
        limit_param_plot("RA_prior", rounds = 10, s=1, file_name = 'contest_prior_tremble=%0.2f' % t, **background_params)
        # limit_param_plot("beta", rounds = 10, s=1, file_name = 'contest_beta_tremble=%0.2f' % t, **background_params)


#a_type, proportion = max(zip(player_types,ssd), key = lambda tup: tup[1])

def Pavlov_gTFT_race():
    today = "./plots/"+date.today().isoformat()+"/"
    TFT = gTFT(y=1,p=1,q=0)
    MRA = WeAgent#ReciprocalAgent
    r = 10
    
    # Replicate Nowak early 90s
    pop = (TFT, AllC, AllD, gTFT(y=1,p=.99,q=.33), Pavlov)
    for t in [0, 0.05]:
        limit_param_plot('s', pop, rounds = r, tremble = t, file_name = 'nowak_replicate_s_tremble=%.2f' % t, plot_dir = today)
    sim_plotter(5000, (0,0,100,0,0), player_types = pop, rounds = r, tremble = 0.05, mu=0.05, s=1, file_name ='nowak_replicate_sim_tremble=0.05', plot_dir = today)

    # Horse race against gTFT and Pavlov
    prior = 0.5
    beta = 10

    trembles = [0, 0.05]
    priors = [.5,.75]
    betas = [3,5,10]
    opponents = (TFT, gTFT(y=1,p=.99,q=.33), AllC, AllD, Pavlov)

    ToM = ('self',)+opponents
    agent = MRA(agent_types = ToM, beta = beta, RA_prior = prior)
    pop = (agent, AllC, AllD, TFT, gTFT(y=1,p=.99,q=.33), Pavlov)

    comparables = tuple(MRA(RA_prior=p, beta = b, agent_types=ToM) for p,b in product(priors,betas))

    for t in trembles:
        limit_param_plot('s', player_types = pop, rounds = r, tremble = t, file_name = 'horse_s_no_random_tremble=%0.2f' % t, plot_dir = today)
        limit_param_plot('rounds',
                         player_types = comparables,
                         opponent_types = opponents,
                         tremble = t,
                         experiment = compare_ssd_v_param,
                         file_name = 'horse_rounds_no_random_tremble=%0.2f' % t,
                         plot_dir = today)

    # Add Random to the ToM
    ToM = ('self', TFT, gTFT(y=1,p=.99,q=.33), AllC, AllD, Pavlov, RandomAgent)
    for t in trembles:
        limit_param_plot('s', player_types = pop, rounds = r, tremble = t, file_name = 'horse_s_will_random_tremble=%.2f' % t,plot_dir = today)
        limit_param_plot('rounds',
                         player_types = comparables,
                         opponent_types = opponents,
                         tremble = t,
                         experiment = compare_ssd_v_param,
                         file_name = 'horse_rounds_with_random_tremble=%.2f' % t,
                         plot_dir = today)

def bc_rounds_contest():
    WA = WeAgent
    prior = 0.5

    RA = WA(RA_prior = prior, agent_types = ('self', ag.AllC, ag.AllD))
    player_types = (RA, ag.TFT, ag.GTFT, ag.Pavlov)

    for t in [0, 0.05]:
        bc_rounds_plot(
            max_rounds = 20,
            experiment = compare_bc_v_rounds,
            player_types = player_types,
            opponent_types = (ag.AllC, ag.AllD),
            tremble = t
        )



def bc_rounds_race():
    file_name = "ToM = %s, beta = %s, prior = %s, tremble = %s"
    plot_dir = "./plots/bc_rounds_race/"

    max_rounds = 20

    priors = [
        #.1,
        .5,
        # .75
    ]

    ToMs = [
        ('self', ag.AllC, ag.AllD, ag.TFT, ag.GTFT, ag.Pavlov)
    ]

    betas = [
        #1,
        #3,
        5,
        # 10,
    ]

    trembles = [
        0,
        0.05
    ]

    for prior, ToM, beta in product(priors,ToMs,betas):
        RA = WeAgent(RA_prior = prior, agent_types = ToM, beta = beta)
        everyone = (RA, ag.AllC, ag.AllD, ag.TFT, ag.GTFT, ag.Pavlov)
        for t in trembles:
            bc_rounds_plot(everyone, max_rounds = max_rounds, tremble = t,
                           plot_dir = plot_dir,
                           file_name = file_name % (ToM,beta,prior,t)
            )

def limit_rounds_race():
    file_name = "ToM = %s, beta = %s, prior = %s, tremble = %s"
    plot_dir = "./plots/limit_rounds_race/"

    MRA = ReciprocalAgent
    SA = SelfishAgent
    AA = AltruisticAgent
    AC = AllC
    AD = AllD
    TFT = gTFT(y=1,p=1,q=0)
    GTFT = gTFT(y=1,p=.99,q=.33)

    max_rounds = 50

    priors = [
        #.1,
        #.5,
        #.75
        #.99
    ]

    ToMs = [
        ('self', AC, AD, TFT, GTFT, Pavlov)
    ]

    betas = [
        #.5,
        #1,
        #3,
        #5,
        10,
    ]

    trembles = [
        0,
        0.05
    ]

    for prior, ToM, beta in product(priors,ToMs,betas):
        RA = WeAgent(RA_prior = prior, agent_types = ToM, beta = beta)
        everyone = (RA, AC, AD, TFT, GTFT, Pavlov)
        for t in trembles:
            limit_param_plot(param = 'rounds', player_types = everyone, max_rounds = max_rounds, tremble = t,
                             plot_dir = plot_dir,
                             file_name = file_name % (ToM,beta,prior,t),
                             extension = '.png'
            )



if __name__ == "__main__":
    # image_contest()
    AllC_AllD_race()
    # Pavlov_gTFT_race()
    # bc_rounds_race()
    # limit_rounds_race()
    assert 0

    MRA = ReciprocalAgent
    SA = SelfishAgent
    AA = AltruisticAgent
    AC = AllC
    AD = AllD
    TFT = gTFT(y=1,p=1,q=0)
    GTFT = gTFT(y=1,p=.99,q=.33)

    prior = 0.5
    Ks = [0,1]
    trembles = [0, 0.05]
    max_rounds = 20
    beta = 1
    
    #for t in trembles:
    #    bc_rounds_plot(
    #        max_rounds = max_rounds,
    #        experiment = compare_bc_v_rounds,
    #        player_types = tuple(MRA(RA_prior = prior, RA_K = k, agent_types = ('self', AC, AD)) for k in Ks) + (TFT, GTFT, Pavlov),
    #        opponent_types = (AC, AD),
    #        tremble = t,
    #        beta = beta,
    #        file_name = 'heat_tremble=%0.2f' % t)

    # assert 0

    
    
    everyone_ToM = ('self', AC, AD, TFT, GTFT, Pavlov, RandomAgent)
    RA = WeAgent(RA_prior = prior, agent_types = everyone_ToM, beta = beta)
    everyone = (RA, AC, AD, TFT, GTFT, Pavlov)
    for t in trembles:
        bc_rounds_plot(everyone, max_rounds = max_rounds, tremble = t)

    assert 0
    
    #limit_param_plot('bc',everyone)
    #limit_param_plot('rounds', everyone)
