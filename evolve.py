from __future__ import division
from collections import Counter, defaultdict
from itertools import product, permutations, izip
from utils import normalized, softmax, excluding_keys
from math import factorial
import numpy as np
from copy import copy
import operator
from experiments import NiceReciprocalAgent, SelfishAgent, ReciprocalAgent, AltruisticAgent
from experiment_utils import multi_call, experiment, plotter, MultiArg, cplotter, memoize, apply_to_args
import matplotlib.pyplot as plt
import seaborn as sns
from experiments import binary_matchup, memoize, matchup_matrix, matchup_plot
from params import default_genome
from indirect_reciprocity import gTFT, AllC, AllD
from params import default_params

def fixed_length_partitions(n,L):
    """
    Integer partitions of n into L parts, in colex order.
    The algorithm follows Knuth v4 fasc3 p38 in rough outline;
    Knuth credits it to Hindenburg, 1779.
    """
    
    # guard against special cases
    if L == 0:
        if n == 0:
            yield []
        return
    if L == 1:
        if n > 0:
            yield [n]
        return
    if n < L:
        return

    partition = [n - L + 1] + (L-1)*[1]
    while True:
        yield partition
        if partition[0] - 1 > partition[1]:
            partition[0] -= 1
            partition[1] += 1
            continue
        j = 2
        s = partition[0] + partition[1] - 1
        while j < L and partition[j] >= partition[0] - 1:
            s += partition[j]
            j += 1
        if j >= L:
            return
        partition[j] = x = partition[j] + 1
        j -= 1
        while j > 0:
            partition[j] = x
            s -= x
            j -= 1
        partition[0] = s

def ordered_partitions(n,L):
    for p in fixed_length_partitions(n,L):
        for perm in permutations(p):
            yield perm

def all_partitions(n,L=None):
    if not L:
        L = n
    if L>=n:
        for l in xrange(1,n+1):
            for part in fixed_length_partitions(n,l):
                for perm in permutations(part+[0]*(L-l)):
                    yield perm
    else:
        for l in xrange(1,L+1):
            for part in fixed_length_partitions(n,l):
                for perm in permutations(part+[0]*(L-l)):
                    yield perm

def binomial(n,k):
    return factorial(n)/(factorial(k)*factorial(n-k))

def stirling(n,k):
    return (1.0/factorial(k))*sum(((-1)**(k-j))*binomial(k,j)*(j**n))

def patterner(Ns,L):
    for n in range(1,Ns+1):
        s = sorted(list(set([p for p in all_partitions(n,L)])))
        l =  Counter(sorted([i[0] for i in s if i[0] is not 0]))
        #print n,L
        #print 'parts',len(s)
        #print sum(l)
        print 'sum',sum([i[0] for i in s if i[0] is not 0])

def seq_sum(n):
    return (n*(n+1)/2)

def sum_of_row(n,L):
    return sum(i[0] for i in set([p for p in all_partitions(n,L)]))

def faces(n,L):
    """
    the sum of the sizes of a particular set accross all possible partitionings
    of n identical objects into L sets, where empty partitions are allowed
    """
    if n == 0:
        return 0
    return binomial(n+L-1,n-1)

def partitions(n,L):
    """
    number of partitions of the integer n into L non-negative summands (i.e. 0 is a valid summand)
    where order matters
    """
    if n == 0:
        return 1

    return binomial(n+L-1, n)

def steady_state(matrix):
    for i,c in enumerate(matrix.T):
        np.testing.assert_approx_equal(np.sum(c),1)
        assert all(c>=0)
    vals,vecs = np.linalg.eig(matrix)
    #print 'values', vals
    #print 'vectors', vecs
    #print 'stuff'
    #for i in zip(vecs.T,vals):
    #    print i
    def almost_1(n):
        return np.isclose(n,1,.001) or (np.isclose(n.real,1,.001) and np.isclose(n.imag,0,.001))
    def negative_vec(vec):
        return all([i<0 or np.isclose(i,0) for i in vec])
    steady_states =[]
    for val,vec in sorted(zip(vals,vecs.T),key = lambda a:a[0]):
        #if almost_1(val):
        #    vec = vec.real
        #    val = val.real
        if np.isclose(val,1):
            if negative_vec(vec):
                steady_states.append((val,np.absolute(vec)))
            elif all(vec>=0):
                steady_states.append((val,vec))
            
 
    try:
        [steady_states] = steady_states
        steady_states = steady_states[1]
    except Exception as e:
        print matrix
        for l,v in zip(vals,vecs.T):
            print l,v
        return steady_states[0][1]
        #print len(steady_states)
        #print steady_states
        #print sorted(steady_states)[0][1]
        #return np.array(normalized(np.absolute(max(steady_states)[-1][1])))
        #return matrix**100
        raise e
    return np.array(normalized([n.real for n in steady_states]))

def invasion_probability(payoff, invader, dominant, pop_size, s):
    """
    calculates the odds of a single individual invading a population of dominant agents

    payoff is a TxT matrix where T is the number of types, payoff[r,o] is r's expected payoff when playing against o
    invader and dominant are integers in range(T) that shouldn't be the same
    pop_size is the total number of individuals in the population
    """
    def f(count):
        return np.exp(s*((count-1)*payoff[invader,invader]+(pop_size-count)*payoff[invader,dominant])/(pop_size-1))
    def g(count):
        return np.exp(s*(count*payoff[dominant,invader]+(pop_size-count-1)*payoff[dominant,dominant])/(pop_size-1))
    accum = 1.0
    for i in reversed(xrange(1, pop_size)):
        g_i,f_i = g(i),f(i)
        assert g_i>=0 and f_i>=0
        accum *= (g_i/f_i)
        accum += 1
    return 1/accum

def invasion_matrix(payoff, pop_size, s):
    """
    returns a matrix M of size TxT where M[a,b] is the probability of a homogeneous population of a becoming
    a homogeneous population of b under weak mutation
    types in this matrix are ordered as in 'payoff'
    """
    type_count = len(payoff)
    transition = np.zeros((type_count,)*2)
    for dominant,invader in permutations(range(type_count),2):
        transition[invader,dominant] = invasion_probability(payoff,invader,dominant,pop_size,s)/(type_count-1)

    #print transition
    #assert False
    for i in range(type_count):
        
        transition[i,i] = 1-np.sum(transition[:,i])
        #print transition[:,i]
        #transition[:,i] = normalized(transition[:,i]-transition[:,i].min())
        
        try:
            np.testing.assert_approx_equal(np.sum(transition[:,i]),1)
        except:
            print transition[:,i]
            print np.sum(transition[:,i])
            raise
    return transition

def limit_analysis(payoff, pop_size, s, **kwargs):
    """
    calculates the steady state under low mutation
    where the states correspond to the homogeneous strategy in the same order as in payoff
    """
    type_count = len(payoff)
    # partition_count = partitions(pop_size, type_count)
    transition = invasion_matrix(payoff, pop_size, s)
    # print "transition"
    # print transition
    ssd = steady_state(transition)
    return ssd


def pop_transition_matrix(payoff, pop_size, s, mu = .001, **kwargs):
    """
    returns a matrix that returns the probability of transitioning from one population composition
    to another

    the index of a population is it's position as given by the 'all_partitions()' function
    """
    type_count = len(payoff)
    partition_count = int(partitions(pop_size,type_count))
    print partition_count
    I = np.identity(type_count)

    part_to_id = dict(map(reversed,enumerate(all_partitions(pop_size,type_count))))
    partition_count = max(part_to_id.values())+1
    transition = np.zeros((partition_count,)*2)
    #print part_to_id
    #print partition_count
    #print transition
    
    for i,pop in enumerate(all_partitions(pop_size,type_count)):
        fitnesses = [np.exp(s*np.dot(pop-I[t],payoff[t])) for t in range(type_count)]
        total_fitness = sum(fitnesses)
        node = np.array(pop)
        for b,d in permutations(xrange(type_count),2):
            if pop[d]==0:
                pass
            else:
                neighbor = pop+I[b] - I[d]
                #print i,part_to_id[tuple(neighbor)]
                death_odds = pop[d] / pop_size
                #birth_odds = np.fitnesses[b]/np.exp(s*(total_fitness)))#*(1-mu)+mu*(1/type_count)
                birth_odds = fitnesses[b] / total_fitness * (1-mu) + mu * (1 / type_count)
                transition[part_to_id[tuple(neighbor)],i] = death_odds * birth_odds
                

    for i in xrange(partition_count):
        transition[i,i] = 1-sum(transition[:,i])

    return transition


def complete_analysis(payoff, pop_size = 100 ,s=1,**kwargs):
    """
    calculates the steady state distribution over population compositions
    """
    type_count = len(payoff)
    #partition_count = partitions(pop_size,type_count)
    part_to_id = dict(enumerate(all_partitions(pop_size,type_count)))
    partition_count = max(part_to_id.keys())
    transition = pop_transition_matrix(payoff,pop_size,s,**kwargs)
    ssd = steady_state(transition)

    pop_sum = np.zeros(type_count)
    for p,partition in zip(ssd,all_partitions(pop_size,type_count)):
        pop_sum += np.array(partition)*p
    #print part_to_id[10]
    return pop_sum/pop_size


@multi_call()
@experiment(unpack = 'record', unordered = ['agent_types'], memoize = False)
def limit_steady_state(player_types = NiceReciprocalAgent, pop_size = 200, size = 3, agent_types = (AltruisticAgent, ReciprocalAgent, NiceReciprocalAgent, SelfishAgent), **kwargs):
    conditions = dict(locals(),**kwargs)
    del conditions['kwargs']
    matchup,types = RA_matchup_matrix(**excluding_keys(conditions,'pop_size','s'))
    ssd = limit_analysis(matchup, **conditions)
    #priors = np.linspace(0,1,size)

    return [{"agent_prior":prior,"percentage":pop} for prior,pop in zip(types,ssd)]

@plotter(limit_steady_state, plot_exclusive_args = ['data'])
def limit_plotter(player_types = ReciprocalAgent, rounds = MultiArg(range(1,21)), data = []):
    print data[data['rounds']==41]
    priors = list(sorted(list(set(data['agent_prior']))))
    print priors
    print len(priors)
    sns.pointplot(data = data, x = 'rounds', y = 'percentage', hue= 'agent_prior', hue_order = priors)
    #for prior in priors:
    #    sns.pointplot(data = data[data['agent_prior'] == prior], x = 'rounds', y = 'percentage', hue= 'agent_prior', hue_order = priors)


@multi_call()
@experiment(unpack = 'record', unordered = ['agent_types'], memoize = False)
def complete_steady_state(player_types = NiceReciprocalAgent, pop_size = 100, size = 3, agent_types = (AltruisticAgent, ReciprocalAgent, NiceReciprocalAgent, SelfishAgent), **kwargs):
    conditions = dict(locals(),**kwargs)
    del conditions['kwargs']
    matchup,types = RA_matchup_matrix(**conditions)
    ssd = complete_analysis(matchup, **conditions)
    #priors = np.linspace(0,1,size)

    return [{"agent_prior":prior,"percentage":pop} for prior,pop in zip(types,ssd)]

@plotter(complete_steady_state, plot_exclusive_args = ['data'])
def complete_plotter(player_types = ReciprocalAgent, rounds = MultiArg(range(1,21)), data = []):
    sns.factorplot(data = data, x = 'rounds', y = 'percentage', hue ='agent_prior', col ='player_types')
    
def agent_sim(payoff, pop, s, mu):
    pop_size = sum(pop)
    type_count = len(payoff)
    I = np.identity(type_count)
    pop = np.array(pop)
    assert type_count == len(pop)
    while True:
        yield pop
        fitnesses = [np.dot(pop - I[t], payoff[t]) for t in xrange(type_count)]
        fitnesses = softmax(fitnesses, s)
        # fitnesses = [np.exp(s*np.dot(pop - I[t], payoff[t])) for t in xrange(type_count)]
        # total_fitness = sum(fitnesses)
        actions = [(b,d) for b,d in permutations(xrange(type_count),2) if pop[d]!=1]
        probs = []
        for b,d in actions:
            death_odds = pop[d] / pop_size
            # birth_odds = (1-mu) * fitnesses[b] / total_fitness + mu * (1/type_count)
            birth_odds = (1-mu) * fitnesses[b] + mu * (1/type_count)
            prob = death_odds * birth_odds
            probs.append(prob)
        actions.append((0, 0))
        probs.append(1 - np.sum(probs))

        probs = np.array(probs)
        action_index = np.squeeze(np.where(np.random.multinomial(1,probs)))
        (b,d) = actions[action_index]
        pop = pop + I[b]-I[d]

@experiment(unpack = 'record', memoize = False)
def agent_simulation(generations, pop, player_types, agent_types= None, **kwargs):
    if agent_types is None:
        agent_types = player_types

    payoffs = matchup_matrix(player_types = player_types, agent_types = agent_types, **kwargs)
    params = default_params()
    populations = agent_sim(payoffs, pop, params['s'], params['mu'])

    record = []
    for n, pop in izip(xrange(generations), populations):
        for t, p in zip(player_types, pop):
            record.append({'generation' : n,
                           'type' : t,
                           'population' : p})
    return record


@plotter(agent_simulation,plot_exclusive_args = ['data'])
def sim_plotter(generations, pop, player_types, agent_types= None, data =[]):
    for hue in data['type'].unique():
        plt.plot(data[data['type']==hue]['generation'], data[data['type']==hue]['population'], label=hue)

    plt.legend()

def logspace(start = .001,stop = 1, samples=10):
    mult = (np.log(stop)-np.log(start))/np.log(10)
    plus = np.log(start)/np.log(10)
    return np.array([0]+list(np.power(10,np.linspace(0,1,samples)*mult+plus)))

def int_logspace(start, stop, samples, base=2):
    return sorted(list(set(np.logspace(start, stop, samples, base=base).astype(int))))
    
@experiment(unpack = 'record', memoize = False)
def limit_v_evo_param(param, player_types, agent_types = None, **kwargs):
    if agent_types is None:
        agent_types = player_types

    payoffs = matchup_matrix(player_types = player_types, agent_types = agent_types, **kwargs)
    matchup_plot(player_types = player_types, agent_types=agent_types, **kwargs)

    if param == 'pop_size':
        # Xs = range(2, 2**10)
        Xs = np.unique(np.geomspace(2, 2**10, 200, dtype=int))
    elif param == 's':
        Xs = logspace(start = .001, stop = 10, samples = 100)
    else:
        print param
        raise
    #print Xs

    params = default_params()
    
    record = []
    for x in Xs:
        params[param] = x
        for t, p in zip(player_types, limit_analysis(payoffs, **params)):
            record.append({
                param : x,
                "type" : t,
                "proportion" : p
            })
    return record

@plotter(limit_v_evo_param)
def limit_evo_plot(param, player_types, data = [], **kwargs):
    fig = plt.figure()
    for hue in data['type'].unique():
        d = data[data['type']==hue]
        p = plt.plot(d[param], d['proportion'], label=hue)
    if param == 'pop_size':
        plt.axes().set_xscale('log',basex=2)
    elif param == 's':
        plt.axes().set_xscale('log')
    plt.legend()

@experiment(unpack = 'record', memoize = False)
def limit_v_sim_param(param, player_types, agent_types=None, **kwargs):
    if param == "RA_prior":
        Xs = np.linspace(0,1,21)
    elif param == "beta":
        Xs = logspace(0,1,11)
    else:
        raise

    params = default_params()
    if agent_types is None:
        agent_types = player_types
        
    record = []
    for x in Xs:
        params[param] = x
        payoffs = matchup_matrix(player_types = player_types, agent_types = agent_types, **params)
        #matchup_plot(player_types = agents, agent_types = agents, xrounds = 10, trials = 100, **dict(kwargs,**{param:x}))
        for t,p in zip(player_types, limit_analysis(payoffs,**defaults)):
            record.append({
                param:x,
                "type":t,
                "proportion":p
            })
    return record

@plotter(limit_v_sim_param)
def limit_sim_plot(param, player_types, data = [], **kwargs):
    fig = plt.figure()
    for hue in data['type'].unique():
        d = data[data['type']==hue]
        p = plt.plot(d[param], d['proportion'], label=hue)
    #if param == "beta":
    #    plt.axes().set_xscale('log',basex=10)
    plt.legend()

def run_plots():
    NRA = NiceReciprocalAgent
    MRA = ReciprocalAgent
    SA = SelfishAgent
    AA = AltruisticAgent
    AC = AllC
    AD = AllD
    TFT = gTFT(y=1,p=1,q=0)
    old_pop1 = (TFT,AC,AD)
    old_pop2 = (TFT,AA,SA)
    for tremble in [
            0,
            .05,
            #.1
    ]:
        for old_pop in [old_pop1,old_pop2]:
            #limit_sim_plot('RA_prior', old_pop, tremble=tremble)
            #limit_sim_plot('beta', old_pop, tremble=tremble)
            limit_evo_plot('s', old_pop, tremble=tremble, K = 1)
            limit_evo_plot('pop_size', old_pop, tremble = tremble, K = 1)
        for RA in [MRA,NRA]:
            pop1 = (RA,AA,SA)
            pop2 = (RA,AC,AD)
            pop3 = (RA,AC,AD,TFT)
            for pop in [#pop1,
                        #pop2,
                        pop3
            ]:
                try:
                    #limit_sim_plot('RA_prior', pop, tremble=tremble)
                    pass
                except:
                    pass
                #limit_sim_plot('beta', pop, tremble=tremble)
                limit_evo_plot('s', pop, tremble=tremble, K = 1)
                limit_evo_plot('pop_size', pop, tremble=tremble, K = 1)
        

def priority_plots():
    NRA = NiceReciprocalAgent
    MRA = ReciprocalAgent
    SA = SelfishAgent
    AA = AltruisticAgent
    AC = AllC
    AD = AllD
    TFT = gTFT(y=1,p=1,q=0)
    tremble = 0
    K = 0
    beta = 3
    prior = .75
    RAs = [
            MRA,
            # NRA
    ]
    Ks = range(1)
    for RA, K in product(RAs, Ks):
        pop1 = (RA,AA,SA)
        # limit_evo_plot('pop_size', pop1)
        # limit_evo_plot('s', pop1)
        # limit_sim_plot('RA_prior', pop1)
        pop2 = (RA,AC,AD)
        # limit_evo_plot('pop_size', pop2, agent_types = pop1, RA_prior = 0.5)
        # limit_evo_plot('s', pop2, agent_types = pop1, RA_prior = 0.5)
        # limit_evo_plot('pop_size', pop2)
        # limit_evo_plot('s', pop2)
        # # limit_sim_plot('RA_prior', pop2)
        # # limit_sim_plot('beta', pop2)
        # # limit_evo_plot('pop_size', pop2, tremble = tremble)
        pop3 = (RA,AC,AD,TFT)
        limit_evo_plot('pop_size', pop3, agent_types = pop3, RA_prior=prior, Ks=K, beta=beta, rounds=200)
        limit_evo_plot('s', pop3, agent_types = pop3, RA_prior=prior, Ks=K, beta=beta, rounds=200)
        # sim_plotter(50000, (0,0,100,0), player_types = pop3, agent_types = pop1, K=K, beta = beta, RA_prior = prior, rounds=50)
        
    # old_pop = (TFT,AC,AD)
    # limit_evo_plot('pop_size', old_pop)
    # limit_evo_plot('s', old_pop)

#test_plots([ReciprocalAgent])
#for RA,k in product([NiceReciprocalAgent],[0,1]):
#sim_plotter(5000,(0,.9,1),(100,0,0), Ks=1, player_type = ReciprocalAgent)
#print RA_matchup_matrix((0,.5,1),player_types = NiceReciprocalAgent,agent_types=(AltruisticAgent, NiceReciprocalAgent, SelfishAgent))

#matchup,types = 
#print types
#print "matchup"
#print matchup.round(3)
#print "ssd"

#print np.round(complete_analysis(matchup,10,.1,mu=.001), decimals = 2)
#print np.round(limit_analysis(matchup,10,.1), decimals = 2)
#print steady_state(rps.T)
#print steady_state(rps)



if __name__ == "__main__":
    #sim_plotter(100000,(0,.6,1), player_type = NRA, s = .1, mu = .05, Ks = 0, pop = (0,100,0))
    
    priority_plots()
    assert 0
    # run_plots()
    
    NRA = NiceReciprocalAgent
    MRA = ReciprocalAgent
    TFT = gTFT(y=1,p=1,q=0)
    SA = SelfishAgent
    AA = AltruisticAgent
    prior = 0.75
    K = 0
    # for RA in [MRA(RA_prior = prior), NRA(RA_prior = prior)]:
    for RA in [
            MRA,
            NRA
    ]:
        tom_types = (SA, AA, RA)
        types = (SA, AA, RA)
        # types = (RA(RA_prior = 0), RA(RA_prior = 1), RA(RA_prior = 0.75, agent_types = (SA, AA, 'self')))
        # types = (AllC, AllD, RA)
        # matchup_plot(player_types = types, agent_types = tom_types, rounds = 10, RA_prior = prior)
        limit_evo_plot(param = 'pop_size', agents = types, K=K)
        limit_evo_plot(param = 's', agents = types, K=K)
    
    #print matchup_matrix(player_types = (MRA,AA), RA_prior = .5, rounds = 10)
    #matchup_plot()
    assert False
    #sim_plotter(100000,(0,.6,1), player_type = NRA, s = .1, mu = .05, Ks = 0, pop = (0,100,0))
    for RA in [
            MRA,
      #      NRA
    ]:
        limit_v_priors(RA)
        #limit_prior_plotter(RA)
        #limit_evo_plotter(plot_name = "ssd v s", RA = RA, param = 'pop_size')
    assert False
    ks = [0,1]
    RAs = [ReciprocalAgent,NiceReciprocalAgent]
    #RAs = [NiceReciprocalAgent]
    ssds = []
    N = 11
    priors = [round(i,2) for i in np.linspace(0,1,N)]#(0,.25,.5,.75,1)
    for RA,k in product(RAs,ks):
        matchup = RA_matchup_matrix(priors,player_types = RA,agent_types=(AltruisticAgent, RA, SelfishAgent),Ks = k)
        s = np.round(limit_analysis(matchup,100,.01),2)
        ssds.append((RA,k,s))
    print priors
    for r,k,s in ssds:
        print r,k
        print s

    


    #print binary_matchup(NiceReciprocalAgent,priors = 0, agent_types = (AltruisticAgent, NiceReciprocalAgent, SelfishAgent))
    #print RA_matchup_matrix(priors = [0,.5,1], player_types = NiceReciprocalAgent, agent_types = (AltruisticAgent, NiceReciprocalAgent, SelfishAgent))

"""
make s the difference between the rewards of two most matchups
"""

#print partitions(100,3)
#rps = np.array([
#    [0,1],
#    [.5,0]]).T
#print np.linalg.eig(rps)
#rps_transition = np.array([
#    [.9,.1,0],
#    [0,.5,.5],
#    [.5,.0,.5]
    #]).T

#print np.linalg.inv(rps_transition)

#def replicator(matchup,pop_vec,generations):
#    for i in range_generations:
#        pop_vec = pop_vec.*
