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

if __name__ == "__main__":
    pass
