from __future__ import division
from collections import Counter, defaultdict
from itertools import product, permutations, combinations, izip
from utils import normalized, softmax, excluding_keys
from math import factorial
import numpy as np
from experiment_utils import multi_call, experiment, plotter, MultiArg
from observability_experiments import indirect_simulator
from functools import partial
from utils import memoized
from multiprocessing import Pool
from experiments import matchup,matchup_matrix_per_round

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

def pop_transition_matrix(payoff, pop_size, s, mu = .001, **kwargs):
    """
    returns a matrix that returns the probability of transitioning from one population composition
    to another

    the index of a population is it's position as given by the 'all_partitions()' function
    """
    type_count = len(payoff)
    I = np.identity(type_count)
    partitions = sorted(set(all_partitions(pop_size,type_count)))
    part_to_id = dict(map(reversed,enumerate(partitions)))
    print sorted(part_to_id.values())
    partition_count = len(part_to_id)
    transition = np.zeros((partition_count,)*2)
    for pop,i in part_to_id.iteritems():
        fitnesses = softmax([np.dot(pop-I[t],payoff[t]) for t in range(type_count)], s)
        node = np.array(pop)
        for b,d in permutations(xrange(type_count),2):
            if pop[d] != 0:
                neighbor = pop+I[b] - I[d]
                #print i,part_to_id[tuple(neighbor)]
                death_odds = pop[d] / pop_size
                #birth_odds = np.fitnesses[b]/np.exp(s*(total_fitness)))#*(1-mu)+mu*(1/type_count)
                birth_odds = fitnesses[b] * (1-mu) + mu * (1 / type_count)
                transition[part_to_id[tuple(neighbor)],i] = death_odds * birth_odds

    for i in xrange(partition_count):
        transition[i,i] = 1-sum(transition[:,i])

    return transition

def complete_analysis(payoff, pop_size, s, **kwargs):
    """
    calculates the steady state distribution over population compositions
    """
    type_count = len(payoff)
    transition = pop_transition_matrix(payoff, pop_size, s, **kwargs)
    ssd = steady_state(transition)

    pop_sum = np.zeros(type_count)
    for p,partition in zip(ssd, sorted(set(all_partitions(pop_size, type_count)))):
        pop_sum += np.array(partition) * p

    return pop_sum / pop_size
