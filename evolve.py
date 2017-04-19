from __future__ import division
from collections import Counter,defaultdict
from itertools import product
from itertools import permutations
from utils import normalized
from math import factorial
import numpy as np
from copy import copy


class probDict(dict):
    def __init__(self,*args,**kwargs):
        keys,vals = zip(*dict(*args,**kwargs).items())
        try:
            assert np.sum(vals) == 1
        except AssertionError:
            vals = normalized(vals)
        super(probDict,self).__init__(zip(keys,vals))

    def sample(self):
        keys,vals = zip(*self.items())
        return keys[np.squeeze(np.where(np.random.multinomial(1,vals)))]

def practical_moran(type_fitness_pairs, agent_types = None, selection_strength = .1, mutation_rate = 0):
    """
    returns a dict mapping agent_types to number of agents.
    this is meant for use in actual simulations, with the real results of a run used as primary inputs

    'type_fitness_pairs' is a list of tuples whose first element is a type and the second a number
    if 'agent_types' is not provided it is inferred from 'type_fitness_pairs'
    'selection_strength' determines how much fitness affects the odds of birth, where 0 means the odds are only a
    function of population
    'mutation_rate' sets the odds of birth being a random type selection independent of population and fitness
    """
    s = selection_strength
    mu = mutation_rate

    type_list,fitness_list = zip(*type_fitness_pairs)

    pop_size=len(type_list)
    type_to_count = Counter(type_list)
    if not agent_types:
        agent_types = type_to_count.keys()
    type_to_fitness = defaultdict(int)
    for agent_type, fitness in type_fitness_pairs:
        type_to_fitness[agent_type] += fitness

    birth_odds_denom = pop_size*(1-s)+s*np.sum(fitness_list)
    random_type_odds = 1.0/len(agent_types)
    birth_odds = {}
    for t in agent_types:
        birth_odds[t] = (type_to_count[t]*(1-s)+s*type_to_fitness[t]/birth_odds_denom)*(1-mu)+(random_type_odds)*mu

    death_type = probDict(type_to_count).sample()
    birth_type = probDict(birth_odds).sample()

    type_to_count[death_type] -= 1
    type_to_count[birth_type] += 1

    return type_to_count

type_fitness_pairs = [('a',70),('b',30)]
#print practical_moran(type_fitness_pairs)

def moran_analysis(type_to_count, payoff, selection_strength = .1, mutation_rate = 0):
    agent_types = type_to_count.keys()
    pop_size = sum(type_to_count.values())

    total_opponents = float(pop_size - 1)
    def expected_reward(a_type,pop):
        """
        returns a dict of opponent types, where
        expected_reward()
        """
        expected_dict = defaultdict(int)

        for player, opponent in product(agent_types,r = 2):
            opponents = type_to_count[opponent]
            if player == opponent:
                opponents -= 1
            expected_dict[player] += payoff[player][opponent]*opponents/total_opponents
        return expected_dict

    death_odds = probDict(type_to_count)

    birth_odds = {t:type_to_count[t]*expected_payoff}

    death_rate = {}



def binary_moran_analysis(agent_types, pop_size, payoff_dict, selection_strength = .1):
    s = selection_strength

    def expected_reward(recipient_type,pop):
        """
        returns the expected reward for a given type 'a_type' if it's population is 'pop'.
        """
        reward = 0
        for opponent_type in agent_types:
            if opponent_type == recipient_type:
                opponent_count = pop-1
            else:
                opponent_count = pop_size-pop

            reward += payoff_dict[recipient_type][opponent_type]*opponent_count/(pop_size-1)

        return (1-s)+s*reward

    def death_birth_ratio(a_type,pop):
        [opponent_type] = [o_type for o_type in agent_types if o_type is not a_type]
        return expected_reward(opponent_type,pop_size-pop)/expected_reward(a_type,pop)

    def invasion_prob(a_type):
        sum_prob = 1
        for j in range(1,pop_size):
            dbr_prod = 1
            for i in range(1,j+1):
                dbr_prod *= death_birth_ratio(a_type,i)
            sum_prob += dbr_prod
        return 1.0/sum_prob

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

#print patterner(3,2)

def self_matches(n):
    if n<2:
        return 0
    return factorial(n)/factorial(n-2)

def testing():
    for i in range(9):
        part = set([p for p in all_partitions(i,3)])
        #print i,sum(self_matches(i[0]) for i in part)
        #print i,faces(i,3),sum(p[0] for p in part),len(set([p for p in all_partitions(i,3)])), partitions(i,3)
    print partitions(1000,3)*3
    print partitions(5,3)-partitions(5,2)
    print faces(5,3)-faces(5,2)
    print partitions(4,3),faces(4,3)



def invasion_probability(payoff, invader, dominant, pop_size,s=1):
    def f(count):
        return (1.0-s)+s*((count-1)*payoff[invader,invader]+(pop_size-count)*payoff[invader,dominant])/(pop_size-1.0)
    def g(count):
        return (1.0-s)+s*(count*payoff[dominant,invader]+(pop_size-count-1)*payoff[dominant,dominant])/(pop_size-1.0)
    accum = 1.0
    for i in reversed(range(1,pop_size)):
        accum *= g(i)/f(i)
        accum += 1
    return 1/accum

def invasion_matrix(payoff,pop_size):
    type_count = len(payoff)
    transition = np.zeros((type_count,)*2)
    #print payoff
    #print invasion_probability(payoff,0,1,pop_size)
    #print invasion_probability(payoff,1,0,pop_size)
    #assert False
    for dominant,invader in permutations(range(type_count),2):
        transition[dominant,invader] = invasion_probability(payoff,invader,dominant,pop_size)

    for i in range(type_count):
        transition[i,i] = 1-sum(transition[:,i])
        #print transition[:,i]
    return transition

def steady_state(matrix):
    vals,vecs = np.linalg.eig(matrix)
    [steady_states] = [vec for vec,val in zip(vecs.T,vals) if val == 1]

    #print matrix
    #print vals
    #print vecs
    #print steady_states
    return steady_states

def pop_graph(types,pop_size):
    composition_count = partitions(pop_size,types)
    id = np.identity(types)
    def get_neighbors(partition):
        partition = np.array(partition)
        return [tuple(partition+id[give]-id[take]) for give,take in permutations(range(types),2)]
    
def pop_transition_matrix(payoff,pop_size):
    type_count = len(payoff)
    partition_count = int(partitions(pop_size,type_count))
    I = np.identity(type_count)

    transition = np.zeros((partition_count+1,)*2)
    part_to_id = dict(map(reversed,enumerate(all_partitions(pop_size,type_count))))
    print part_to_id
    print partition_count
    print transition
    for i,pop in enumerate(all_partitions(pop_size,type_count)):
        fitnesses = [np.dot(pop-I[t],payoff[t]) for t in range(type_count)]
        total_fitness = sum(fitnesses)
        node = np.array(pop)
        for b,d in permutations(xrange(type_count),2):
            if pop[d]==0:
                pass
            else:
                neighbor = pop+I[b]-I[d]
                print i,part_to_id[tuple(neighbor)]
                death_odds = pop[d]/pop_size
                birth_odds = fitnesses[b]/total_fitness
                transition[i,part_to_id[tuple(neighbor)]]=death_odds*birth_odds

    for i in xrange(partition_count):
        transition[i,i] = 1-sum(transition[:,i])

    return transition

def complete_analysis(payoff,pop_size):
    type_count = len(payoff)
    partition_count = partitions(pop_size,type_count)
    part_to_id = dict(enumerate(all_partitions(pop_size,type_count)))
    transition = pop_transition_matrix(payoff,pop_size)
    ssd = steady_state(transition)
    pop_sum = np.zeros(type_count)
    for p,partition in zip(ssd,all_partitions(pop_size,type_count)):
        pop_sum += np.array(partition)*p
    print part_to_id[10]
    print pop_sum/(partition_count-1)

for i,b in product(xrange(7),xrange(3)):
    print i,b

print partitions(100,3)
a = np.array([
    [1,1],
    [1,1]
])
#print a

#for i in range(3):
#    print a[:,i]
complete_analysis(a,100)
#steady_state(invasion_matrix(a,50))

