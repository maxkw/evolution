from __future__ import division
import numpy as np
import itertools
import random
import scipy as sp
from collections import Counter
import math

def softmax(vector, beta):
    ''' returns the softmax of the vector,'''
    e_x = np.exp(beta * (np.array(vector)-max(vector)))
    return e_x / e_x.sum()

def sample_softmax(utility, beta):
    
    return np.where(np.random.multinomial(1,softmax(utility, beta)))[0][0]

def softmax_utility(utility, beta):
    actions = list()
    utils = list()
    for a, u in utility.items():
        actions.append(a)
        utils.append(u)

    return {a:u for a, u in zip(actions, softmax(utils, beta))}

def sample_hardmax(utility):
    # Hard-Max
    best_utility = max(utility.values())
    best_actions = list()
    for choice, u in utility.iteritems():
        if u == best_utility:
            best_actions.append(choice)
            
    return random.choice(best_actions)

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def flip(p = 0.5):
    return np.random.rand() < p
