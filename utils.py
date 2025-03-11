import numpy as np
import scipy as sp
import itertools
import random
import pandas as pd
from functools import wraps, reduce
from pickle import load, dump
from copy import deepcopy
from joblib import Memory
import os
import params

if params.memoized:
    CACHEDIR = os.path.expanduser('~/evocache/')
else:
    CACHEDIR = None
memory = Memory(location=CACHEDIR, verbose=0)

pd.set_option('display.precision', 5)

def splits(n):
    """
    starting from 0, produces the sequence 2, 3, 5, 9, 17...
    This sequence is the answer to the question 'if you have 2 points and find the midpoint,
    then find the midpoints of each adjacent pair of points, and do that over and over,
    how many points will you have after n iterations?'

    excellent for populating real-valued parameter ranges since they recycle points
    when used as
    np.linspace(min, max, splits(n))
    """
    assert n>=0
    i = n+1
    return int((2**i-0**i)/2 + 1)

def logspace(start = .001,stop = 1, samples=10):
    mult = (np.log(stop)-np.log(start))/np.log(10)
    plus = np.log(start)/np.log(10)
    return np.array([0]+list(np.power(10,np.linspace(0,1,samples)*mult+plus)))

def int_logspace(start, stop, samples, base=2):
    return sorted(list(set(np.logspace(start, stop, samples, base=base).astype(int))))

def softmax(vector, beta):
    ''' returns the softmax of the vector,'''
    if beta == np.Inf:
        e_x = np.array(vector) == max(vector)
        return e_x / e_x.sum()
    else:
        return sp.special.softmax(beta*vector)

def sample_softmax(utility, beta):
    
    return np.where(np.random.multinomial(1,softmax(utility, beta)))[0][0]

def softmax_utility(utility, beta):
    actions = list()
    utils = list()
    for a, u in list(utility.items()):
        actions.append(a)
        utils.append(u)

    return {a:u for a, u in zip(actions, softmax(utils, beta))}

def sample_hardmax(utility):
    # Hard-Max
    best_utility = max(utility.values())
    best_actions = list()
    for choice, u in utility.items():
        if u == best_utility:
            best_actions.append(choice)
            
    return random.choice(best_actions)

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def flip(p = 0.5):
    """return true with probability p"""
    return np.random.rand() < p

def normalized(array):
    return array / np.sum(array)

assert np.sum(normalized(np.array([.2,.3]))) ==1

def dict_hash(dict):
    return hash(tuple(sorted(dict.items())))

class HashableDict(dict):
    def __hash__(self):
        return dict_hash(self)

_issubclass = issubclass
def _issubclass(C,B):
    try:
        return issubclass(C,B)
    except TypeError as e:
        try:
            return _issubclass(C.type,B)
        except:
            return False

def excluding_keys(d,*keys):
    return dict((k,v) for k,v in d.items() if k not in keys)
