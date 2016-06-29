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
    """return true with probability p"""
    return np.random.rand() < p

def memoized(f):
    """ Memoization decorator for a function taking one or more arguments. """
    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret
            
    return memodict().__getitem_

@memoized
def namedArrayConstructor(fields, className = "NamedArray"):
    """
    this takes a canonical sequence of hashable objects and returns a class of arrays
    where each of the elements in the first dimension of the array can be indexed 
    using the object in the corresponding position in the canonical sequence

    these classes are closed on themselves, operating two arrays with the same seed sequence 
    will produce an array that can be indexed using the same sequence.
   
    Note: This constructor should always be memoized to avoid creation of functionally identical classes
    """

    assert len(fields) == len(set(fields))
    
    reference = dict(map(reversed,enumerate(fields)))
    class NamedArray(np.ndarray):
        """
        these arrays function exactly the same as normal np.arrays
        except that they can also be indexed using the elements provided to the constructor
        in short, if we assume that the sequence 'fields' contains the element 'field'

        namedArray = namedArrayConstructor(fields)

        array = namedArray(sequence)

        array[fields.index(field)] == array[field]
        """
        
        default_name = className
        def __new__(self, seq, array_name = default_name):
            self.array_name = array_name
            self.__reset_repr_expr__()
            return np.asarray(seq).view(self)

        def __reset_repr_expr__(self):
            self.repr_expr = '{name}[{seq}]'.format(
                name = self.array_name,
                seq = ", ".join("{}={{:.2f}}".format(field) for field in fields))
        
        def __getitem__(self,*keys):
            try:
                return super(NamedArray,self).__getitem__(*keys)
            except IndexError:
                if not all(key in reference for key in keys):
                    raise
                else:
                    keys = (reference[key] for key in keys)
                    return super(NamedArray,self).__getitem__(*keys)#reference[keys[0]])
            
        def __repr__(self):
            return self.repr_expr.format(*self)

        def __str__(self):
            return self.__repr__()

        def __setitem__(self,*keys,**vals):
            try:
                super(NamedArray,self).__setitem__(*keys,**vals)
            except IndexError:
                keys = (reference[keys[0]],)+keys[1:]
                super(NamedArray,self).__setitem__(*keys,**vals)
                
        def rename(self,name):
            self.array_name = name
            self.__reset_repr_expr__()

    return NamedArray

def normalized(array):
    return array/np.sum(array)
assert np.sum(normalized(np.array([.2,.3]))) ==1
