from __future__ import division
import numpy as np
import itertools
import random
import pandas as pd
from functools import wraps, reduce
from pickle import load, dump
from copy import deepcopy

pd.set_option('precision',5)

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
    else:
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

def memoize(obj):
    cache = obj.cache = {}

    @wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return deepcopy(cache[key])
    return memoizer

@memoize
def namedArrayConstructor(fields, className = "NamedArray"):
    """
    this takes a canonical sequence of hashable objects and returns a class of arrays
    where each of the elements in the first dimension of the array can be indexed 
    using the object in the corresponding position in the canonical sequence

    these classes are closed on themselves, operating two arrays with the same seed sequence 
    will produce an array that can be indexed using the same sequence.
   
    Note: This constructor should always be memoized to avoid creation of functionally identical classes
    """
    fields = tuple(fields)
    assert len(fields) == len(set(fields))

    fields_error_msg = "\nThe following are also valid indices: "+", ".join(str(field) for field in fields)
    reference = dict(map(reversed,enumerate(fields)))
    repr_expr = '{name}[{seq}]'.format(
        name = className,
        seq = ", ".join("{}={{:.2f}}".format(field) for field in fields))
    class NamedArray(np.ndarray):
        """
        these arrays function exactly the same as normal np.arrays
        except that they can also be indexed using the elements provided to the constructor
        in short, if we assume that the sequence 'fields' contains the element 'field'

        namedArray = namedArrayConstructor(fields)

        array = namedArray(sequence)

        array[fields.index(field)] == array[field]
        """
        #def __repr__(self):
        #    return np.array(self).__repr__()
        def __new__(self, seq):            
            return np.asarray(seq).view(self)
        
        #def __init__(self,seq,array_name = default_name):
        #    self.__reset_repr_expr__()

        def __reset_repr_expr__(self):
            repr_expr = '{name}[{seq}]'.format(
                name = self.array_name,
                seq = ", ".join("{}={{:.2f}}".format(field) for field in fields))

        def __str__(self):
            try:
                return repr_expr.format(*self)
            except:
                return super(NamedArray,self).__str__()
        
        def __getitem__(self,*args,**kwargs):
            try:
                return super(NamedArray,self).__getitem__(*args,**kwargs)
            except IndexError as indexError:
                try:
                    [keys] = args
                    try:
                        keys = reference[keys]
                    except TypeError:
                        keys = [reference[key] for key in keys]
                    return super(NamedArray,self).__getitem__(keys)
                
                except KeyError:
                    indexError = type(indexError)(indexError.message+fields_error_msg) 
                    raise indexError

        def __setitem__(self,*args,**kwargs):
            try:
                super(NamedArray,self).__setitem__(*args,**kwargs)
            except IndexError as indexError:
                try:
                    keys,vals = args
                    try:
                        keys = reference[keys]
                    except TypeError:
                        keys = [reference[key] for key in keys]  
                    super(NamedArray,self).__setitem__(keys,vals,**kwargs)
                
                except KeyError:
                    indexError = type(indexError)(indexError.message+fields_error_msg) 
                    raise indexError
                                
        def rename(self,name):
            self.array_name = name
            self.__reset_repr_expr__()

    return NamedArray


def pickled(obj,path,mode = "w"):
    with open(path,mode) as file:
        dump(obj,file)
        return obj
def unpickled(path,mode = "r"):
    with open(path,mode) as file:
        return load(file)

def normalized(array):
    return array / np.sum(array)

assert np.sum(normalized(np.array([.2,.3]))) ==1

def dict_hash(dict):
    return hash(tuple(sorted(dict.iteritems())))

class HashableDict(dict):
    def __hash__(self):
        return dict_hash(self)

def dict_to_kwarg_str(d):
    return "(%s)" % ",".join(["%s=%s" % (key,val) for key,val in d.iteritems()])

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
    return dict((k,v) for k,v in d.iteritems() if k not in keys)

def justcaps(t):
    return filter(str.isupper,t.__name__)

def compose(*funcs):
    return reduce(lambda f,g: lambda x: f(g(x)), funcs, lambda x: x)

