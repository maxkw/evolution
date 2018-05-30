import os.path
import pandas as pd
import collections
from itertools import product
from inspect import getargspec
from collections import OrderedDict
import numpy as np
from copy import copy,deepcopy
from utils import pickled, unpickled
from operator import itemgetter
import matplotlib.pyplot as plt
import sys
from functools import wraps
from copy import deepcopy

def is_sequency(obj):
    if isinstance(obj,basestring):
        return False
    return isinstance(obj,(collections.Sequence,np.ndarray))

class multiRefOrderedDict(OrderedDict):
    def __getitem__(self,keys):
        if is_sequency(keys):
            return [self[key] for key in keys]
        else:
            return dict.__getitem__(self,keys)

def dict_intersect(dict1,dict2):
    return {key:val for key,val in dict1.iteritems() if key in dict2}

def product_of_vals(orderedDict):
    keys,val_lists = orderedDict.keys(),orderedDict.values()
    return [OrderedDict(zip(keys,vals)) for vals in apply(product,val_lists)]

def dict_hash(d):
    return hash(str(tuple(sorted(d.iteritems()))))

class hashableDict(dict):
    def __hash__(self):
        return dict_hash(self)

def fun_call_labeler(method,args,kwargs,intolerant = True):
    """
    given a defined method to be called as 'method(*args,**kwargs)'
    this function returns 3 dicts where the keys are the argument names and the vals those provided
    the first contains all args known at function call, this may even include kwargs not defined by it
    the second contains only those arguments specified in the function definition
    the third contains only those arguments provided but not specified in the definition
    """

    try:
        arg_names,varargs,keywords,default_values = method.__getargspec__
    except:
        arg_names,varargs,keywords,default_values = method.__getargspec__ = getargspec(method)
    given_values = args

    #the dict is first populated with the default values if there are any
    known_args = {}
    try:
        #zip from the bottom up to properly align
        known_args.update(apply(zip,map(reversed,[arg_names,default_values])))
    except:
        #default_values might be None
        pass

    #whether any are provided or not, then they are overwritten here
    known_args.update(kwargs)

    #overwrite only happens for given_values
    known_args.update(zip(arg_names,given_values))

    #if there are fewer than the expected number of arguments
    #call the wrapped function and let it handle the exception

    expected_arg_missing = not all(name in known_args for name in arg_names)
    if expected_arg_missing and intolerant:
        method(**known_args)

    expected_args = [(k,v) for k,v in known_args.iteritems() if k in arg_names]
    defined_args = OrderedDict(sorted(expected_args, key = lambda(k,v): arg_names.index(k)))
    #defined_args = OrderedDict((arg,known_args[arg]) for arg in arg_names)
    undefined_args = OrderedDict((k,v) for k,v in sorted(known_args.iteritems(),key=itemgetter(0)) if k not in defined_args)
    known_args = OrderedDict(defined_args.items()+undefined_args.items())
    call_data =  {
        "args":known_args,
        "defined_args":defined_args,
        "undefined_args":undefined_args,
        "valid_args":known_args if keywords else defined_args,
        "arg_names":arg_names,
        "varargs":varargs,
        "keywords":keywords,
        "default_values":default_values,
        "defined_call": method.__name__+"(%s)" % ",".join(["%s=%s" % (key,val) for key,val in defined_args.iteritems()]),
        "call": method.__name__+"(%s)" % ", ".join(["%s=%s" % (key,val) for key,val in known_args.iteritems()]),
        "make_call_str" : lambda d: method.__name__+"(%s)" % ", ".join(["%s=%s" % (key,val) for key,val in d.iteritems()])
    }
    return call_data


class MultiArg(list):
    def __init__(self,args):
        list.__init__(self,args)

def unordered(maybe_seq):
    try:
        return tuple(sorted(maybe_seq, key = lambda x: repr(x[0])))
    except:
        try:
            return tuple(sorted(maybe_seq))
        except TypeError:
            return maybe_seq

def twinned(maybe_pair):
    try:
        maybe_pair[1]
        return tuple(maybe_pair[:2])
    except IndexError:
        return (maybe_pair[0],)*2
    except TypeError:
        return (maybe_pair,)*2

def dict_key_map(d,fun_to_arg):
    mapped_d = deepcopy(d)
    for arg_name, arg_val in d.iteritems():
        for function, arg_names in fun_to_arg.iteritems():
            if arg_name in arg_names:
                if isinstance(arg_val, MultiArg):
                    mapped_d[arg_name] = MultiArg(map(function,arg_val))
                else:
                    mapped_d[arg_name] = function(arg_val)
    return mapped_d

def transform(functions):
    name_2_fun = {f.__name__:f for f in functions}
    def dec_wrapper(dec):
        def dec_call(*args,**kwarg):
            dec_call_data = fun_call_labeler(dec,*args,**kwargs)
            dec_args = dec_call_data['args']
            for k,v in dec_args.iteritems():
                if k in name_2_fun:
                    {name_2_fun[k]:arg_dict[k] for k in arg_dict if k in name_2_fun}




def copy_function_identity(parent,decorator = None):
    def inherit_attributes(inheritor):
        inheritor.__name__ = parent.__name__
        try:
            inheritor.__getargspec__ = parent.__getargspec__
        except:
            inheritor.__getargspec__ = getargspec(parent)
        if decorator:
            inheritor._decorator = decorator
        return inheritor
    return inherit_attributes


def transform_inputs(*functions):
    """
    takes a list of functions returns a function that when given a dictionary
    with the name of a provided function as a key and a list of argument names
    returns a function
    """
    name_to_fun = {f.__name__:f for f in functions}
    def transformer(dec_kwargs):
        fun_to_arg_names = {}
        for k,v in dec_kwargs.iteritems():
            if k in name_to_fun:
                fun_to_arg_names[name_to_fun[k]] = v
        def transform_args(arg_dict):
            return dict_key_map(arg_dict,fun_to_arg_names)
        return transform_args
    return transformer

experiment_transformer = transform_inputs(tuple,unordered)
def experiment(unpack = False, trials = None, overwrite = False, memoize = True, verbose = 0,**kwargs):
    data_dir = './memo_cache/'
    default_trials = trials

    transform_arg_dict = experiment_transformer(kwargs)
    def wrapper(function):
        @copy_function_identity(function)
        def experiment_call(*args,**kwargs):
            memoized = kwargs.get('memoized', memoize)

            for k in ['memoized']:
                if k in kwargs:
                    del kwargs[k]
            call_data = fun_call_labeler(function,args,kwargs)
            return_keys = call_data['args'].get('return_keys', None)
            original_call = copy(call_data)
            try:
                del call_data['args']['return_keys']
                call_data = fun_call_labeler(function,[],call_data['args'])
            except:
                pass
            arg_dict = call_data['args']

            if default_trials:
                try:
                    trials = range(arg_dict['trials'])
                except TypeError:
                    trials = arg_dict['trials']
                except KeyError:
                    trials = range(default_trials)

                try:
                    del arg_dict['trials']
                    call_data = fun_call_labeler(function,[],arg_dict)
                except:
                    pass
            else:
                trials = [0]


            args = experiment_call._last_args = transform_arg_dict(call_data['valid_args'])

            try:
                if return_keys:
                    arg_hash = dict_hash(dict(args,**{'return_keys':return_keys}))
                else:
                    arg_hash = dict_hash(args)
                if verbose >=1:
                    print "\nExperiment "+str(call_data['call'])#+"\n"+str(arg_hash)
            except TypeError as te:
                print "these are the provided args\n",args
                raise te

            if memoize and not os.path.exists(data_dir):
                os.makedirs(data_dir)

            
            cache_file =data_dir+str(arg_hash)+".pkl"
            if default_trials:
                try:
                    assert memoized and not overwrite
                    print "Loading cache...",
                    cache = pd.read_pickle(cache_file)
                    cached_trials = list(cache['trial'].unique())
                    uncached_trials = [trial for trial in trials if trial not in cached_trials]

                    #import pdb;pdb.set_trace()

                    print "cache loaded! cached:%s, need to compute:%s" % (len(cached_trials),len(uncached_trials))
                    
                except Exception as e:
                    print "No cache loaded."
                    if 'trial' in args.keys():
                        cols = args.keys()
                    else:
                        cols = args.keys()+['trial']
                    cache = pd.DataFrame(columns = cols)
                    uncached_trials = trials
            else:
                try:
                    assert memoized and not overwrite
                    print "Loading cache...",
                    cache = pd.read_pickle(cache_file)
                    uncached_trials = []
                except Exception as e:
                    print "No cache loaded."
                    cols = args.keys()
                    cache = pd.DataFrame(columns = cols)
                    uncached_trials = [0]


            #print arg_hash
            #print cached_trials
            #print uncached_trials
            #assert 0

            call_args = copy(args)
            results = []
            total_calls = float(len(uncached_trials))
            landmark = step = .1
            total_ticks = 63
            if verbose == 3:
                pass
            for n,trial in enumerate(uncached_trials):
                np.random.seed(trial)
                if "trial" in call_args:
                    print args
                    assert 0
                result = function(**copy(call_args))
                if default_trials:
                    args['trial'] = trial
                if not unpack:
                    results.append(dict(args,**{'result':result}))
                elif unpack == 'dict':
                    if return_keys:
                        results.append(dict(args,**{k:v for k,v in result.iteritems() if k in return_keys}))
                    else:
                        results.append(dict(args,**result))
                elif unpack == 'record':
                    for d in result:
                        if return_keys:
                            results.append(dict(args,**{k:v for k,v in d.iteritems() if k in return_keys}))
                        else:
                            results.append(dict(args,**d))

                if verbose == 3:
                    ticks = int(total_ticks*(n+1)/total_calls)
                    bar = 'trial progress ['+'='*ticks+' '*(total_ticks-ticks)+']'
                    sys.stdout.write("\r%s"%bar)
                    sys.stdout.flush()
                if n/total_calls >= landmark:
                    if verbose ==2:
                        print "%s/%s trials completed" % (n, int(total_calls))
            if verbose == 3:
                if len(uncached_trials)==0:
                    print "Trials loaded from cache!"
                else:
                    print ""
            #consolidate new and old results and save
            cache = pd.concat([cache,pd.DataFrame(results)])
            if memoized:
                cache.to_pickle(cache_file)
            
            #return only those trials that were asked for
            if default_trials:
                return cache.query('trial in %s' % trials)
            else:
                return cache
        experiment_call._decorator = 'experiment'
        return experiment_call
    return wrapper


def multi_call(unpack = False, verbose = 2, **kwargs):

    transform_arg_dict = experiment_transformer(kwargs)
    def wrapper(function):
        """
        the keys of the OrderedDict are the names of the arguments,
        the values are the values that ended up being passed in
        whether or not an argument was called with default values or with keywords is not recorded
        likewise any change in the order in which arguments are provided, in the case of keywords
        is not preserved.

        saves named arguments to object.init_args
        for every arg in arglist stores given value in object.arg
        calls super init with extra kwargs
        """

        decorator = None
        try:
            decorator = function._decorator
        except:
            pass

        unpack_ = unpack

        if decorator == 'experiment':
            unpack_ = 'DataFrame'

        @copy_function_identity(function)
        def m_call(*args,**kwargs):
            if verbose and verbose>0:
                pass
                #print "Expanding multicall for function "+function.__name__
            call_data = fun_call_labeler(function,args,kwargs)

            expected_args = call_data['valid_args']
            expected_args = transform_arg_dict(expected_args)

            m_call._last_args = expected_args

            dynamic_args={}; static_args={}
            for key,val in expected_args.iteritems():
                if isinstance(val,MultiArg):
                    dynamic_args[key] = val
                else:
                    static_args[key] = val

            if dynamic_args:
                arg_calls = [dict(static_args,**d_args) for d_args in product_of_vals(dynamic_args)]
            else:
                arg_calls = [static_args]
                return function(**static_args)

            if verbose and verbose>=1:
                print " %s calls..." % len(arg_calls)
                print "Processing %s calls..." % len(arg_calls)
            if unpack_ == 'DataFrame':
                dfs = []
                total_calls = float(len(arg_calls))
                landmark = step = .1
                total_ticks = 63
                for n,arg_call in enumerate(arg_calls,start = 1):
                    dfs.append(function(**arg_call))
                    if verbose == 3:
                        ticks = int(total_ticks*n/total_calls)
                        bar = 'call progress  ['+'='*ticks+' '*(total_ticks-ticks)+']'
                        sys.stdout.write("\r%s"%bar)
                        sys.stdout.flush()
                    if n/total_calls >= landmark:
                        landmark += step
                        if verbose == 2:
                            print "%s/%s of calls computed" % (n,int(total_calls))
                if verbose == 3:
                    print ""
                ret = pd.concat(dfs)#[function(**arg_call) for arg_call in arg_calls])
            else:
                raise NotImplemented
            return ret

        #call.__name__ = function.__name__
        #call.__getargspec__ = function.__getargspec__
        return m_call
    return wrapper

def or_query(field,ls):
    return " or ".join(["%s == %s" % (field, i) for i in ls])

def plotter(experiment, 
            default_plot_dir="./plots/",
            default_file_name = None,
            default_extension = ".pdf",
            experiment_args=[],
            plot_exclusive_args = ['data', 'graph_kwargs', 'stacked']):
    def wrapper(plot_fun):
        try:
            assert 'data' in getargspec(plot_fun)[0]
        except:
            print "wrapped function must have a 'data' argument"
            raise
        default_experiment = experiment
        def make_arg_dicts(args,kwargs):
            try:
                plot_fun_call_data = fun_call_labeler(plot_fun,args,kwargs)
            except TypeError as e:
                incomplete_plot_fun_args = fun_call_labeler(plot_fun,args,kwargs,intolerant = False)['args']
                #if 'experiment' in incomplete_plot_fun_args:
                #    experiment = plot_fun_call_data['args']['experiment']
                #else:
                #    experiment = dexperiment
                experiment = incomplete_plot_fun_args.get('experiment', default_experiment)
                fun_args = fun_call_labeler(experiment,[],incomplete_plot_fun_args)['args']
                try:
                    if not fun_args['data']:
                        del fun_args['data']
                except:
                    pass
                plot_fun_call_data = fun_call_labeler(plot_fun,[],fun_args)
            return plot_fun_call_data
        
        def call(*args, **kwargs):
            plot_dir = kwargs.pop("plot_dir",default_plot_dir)
            extension = kwargs.pop("extension",default_extension)
            file_name = kwargs.pop('file_name',default_file_name)
            plot_trials = kwargs.pop('plot_trials', False)
            close = kwargs.pop('close',True)

            #if not plot_name:
            #    save_file = plot_fun.__name__+"(%s).pdf"
            #else:
            #    save_file = plot_name+".pdf"
            call_data = make_arg_dicts(args,kwargs)
            plot_args = call_data['valid_args']
            if False:#type(call_data['valid_args'].get('data',[])) == pd.DataFrame:
                ret = plot_fun(**call_data['valid_args'])
            else:
                fun = call_data['args'].get('experiment',experiment)

                if plot_exclusive_args:
                    try:
                        experiment_args = {k:v for k,v in call_data['args'].iteritems() if k not in plot_exclusive_args}
                    except TypeError:
                        given_args = call_data['args']
                        experiment_argnames = get_arg_dicts(fun,[],given_args)['defined_args']
                        plot_exclusive_argnames = [k for k in given_args
                                                   if k in call_data['defined_args'] and
                                                   k not in experiment_argnames.keys()+experiment_args]
                        experiment_args = dict((key,val) for key,val in given_args.items() if key not in plot_exclusive_argnames)
                    experiment_call_data = fun_call_labeler(fun,[],experiment_args)
                    call_args = experiment_call_data['valid_args']
                else:
                    call_args = get_arg_dicts(fun,[],call_data['args]'])['valid_args']
                if 'experiment' in call_args:
                    call_args['experiment'] = call_args['experiment'].__name__
                data = fun(**call_args)
                plot_args = call_data['valid_args']

                try:
                    for key,val in fun._last_args.iteritems():
                        if key in plot_args:
                            plot_args[key] = val
                except:
                    pass

            if type(call_data['valid_args'].get('data',[])) == pd.DataFrame:
                data = call_data['valid_args']['data']

            ret = plot_fun(**dict(plot_args,**{'data':data}))

            if not file_name:
                file_name = call_data['make_call_str'](call_args)

            if plot_trials:
                trial_plot_dir = plot_dir+file_name+"/"
                for t,d in data.groupby('trial'):
                    call(*args,**dict(kwargs,**dict(data=d,
                                                    plot_dir=trial_plot_dir,
                                                    file_name = str(int(t)),
                                                    extension = extension,
                                                    plot_trials = False)))

            save_file = plot_dir+file_name+extension
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            try:
                plt.savefig(save_file)
            except IOError as e:
                print e
                print file_name
                new_file_name = raw_input("Automatic Filename Too Long. Enter new one:")
                save_file = plot_dir+new_file_name+extension
                plt.savefig(save_file)

            if close:
                plt.close()

            return ret
        call.make_arg_dicts = make_arg_dicts
        return call
    return wrapper

def get_arg_dicts(fun,args,kwargs):
    try:
        return fun.make_arg_dicts(args,kwargs)
    except Exception as e:
        print e
        return fun_call_labeler(fun,args,kwargs)

class Decorator(object):
    def __init__(self,*args,**kwargs):
        pass

    def __call__(self,function):
        self.decorated = function
        if isinstance(function,Decorator):
            self.base_fun = function.base_fun
        self.call_with_arg_dicts.__dict__['make_arg_dicts'] = self.make_arg_dicts
        return self.call_with_arg_dicts

    def call_with_arg_dicts(self,*args,**kwargs):
        return self.call(self.make_arg_dicts(args,kwargs))
    def make_arg_dicts(self,args,kwargs):
        return fun_call_labeler(self.decorated,args,kwargs)

    def call(self,call_data):
        return self.decorated(**call_data['valid_args'])


def dict_key_map(d,fun_to_arg):
    mapped_d = deepcopy(d)
    for arg_name, arg_val in d.iteritems():
        for function, arg_names in fun_to_arg.iteritems():
            if arg_name in arg_names:
                if isinstance(arg_val, MultiArg):
                    mapped_d[arg_name] = MultiArg(map(function,arg_val))
                else:
                    mapped_d[arg_name] = function(arg_val)
    return mapped_d

def apply_to_args(**f_to_argnames):
    f_to_argnames = {eval(name):argnames for name,argnames in f_to_argnames.iteritems()}
    def wrapper(function):
        @copy_function_identity(function)
        def call(*args,**kwargs):
            raw_arg_dict = get_arg_dicts(function,args,kwargs)['args']
            arg_dict = dict_key_map(raw_arg_dict,f_to_argnames)
            return function(**arg_dict)
        return call
    return wrapper

def save_prompt(obj,path):
    print obj
    save = input("save this object to "+ path+"?")

    if save:
        print "saving object"
        try:
            record = unpickled(path)
            record.append(obj)
            pickled(record,path)
        except:
            pickled([obj],path)
    else:
        print "object not saved"

def literal(constructor):
    def call(*args,**kwargs):
        fun_call_string = fun_call_labeler(constructor,args,kwargs)['defined_call']
        call.__name__ = constructor.__name__
        ret = constructor(*args,**kwargs)
        ret.name = ret._name = fun_call_string
        return ret
    return call



if __name__ == "__main__":
    a = MultiArg(i for i in range(5))
    print isinstance(a,MultiArg)
    #@multi_call()
    #def add(a = range(5), b = range(5)):
    #    return a+b
    #add()
    #@multi_call(static='factors')
    #def prod(factors=[7,5]):
    #    res = 1
    #    for factor in factors:
    #        res *= factor
    #    return res
    #prod()
    #prod([5,7])
"end"
