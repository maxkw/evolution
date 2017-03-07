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

### for experiments
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

def dict_hash(dict):
    return hash(tuple(sorted(dict.iteritems())))

def fun_call_labeler(method,args,kwargs):
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
    if len(known_args) < len(arg_names):
        method(*args,**kwargs)

    defined_args = OrderedDict((arg,known_args[arg]) for arg in arg_names)
    undefined_args = OrderedDict((k,v) for k,v in sorted(known_args.iteritems(),key=itemgetter(0)) if k not in defined_args)
    known_args = OrderedDict(defined_args.items()+undefined_args.items())

    return {"args":known_args,
            "defined_args":defined_args,
            "undefined_args":undefined_args,
            "valid_args":known_args if keywords else defined_args,
            "arg_names":arg_names,
            "varargs":varargs,
            "keywords":keywords,
            "default_values":default_values,
            "defined_call": method.__name__+"(%s)" % ",".join(["%s=%s" % (key,val) for key,val in defined_args.iteritems()]),
            "call": method.__name__+"(%s)" % ",".join(["%s=%s" % (key,val) for key,val in known_args.iteritems()])
    }

def multi_call(static=[],unordered=[], unpack = False, verbose = 2):
    def wrapper(method):
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
        data_dir = "memo_cache/"
        data_file = data_dir + method.__name__+'.pkl'
        def call(*args,**kwargs):
            call_data = fun_call_labeler(method,args,kwargs)
            expected_args = call_data["defined_args"]
            keywords = call_data["keywords"]

            try:
                #trials is a number
                trials = range(expected_args["trial"])
            except KeyError:
                #trials isn't even provided as a parameter
                trials = [0]
            except:
                #trials is a list of trials to be run
                trials = expected_args["trial"]

            try:
                del expected_args["trial"]
            except KeyError:
                pass

            assert 'trial' not in expected_args


            #identify items that are sequences
            dynamic_args={};static_args={}
            for key,val in expected_args.iteritems():
                if is_sequency(val):
                    if key in static:
                        if key in unordered:
                            try:
                                static_args[key] = tuple(sorted(val))
                            except TypeError as e:
                                print "If an argument is set as both 'static' and 'unordered' it must be a sequence"
                                raise e
                        else:
                            try:
                                static_args[key] = tuple(val)
                            except TypeError:
                                sattic_args[key] = val
                    else:
                        if key in unordered:
                            dynamic_args[key] = list(set([tuple(sorted(v)) for v in val]))
                        else:
                            dynamic_args[key] = val

                else:
                    static_args[key]=val
            #dynamic_args = {key:val for key,val  in expected_args.items() if is_sequency(val) and not in static}
            #static_args = {key:val for key,val in expected_args.items() if key not in dynamic_args}

            #orderless_dynamic = [key for key in dynamic_args.iterkeys() if key in orderless]
            #orderless_static = [key for key in static_args.iterkeys() if key in orderless]

            def updated(dict1,dict2):
                ret = copy(dict1)
                ret.update(dict2)
                return ret

            if dynamic_args:
                all_arg_calls = [updated(static_args,d_args) for d_args in product_of_vals(dynamic_args)]
            else:
                all_arg_calls = [static_args]


            #file to queriable
            #queriable x dict -> bool
            #queriable x dicts -> queriable
            #queriable to file

            #loads the cache from pkl, if no cache exists, make one
            try:
                print "Attempting to load cache from: ",data_file
                cache = pd.read_pickle(data_file)
                print "Cache successfuly loaded, contains %s computed calls." % len(cache)
            except Exception as e:
                print str(e)
                print "First time running, creating new cache..."
                arg_names = call_data["arg_names"]
                if 'trials' in arg_names:
                    cache =  pd.DataFrame(columns = all_arg_calls[0].keys()+["return","arg_hash"])
                else:
                    cache =  pd.DataFrame(columns = all_arg_calls[0].keys()+["trial","return","arg_hash"])
                cache["arg_hash"] = cache["arg_hash"].astype(np.int64)
                print "...done"

            arg_hashes = map(dict_hash,all_arg_calls)
            uncomputed_arg_calls = []; append_calls =  uncomputed_arg_calls.extend
            for arg_hash, arg_call in zip(arg_hashes,all_arg_calls):
                #print arg_call
                #print arg_hash
                #print cache.query('arg_hash == %s' % arg_hash)
                cached_trials = list(cache.query('arg_hash == %s' % arg_hash)['trial'])
                uncached_trials = [trial for trial in trials if trial not in cached_trials]
                assert 'trial' not in arg_call
                try:

                    assert (not uncached_trials) or len(list(product(uncached_trials,[(arg_hash,arg_call)]))) == len(trials)
                except:
                    "If there are any uncached trials"
                    it  = list(product(uncached_trials,[(arg_hash,arg_call)]))
                    #print "the thing"
                    #print it
                    #print uncached_trials

                if uncached_trials:
                    append_calls(product(uncached_trials,[(arg_hash,arg_call)]))

            results = []
            number_of_calls = len(uncomputed_arg_calls)

            if verbose:
                print "There are", number_of_calls, "uncomputed method calls out of",len(all_arg_calls)*len(trials)

            for n,(trial,(arg_hash,arg_call)) in enumerate(uncomputed_arg_calls):
                try:
                    arg_call = deepcopy(arg_call)
                except Exception as e:
                    print arg_call
                    print e
                    raise e
                if verbose>1:
                    print n+1,"/",number_of_calls
                    print "current arg call:",arg_call
                
                np.random.seed(trial)

                if keywords:
                    reference = copy(arg_call)
                    result = method(trial = trial, expected_args = reference,**arg_call)

                else:
                    try:
                        #print arg_call
                        result = method(trial = trial,**arg_call)
                    except Exception as e:
                        result = method(**arg_call)

                if verbose>1:
                    print "returned:",result

                arg_call.update({"arg_hash":arg_hash, "trial":trial})
                if not unpack:
                    arg_call['return'] = result
                    results.append(arg_call)
                elif unpack == 'dict':
                    arg_call.update(result)
                    results.append(arg_call)
                elif unpack == 'record':
                    for record in result:
                        new_arg_call = deepcopy(arg_call)
                        new_arg_call.update(record)
                        results.append(new_arg_call)
            if results:
                #make a df with the results
                to_cache = pd.DataFrame(results)

                #check that we can actually find our entries once they're in pd format
                #if one of our results can't be queried out of the new pd, that's a problem
                for result in results:
                    try:
                        assert not to_cache.query('arg_hash == %s and trial == %s' % (result['arg_hash'],result['trial'])).empty
                    except:
                        print "SOMETHING WENT WRONG"
                        print result
                        print to_cache

                #actually add the results to the existing table
                cache = cache.append(to_cache)#ignore_index=True)

                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                try:
                    cache.to_pickle(data_file)
                except Exception as e:
                    print e
                    print cache
                    raise e
            #print arg_hashes
            #print "cache",cache
            #print 'arg_hash == %s' % arg_hashes

            # result = cache.query(or_query('arg_hash',arg_hashes)).query('trial in %s' % trials) #
            
            #print cache['arg_hash']
            #print type(arg_hashes[0])
            #print arg_hashes
            #result = cache[cache.isin({'arg_hash': arg_hashes})]
            #result = cache.query('arg_hash in %s' % map(float,arg_hashes))
            result = cache.query('arg_hash in %s' % arg_hashes).query('trial in %s' % trials)
            #import pdb; pdb.set_trace()
            #result = cache[cache.arg_hash.isin(arg_hashes)]
            #print result
            print ""
            return result
        call.__name__ = method.__name__
        call.__getargspec__ = getargspec(method)
        return call
    return wrapper




def multi_call_2(static=[],unordered=[], unpack = False, verbose = 2):
    #print verbose
    def wrapper(method):
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
        data_dir = "memo_cache/"+method.__name__+"/"
        data_file = data_dir + method.__name__+'.pkl'
        
        def call(*args,**kwargs):
            print "Calling function "+method.__name__
            call_data = fun_call_labeler(method,args,kwargs)
            expected_args = call_data["defined_args"]
            keywords = call_data["keywords"]

            try:
                #trials is a number
                trials = range(expected_args["trial"])
            except KeyError:
                #trials isn't even provided as a parameter
                trials = [0]
            except:
                #trials is a list of trials to be run
                trials = expected_args["trial"]

            try:
                del expected_args["trial"]
            except KeyError:
                pass

            #identify items that are sequences
            dynamic_args={};static_args={}
            for key,val in expected_args.iteritems():
                if is_sequency(val):
                    if key in static:
                        if key in unordered:
                            try:
                                static_args[key] = tuple(sorted(val))
                            except TypeError as e:
                                print "If an argument is set as both 'static' and 'unordered' it must be a sequence"
                                raise e
                        else:
                            try:
                                static_args[key] = tuple(val)
                            except TypeError:
                                sattic_args[key] = val
                    else:
                        if key in unordered:
                            dynamic_args[key] = list(set([tuple(sorted(v)) for v in val]))
                        else:
                            dynamic_args[key] = val

                else:
                    static_args[key]=val

            def updated(dict1,dict2):
                ret = copy(dict1)
                ret.update(dict2)
                return ret
            def hash_2_filename(h):
                return data_dir+str(h)+".pkl"

            if dynamic_args:
                all_arg_calls = [updated(static_args,d_args) for d_args in product_of_vals(dynamic_args)]
            else:
                all_arg_calls = [static_args]
            arg_hashes = map(dict_hash,all_arg_calls)

            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            to_compute = 0
            new_args = 0
            process_list = []
            print "Processing %s calls, %s trials per call..." % (len(all_arg_calls),len(trials))
            print "Loading caches..."
            n_trials = len(trials)
            for arg_hash,arg_call in zip(arg_hashes,all_arg_calls):
                
                try:
                    cache = pd.read_pickle(data_dir+str(arg_hash)+".pkl")
                    
                    cached_trials = list(cache['trial'])
                    uncached_trials = [trial for trial in trials if trial not in cached_trials]
                    #print "\t...cache %s found, contains %s calls. %s/%s trials computed." % (arg_hash,len(cached_trials),n_trials-len(uncached_trials),n_trials)
                except Exception as e:
                    print str(e)
                    cache = pd.DataFrame(columns = all_arg_calls[0].keys()+['trial'])
                    #print "\t...hash:%s not found, new cache created." % arg_hash
                    new_args += 1
                    uncached_trials = trials
                #print "new cache",cache
                to_compute += len(uncached_trials)

                process_list.append((arg_hash,arg_call,cache,uncached_trials))

            #print "There are %s new calls." % new_args
            print "There are %s/%s total uncomputed calls." % (to_compute,len(trials)*len(all_arg_calls))
            ret = []
            for arg_hash,arg_call,cache,uncached_trials in process_list:
                if uncached_trials:
                    #print "Running %s trials of condition:\n\t" % len(uncached_trials), arg_call#fun_call_labeler(method,arg_call)['defined_call']
                    results =[]
                    for trial in uncached_trials:
                        np.random.seed(trial)
                        args = copy(arg_call)

                        if keywords:
                            #if it has a kwargs then stuff the arg call in there
                            result = method(trial = trial, expected_args = args, **args)
                        else:
                            try:
                                #try to pass in the trial keyword
                                result = method(trial = trial, **args)
                            except Exception as e:
                                #the function in question doesn't need a trial keyword
                                result = method(**args)
                        
                        #stuff the 'arg_hash' and 'trial' fields in there
                        args.update({"arg_hash":arg_hash, "trial":trial})
                        if not unpack:
                            args['return'] = result
                            results.append(args)
                        elif unpack == 'dict':
                            args.update(result)
                            #print args.keys()
                            results.append(args)
                        elif unpack == 'record':
                            for record in result:
                                new_args = deepcopy(args)
                                new_args.update(record)
                                results.append(new_args)
                    #add new results to the cache
                    cache = pd.concat([cache,pd.DataFrame(results)])
                    #overwrite the old cache file
                    cache.to_pickle(hash_2_filename(arg_hash))
                    #add cache to list of caches to be returned
                ret.append(cache)

            #concatenate all caches into one large dataframe
            ret = pd.concat(ret)
            #narrow it down to only the trials you need
            ret = ret.query('trial in %s' % trials)
            return ret
        call.__name__ = function.__name__
        call.__getargspec__ = getargspec(function)
        return call
    return wrapper


class MultiArg(list):
    def __init__(self,args):
        list.__init__(self,args)

def unordered(maybe_seq):
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

input_transformers = [twinned,unordered]
def apply_to_inputs(**fun_name_to_arg_names):
    name_to_fun = {f.__name__:f for f in functions}
    fun_to_arg_names = {}
    for k,v in fun_name_to_arg_names.iteritems():
        if k in name_to_fun:
            fun_to_arg_names[name_to_fun[k]] = v

experiment_transformer = transform_inputs(unordered,twinned)
def experiment(unpack = False, trials = 1, overwrite = False, memoize = True, **kwargs):
    data_dir = './memo_cache/'
    default_trials = trials

    transform_arg_dict = experiment_transformer(kwargs)
    def wrapper(function):
        def call(*args,**kwargs):
            call_data = fun_call_labeler(function,args,kwargs)

            arg_dict = call_data['args']
            print "Processing: "
            print call_data['defined_call']

            try:
                trials = range(arg_dict['trials'])
            except TypeError:
                trials = arg_dict['trials']
            except KeyError:
                trials = range(default_trials)
            print "trials:%s" % len(trials)

            try:
                del arg_dict['trials']
            except:
                pass

            #if it can handle keywords, give it everything
            args = call._last_args = transform_arg_dict(call_data['valid_args'])

            try:
                arg_hash = dict_hash(args)
            except TypeError as te:
                print args
                raise te

            if memoize and not os.path.exists(data_dir):
                os.makedirs(data_dir)
            cache_file =data_dir+str(arg_hash)+".pkl"
            #print "Loading cache..."
            try:
                assert memoize and not overwrite 
                cache = pd.read_pickle(cache_file)
                cached_trials = list(cache['trial'])
                uncached_trials = [trial for trial in trials if trial not in cached_trials]

                #print "...cache loaded! Entries:%s, New Entries:%s" % (len(cache),len(uncached_trials))

            except Exception as e:
                #print str(e)
                #print "...new cache created."
                if 'trial' in args.keys():
                    cols = args.keys()
                else:
                    cols = args.keys()+['trial']
                cache = pd.DataFrame(columns = cols)
                uncached_trials = trials

            results = []
            for trial in uncached_trials:
                np.random.seed(trial)
                result = function(**copy(args))
                args['trial'] = trial
                if not unpack:
                    results.append(dict(args,**{'result':result}))
                elif unpack == 'dict':
                    results.append(dict(args,**result))
                elif unpack == 'record':
                    for d in result:
                        results.append(dict(args,**d))

            #consolidate new and old results and save
            cache = pd.concat([cache,pd.DataFrame(results)])
            if memoize:
                cache.to_pickle(cache_file)

            #return only those trials that were asked for
            return cache.query('trial in %s' % trials)
        call.__name__ = function.__name__
        call.__getargspec__ = getargspec(function)
        call._decorator = 'experiment'
        return call
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

        def call(*args,**kwargs):
            if verbose and verbose>0:
                print "Expanding multicall for function "+function.__name__
            call_data = fun_call_labeler(function,args,kwargs)

            expected_args = call_data['valid_args']
            expected_args = transform_arg_dict(expected_args)

            call._last_args = expected_args

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

            if verbose and verbose>0:
                print "Processing %s calls..." % len(arg_calls)
            if unpack_ == 'DataFrame':
                ret = pd.concat([function(**arg_call) for arg_call in arg_calls])
            else:
                raise NotImplemented

            if verbose and verbose>0:
                print "...done!\n"

            
            return ret

        call.__name__ = function.__name__
        call.__getargspec__ = function.__getargspec__
        return call
    return wrapper

def or_query(field,ls):
    return " or ".join(["%s == %s" % (field, i) for i in ls])

def plotter(experiment,plot_dir="./plots/"):
    def wrapper(plot_fun):
        try:
            assert 'data' in getargspec(plot_fun)[0]
        except:
            print "wrapped function must have a 'data' argument"
            raise
        def call(*args,**kwargs):
            save_file = plot_fun.__name__+"(%s).pdf"
            plot_fun_call_data = fun_call_labeler(plot_fun,args,kwargs)

            plot_args = plot_fun_call_data['valid_args']

            if plot_fun_call_data['defined_args'].get('data',False):
                ret = plot_fun(**plot_fun_call_data['defined_args'])
            else:
                if 'experiment' in plot_fun_call_data['args']:
                    fun = plot_fun_call_data['args']['experiment']
                else:
                    fun = experiment

                given_args = plot_fun_call_data['args']

                experiment_argnames = fun_call_labeler(fun,[],given_args)['defined_args']

                plot_exclusive_argnames = [k for k in given_args
                                           if k in plot_fun_call_data['defined_args'] and
                                           k not in experiment_argnames]

                experiment_args = dict((key,val) for key,val in given_args.items() if key not in plot_exclusive_argnames)
                experiment_call_data = fun_call_labeler(fun,[],experiment_args)

                call_args = experiment_call_data['valid_args']
                data = fun(**call_args)

                plot_args = plot_fun_call_data['defined_args']

                try:
                    for key,val in fun._last_args.iteritems():
                        if key in plot_args:
                            plot_args[key] = val
                except:
                    pass
                ret = plot_fun(**dict(plot_args,**{'data':data}))
                save_file = plot_dir+save_file % experiment_call_data['call']
                plt.savefig(save_file)
            plt.close()
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
