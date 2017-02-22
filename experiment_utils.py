import os.path
import pandas as pd
import collections
from itertools import product
from inspect import getargspec
from collections import OrderedDict
import numpy as np
from copy import copy,deepcopy
from utils import pickled, unpickled

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

    defined_args = multiRefOrderedDict((arg,known_args[arg]) for arg in arg_names)
    undefined_args = {key:val for key,val in known_args.iteritems() if key not in defined_args}
 
    return {"known_args":known_args,
            "defined_args":defined_args,
            "undefined_args":undefined_args,
            "arg_names":arg_names,
            "varargs":varargs,
            "keywords":keywords,
            "default_values":default_values,
            "defined_call": method.__name__+"(%s)" % ",".join(["%s=%s" % (key,val) for key,val in defined_args.iteritems()])
    }

def multi_call(static=[],unordered=[],verbose = 2):
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
                    arg_call["result"] = method(trial = trial, expected_args = reference,**arg_call)

                else:
                    try:
                        #print arg_call
                        arg_call["return"] = method(trial = trial,**arg_call)
                    except Exception as e:
                        arg_call["return"] = method(**arg_call)

                arg_call.update({"arg_hash":arg_hash, "trial":trial})

                if verbose>1:
                    print "returned:",arg_call["return"]

                results.append(arg_call)

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

def or_query(field,ls):
    return " or ".join(["%s == %s" % (field, i) for i in ls])

def plotter(default_plot_dir="./plots/",plot_dir_arg_name = 'plot_dir'):
    """
    makes sure plotting directory exists for function
    """
    def wrapper(plot_fun):
        def call(*args,**kwargs):
            call_data = fun_call_labeler(plot_fun, args,kwargs)
            try:
                plot_dir = call_data['defined_args'][plot_dir_arg_name]
            except KeyError:
                plot_dir = default_plot_dir
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            it = plot_fun(*args,**kwargs)
            #plt.save(plot_dir+"foo.pdf")
            return it
        call.__name__ = plot_fun.__name__
        return call
    return wrapper

def experiment(plot_fun,plot_dir="./plots/"):
    def handler(procedure):
        experiment_name = experiment.__name__
        run = multi_call(procedure)
        save_dir = "./"+experiment_name
        def call(plot = True, *args, **kwargs):
            call_data = fun_call_labeler(procedure,args,kwargs)
            conditions = '(%s)' % ", ".join(["%s=%s" % (arg,value) for arg,value in call_data["defined_args"].items()])
            save_file = plot_dir+experiment_name+conditions+".pdf"
            data = run(*args,**kwargs)
            if plot:
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                plot_fun(data,save_file)
            return data
        return call
    return handler

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
    @multi_call()
    def add(a = range(5), b = range(5)):
        return a+b
    add()
    @multi_call(static='factors')
    def prod(factors=[7,5]):
        res = 1
        for factor in factors:
            res *= factor
        return res
    prod()
    prod([5,7])
"end"
