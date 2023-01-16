import os.path
import pandas as pd
from itertools import product
from inspect import getargspec
from collections import OrderedDict
import numpy as np
from copy import copy, deepcopy
from operator import itemgetter
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
import params
from utils import CACHEDIR

def product_of_vals(orderedDict):
    keys, val_lists = list(orderedDict.keys()), list(orderedDict.values())
    return [OrderedDict(list(zip(keys, vals))) for vals in product(*val_lists)]


def fun_call_labeler(method, args, kwargs, intolerant=True):
    """
    given a defined method to be called as 'method(*args,**kwargs)'
    this function returns 3 dicts where the keys are the argument names and the vals those provided
    the first contains all args known at function call, this may even include kwargs not defined by it
    the second contains only those arguments specified in the function definition
    the third contains only those arguments provided but not specified in the definition
    """

    try:
        arg_names, varargs, keywords, default_values = method.__getargspec__
    except:
        arg_names, varargs, keywords, default_values = (
            method.__getargspec__
        ) = getargspec(method)
    given_values = args

    # the dict is first populated with the default values if there are any
    known_args = {}
    try:
        # zip from the bottom up to properly align
        known_args.update(list(zip(*list(map(reversed, [arg_names, default_values])))))
    except:
        # default_values might be None
        pass

    # whether any are provided or not, then they are overwritten here
    known_args.update(kwargs)

    # overwrite only happens for given_values
    known_args.update(list(zip(arg_names, given_values)))

    # if there are fewer than the expected number of arguments
    # call the wrapped function and let it handle the exception
    expected_arg_missing = not all(name in known_args for name in arg_names)
    if expected_arg_missing and intolerant:
        method(**known_args)

    expected_args = [(k, v) for k, v in known_args.items() if k in arg_names]
    defined_args = OrderedDict(
        sorted(expected_args, key=lambda k_v: arg_names.index(k_v[0]))
    )
    # defined_args = OrderedDict((arg,known_args[arg]) for arg in arg_names)
    undefined_args = OrderedDict(
        (k, v)
        for k, v in sorted(iter(known_args.items()), key=itemgetter(0))
        if k not in defined_args
    )
    known_args = OrderedDict(list(defined_args.items()) + list(undefined_args.items()))
    call_data = {
        "args": known_args,
        "defined_args": defined_args,
        "undefined_args": undefined_args,
        "valid_args": known_args if keywords else defined_args,
        "arg_names": arg_names,
        "varargs": varargs,
        "keywords": keywords,
        "default_values": default_values,
        "defined_call": method.__name__
        + "(%s)"
        % ",".join(["%s=%s" % (key, val) for key, val in defined_args.items()]),
        "call": method.__name__
        + "(%s)"
        % ", ".join(["%s=%s" % (key, val) for key, val in known_args.items()]),
        "make_call_str": lambda d: method.__name__
        + "(%s)" % ", ".join(["%s=%s" % (key, val) for key, val in d.items()]),
    }
    return call_data


class MultiArg(list):
    def __init__(self, args):
        list.__init__(self, args)


def unordered(maybe_seq):
    try:
        return tuple(sorted(maybe_seq, key=lambda x: repr(x[0])))
    except:
        try:
            return tuple(sorted(maybe_seq))
        except TypeError:
            return maybe_seq


def dict_key_map(d, fun_to_arg):
    mapped_d = deepcopy(d)
    for arg_name, arg_val in d.items():
        for function, arg_names in fun_to_arg.items():
            if arg_name in arg_names:
                if isinstance(arg_val, MultiArg):
                    mapped_d[arg_name] = MultiArg(list(map(function, arg_val)))
                else:
                    mapped_d[arg_name] = function(arg_val)
    return mapped_d


def copy_function_identity(parent, decorator=None):
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
    name_to_fun = {f.__name__: f for f in functions}

    def transformer(dec_kwargs):
        fun_to_arg_names = {}
        for k, v in dec_kwargs.items():
            if k in name_to_fun:
                fun_to_arg_names[name_to_fun[k]] = v

        def transform_args(arg_dict):
            return dict_key_map(arg_dict, fun_to_arg_names)

        return transform_args

    return transformer

experiment_transformer = transform_inputs(tuple, unordered)

def experiment(overwrite=False, memoize=True, **kwargs):
    if memoize and not os.path.exists(CACHEDIR):
        os.makedirs(CACHEDIR)

    transform_arg_dict = experiment_transformer(kwargs)

    def wrapper(function):
        @copy_function_identity(function)
        def experiment_call(*args, **kwargs):
            memoized = kwargs.get("memoized", memoize)
            if "memoized" in kwargs:
                del kwargs["memoized"]

            trials = list(range(kwargs["trials"]))
            del kwargs["trials"]
            
            call_data = fun_call_labeler(function, args, kwargs)

            args = experiment_call._last_args = transform_arg_dict(
                call_data["valid_args"]
            )

            try:
                # Try to hash the arg dict to get a unique file name for caching. 
                arg_hash = joblib.hash(str(tuple(sorted(args.items()))))

            except TypeError as te:
                print("these are the provided args\n", args)
                raise te

            cache_file = CACHEDIR + str(arg_hash) + ".pkl"
            
            if os.path.exists(cache_file) and memoized and not overwrite:
                # print "Loading cache...",
                cache = pd.read_pickle(cache_file)
                cached_trials = cache.index.levels[cache.index.names.index('trial')].tolist()
                uncached_trials = [
                    trial for trial in trials if trial not in cached_trials
                ]

                # print "cache loaded! cached:%s, need to compute:%s" % (
                #     len(cached_trials),
                #     len(uncached_trials),
                # )
            else:
                # except Exception as e:
                # print "No cache loaded. "
                # cols = args.keys() + ["trial"]
                # cache = pd.DataFrame(columns=cols)
                cache = pd.DataFrame()
                uncached_trials = trials

            results = []
            # for trial in tqdm(uncached_trials, disable=(verbose == 0)):

            # Check the length to avoid the progress bars from getting messed up
            if len(uncached_trials):
                for trial in tqdm(uncached_trials, disable=params.n_jobs > 1 or params.disable_tqdm):
                    np.random.seed(trial)
                    result = function(**copy(args))
                    for d in result:
                        d['trial'] = trial
                        results.append(d)
            
            # consolidate new and old results and save
            if results:
                # Index these values since it acts as a quick form of
                # compression since these fields have massively
                # repeated values.
                results = pd.DataFrame(results).set_index(['trial', 'player_types', 'type'])
                cache = pd.concat([cache, results])
                if memoized:
                    cache.to_pickle(cache_file)

            return cache.query("trial in %s" % trials).reset_index(level=cache.index.names)

        experiment_call._decorator = "experiment"
        return experiment_call

    return wrapper

                                         
def multi_call(unpack=False, **kwargs):

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
            if decorator != "experiment": raise NotImplementedError
        except:
            pass

        @copy_function_identity(function)
        def m_call(*args, **kwargs):
            call_data = fun_call_labeler(function, args, kwargs)

            expected_args = call_data["valid_args"]
            expected_args = transform_arg_dict(expected_args)

            m_call._last_args = expected_args

            dynamic_args = {}
            static_args = {}
            for key, val in expected_args.items():
                if isinstance(val, MultiArg):
                    dynamic_args[key] = val
                else:
                    static_args[key] = val

            if dynamic_args:
                arg_calls = [
                    dict(static_args, **d_args)
                    for d_args in product_of_vals(dynamic_args)
                ]
            else:
                arg_calls = [static_args]
                return function(**static_args)
    
            dfs = Parallel(n_jobs=params.n_jobs)(delayed(function)(**arg_call) for arg_call in tqdm(arg_calls, disable=params.disable_tqdm))
            dfs = pd.concat(dfs)
            return dfs
            

        return m_call

    return wrapper


def plotter(
    experiment,
    default_plot_dir="./plots/",
    default_file_name=None,
    default_extension=".pdf",
    experiment_args=[],
    plot_exclusive_args=["data", "graph_kwargs", "stacked"],
):
    def wrapper(plot_fun):
        try:
            assert "data" in getargspec(plot_fun)[0]
        except:
            print("wrapped function must have a 'data' argument")
            raise
        default_experiment = experiment

        def make_arg_dicts(args, kwargs):
            try:
                plot_fun_call_data = fun_call_labeler(plot_fun, args, kwargs)
            except TypeError as e:
                incomplete_plot_fun_args = fun_call_labeler(
                    plot_fun, args, kwargs, intolerant=False
                )["args"]
                # if 'experiment' in incomplete_plot_fun_args:
                #    experiment = plot_fun_call_data['args']['experiment']
                # else:
                #    experiment = dexperiment
                experiment = incomplete_plot_fun_args.get(
                    "experiment", default_experiment
                )
                fun_args = fun_call_labeler(experiment, [], incomplete_plot_fun_args)[
                    "args"
                ]
                try:
                    if not fun_args["data"]:
                        del fun_args["data"]
                except:
                    pass
                plot_fun_call_data = fun_call_labeler(plot_fun, [], fun_args)
            return plot_fun_call_data

        def call(*args, **kwargs):
            plot_dir = kwargs.pop("plot_dir", default_plot_dir)
            extension = kwargs.pop("extension", default_extension)
            file_name = kwargs.pop("file_name", default_file_name)
            plot_trials = kwargs.pop("plot_trials", False)
            close = kwargs.pop("close", True)

            # if not plot_name:
            #    save_file = plot_fun.__name__+"(%s).pdf"
            # else:
            #    save_file = plot_name+".pdf"
            call_data = make_arg_dicts(args, kwargs)
            plot_args = call_data["valid_args"]
            if False:  # type(call_data['valid_args'].get('data',[])) == pd.DataFrame:
                ret = plot_fun(**call_data["valid_args"])
            else:
                fun = call_data["args"].get("experiment", experiment)

                if plot_exclusive_args:
                    try:
                        experiment_args = {
                            k: v
                            for k, v in call_data["args"].items()
                            if k not in plot_exclusive_args
                        }
                    except TypeError:
                        given_args = call_data["args"]
                        experiment_argnames = get_arg_dicts(fun, [], given_args)[
                            "defined_args"
                        ]
                        plot_exclusive_argnames = [
                            k
                            for k in given_args
                            if k in call_data["defined_args"]
                            and k not in list(experiment_argnames.keys()) + experiment_args
                        ]
                        experiment_args = dict(
                            (key, val)
                            for key, val in list(given_args.items())
                            if key not in plot_exclusive_argnames
                        )
                    experiment_call_data = fun_call_labeler(fun, [], experiment_args)
                    call_args = experiment_call_data["valid_args"]
                else:
                    call_args = get_arg_dicts(fun, [], call_data["args]"])["valid_args"]
                if "experiment" in call_args:
                    call_args["experiment"] = call_args["experiment"].__name__
                data = fun(**call_args)
                plot_args = call_data["valid_args"]

                try:
                    for key, val in fun._last_args.items():
                        if key in plot_args:
                            plot_args[key] = val
                except:
                    pass

            if type(call_data["valid_args"].get("data", [])) == pd.DataFrame:
                data = call_data["valid_args"]["data"]

            ret = plot_fun(**dict(plot_args, **{"data": data}))

            if not file_name:
                file_name = call_data["make_call_str"](call_args)

            if plot_trials:
                trial_plot_dir = plot_dir + file_name + "/"
                for t, d in data.groupby("trial"):
                    call(
                        *args,
                        **dict(
                            kwargs,
                            **dict(
                                data=d,
                                plot_dir=trial_plot_dir,
                                file_name=str(int(t)),
                                extension=extension,
                                plot_trials=False,
                            )
                        )
                    )

            save_file = plot_dir + file_name + extension
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            try:
                plt.savefig(save_file)
            except IOError as e:
                print(e)
                print(file_name)
                new_file_name = input("Automatic Filename Too Long. Enter new one:")
                save_file = plot_dir + new_file_name + extension
                plt.savefig(save_file)

            if close:
                plt.close()

            return ret

        call.make_arg_dicts = make_arg_dicts
        return call

    return wrapper


def get_arg_dicts(fun, args, kwargs):
    try:
        return fun.make_arg_dicts(args, kwargs)
    except Exception as e:
        print(e)
        return fun_call_labeler(fun, args, kwargs)


def dict_key_map(d, fun_to_arg):
    mapped_d = deepcopy(d)
    for arg_name, arg_val in d.items():
        for function, arg_names in fun_to_arg.items():
            if arg_name in arg_names:
                if isinstance(arg_val, MultiArg):
                    mapped_d[arg_name] = MultiArg(list(map(function, arg_val)))
                else:
                    mapped_d[arg_name] = function(arg_val)
    return mapped_d

if __name__ == "__main__":
    a = MultiArg(i for i in range(5))
    print(isinstance(a, MultiArg))
    # @multi_call()
    # def add(a = range(5), b = range(5)):
    #    return a+b
    # add()
    # @multi_call(static='factors')
    # def prod(factors=[7,5]):
    #    res = 1
    #    for factor in factors:
    #        res *= factor
    #    return res
    # prod()
    # prod([5,7])
"end"
