import os
from indirect_reciprocity import ReciprocalAgent,SelfishAgent
import pandas as pd
import collections
from itertools import product
from inspect import getargspec
from collections import OrderedDict

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

#def dict_query(df,vals_dict):
#    item_queries = [];append = item_queries.append
#    for key,vals in vals_dict.items():
#        if is_sequency(vals):
#            append("(%s)" % " | ".join(["(%s == %s)" % (key,val) for val in vals]))
#        else:
#            append("(%s == %s)" % (key,vals))
#    query_string =  " & ".join(item_queries)
#    return df.query(query_string)




replace_dict = {basestring:(lambda str: "'%s'" % str)}

def dict_query(df,vals_dict):
    and_clauses = [];append = and_clauses.append
    for key,vals in vals_dict.items():
        if is_sequency(vals):
            or_clauses = []
            for val in vals:
                try:
                    or_clauses.append("(df['%s'] == %s)" % (key,val.__name__))
                except AttributeError:
                    for type_ in replace_dict:
                        if isinstance(val,type_):
                            or_clauses.append("(df['%s'] == %s)" % (key,replace_dict[type_](val)))
                            break
                    else:
                        or_clauses.append("(df['%s'] == %s)" % (key,val))                    
            append("(%s)" % " | ".join(or_clauses))
        else:
            try:
                append("(df['%s'] == %s)" % (key,vals.__name__))
            except AttributeError:
                for type_ in replace_dict:
                    if isinstance(vals,type_):
                        append("(df['%s'] == %s)" % (key,replace_dict[type_](vals)))
                        break
                else:
                    append("(df['%s'] == %s)" % (key,vals))
                
    query_string =  "df[%s]" % " & ".join(and_clauses)
    try:
        return eval(query_string)
    except:
        print "there was an error", query_string
        return eval(query_string)
    
def df_contains_dict(df,vals_dict):
    query = dict_query(df,vals_dict)
    return not query.empty

def log_init(method):
    """
    records the last call of a method as an OrderedDict
    in an automatically created dictionary at object.method_calls
    whose keys are the names of the methods
    and whose values are the lists of succesive dicts.

    the keys of the OrderedDict are the names of the arguments,
    the values are the values that ended up being passed in
    whether or not an argument was called with default values or with keywords is not recorded
    likewise any change in the order in which arguments are provided, in the case of keywords
    is not preserved.

    saves named arguments to object.init_args
    for every arg in arglist stores given value in object.arg
    calls super init with extra kwargs

    """
    def wrapper(*args,**kwargs):
        data_dir = "memo_cache/"
        data_file = data_dir + method.__name__+'.pkl'

        self = args[0]
       
        arg_names,varargs,keywords,default_values = getargspec(method)

        #exclude 'self' from the tracked arguments
        arg_names = arg_names[1:]
        given_values = args[1:]

        #the dict is first populated with the default values if there are any
        argname_2_value = {}
        try:
            #zip from the bottom up to properly align
            argname_2_value.update(apply(zip,map(reversed,[arg_names,default_values])))
        except:
            #default_values might be None
            pass
        
        #whether any are provided or not, then they are overwritten here
        argname_2_value.update(kwargs)

        #overwrite only happens for 
        argname_2_value.update(zip(arg_names,given_values))

        #if there are fewer than the expected number of arguments
        #call the wrapped function and let it handle the exception
        if len(argname_2_value) < len(arg_names):
            method(*args,**kwargs)

        ### get a dict of the args explicitly named in the init definition
        self.expected_args = expected_args = named_args = multiRefOrderedDict((arg,argname_2_value[arg]) for arg in arg_names)

        #self.unexpected_args = {item for item in argname_2_value.items() if item not in arg_names}
        self.__dict__.update(named_args)


        #identify items that are sequences
        dynamic_args = {key:val for key,val in expected_args.items() if is_sequency(val)}
        static_args = {key:val for key,val in expected_args.items() if key not in dynamic_args}

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
        try:
            print data_file
            cache = pd.read_pickle(data_file)
        except:
            print "First time running"
            cache = pd.DataFrame(columns = all_arg_calls[0].keys())
            
        uncomputed_arg_calls = [arg_call for arg_call in all_arg_calls
                                if dict_query(cache,arg_call).empty]
        to_cache = []
        for arg_call in uncomputed_arg_calls:
            arg_call["result"] = method(**arg_call)
            to_cache.append(arg_call)

        to_cache = pd.DataFrame(to_cache)
        new_cache = pd.concatenate(cache,to_cache)

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        new_cache.to_pickle(data_file)

        return dict_query(new_cache,expected_args)
        
        
        ### get a dict of the args explicitly named in the super init or not named
        ### in this definition. pass that dict to super_init
        super_init = super(self.__class__,self).__init__
        super_init_arg_names = getargspec(super_init)[0]
        super_kwargs = {arg_name:val for arg_name,val in argname_2_value.items()
                        if arg_name in super_init_arg_names or
                        arg_name not in named_args}
        self.super_init_args = super_kwargs
        super(self.__class__,self).__init__(**super_kwargs)
        
        try:
            self.method_calls[method.__name__].append(argname_2_value)
        except:
            self.method_calls = {}
            self.method_calls[method.__name__] = [argname_2_value]
        return method(*args,**kwargs)
    return wrapper



def multi_method_call(method):
    """
    records the last call of a method as an OrderedDict
    in an automatically created dictionary at object.method_calls
    whose keys are the names of the methods
    and whose values are the lists of succesive dicts.

    the keys of the OrderedDict are the names of the arguments,
    the values are the values that ended up being passed in
    whether or not an argument was called with default values or with keywords is not recorded
    likewise any change in the order in which arguments are provided, in the case of keywords
    is not preserved.

    saves named arguments to object.init_args
    for every arg in arglist stores given value in object.arg
    calls super init with extra kwargs

    """
    def wrapper(*args,**kwargs):
        self = args[0]
        data_dir = "memo_cache/"
        data_file = data_dir + self.__class__.__name__ +"."+ method.__name__+'.pkl'
        
        method_name = method.__name__
        arg_names,varargs,keywords,default_values = getargspec(method)

        #exclude 'self' from the tracked arguments
        arg_names = arg_names[1:]
        given_values = args[1:]

        #the dict is first populated with the default values if there are any
        argname_2_value = {}
        try:
            #zip from the bottom up to properly align
            argname_2_value.update(apply(zip,map(reversed,[arg_names,default_values])))
        except:
            #default_values might be None
            pass
        
        #whether any are provided or not, then they are overwritten here
        argname_2_value.update(kwargs)

        #overwrite only happens for 
        argname_2_value.update(zip(arg_names,given_values))

        #if there are fewer than the expected number of arguments
        #call the wrapped function and let it handle the exception
        if len(argname_2_value) < len(arg_names):
            method(*args,**kwargs)

        ### get a dict of the args explicitly named in the init definition
        self.expected_args = expected_args = named_args = multiRefOrderedDict((arg,argname_2_value[arg]) for arg in arg_names)

        #self.unexpected_args = {item for item in argname_2_value.items() if item not in arg_names}
        self.__dict__.update(named_args)


        #identify items that are sequences
        dynamic_args = {key:val for key,val in expected_args.items() if is_sequency(val)}
        static_args = {key:val for key,val in expected_args.items() if key not in dynamic_args}

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


        try:
            cache = pd.read_pickle(data_file)
        except:
            print "First time running"
            cache = pd.DataFrame(columns = all_arg_calls[0].keys())

        #print "Given the following arg_calls:",all_arg_calls

        uncomputed_arg_calls = [arg_call for arg_call in all_arg_calls
                                if not df_contains_dict(cache,arg_call)]
        
        to_cache = []
        number_of_calls = len(uncomputed_arg_calls)
        print "There are", number_of_calls, "uncomputed method calls"
        for n,arg_call in enumerate(uncomputed_arg_calls):
            print n+1,"/",number_of_calls
            print arg_call
            arg_call["result"] = method(**updated(arg_call,{"self":self}))
            to_cache.append(arg_call)

        to_cache = pd.DataFrame(to_cache)
        new_cache = pd.concat([cache,to_cache])

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        new_cache.to_pickle(data_file)

        return dict_query(new_cache,expected_args)
    return wrapper

def multi_call(method):
    """

    This one is for normal functions

    records the last call of a method as an OrderedDict
    in an automatically created dictionary at object.method_calls
    whose keys are the names of the methods
    and whose values are the lists of succesive dicts.

    the keys of the OrderedDict are the names of the arguments,
    the values are the values that ended up being passed in
    whether or not an argument was called with default values or with keywords is not recorded
    likewise any change in the order in which arguments are provided, in the case of keywords
    is not preserved.

    saves named arguments to object.init_args
    for every arg in arglist stores given value in object.arg
    calls super init with extra kwargs

    """
    def wrapper(*args,**kwargs):
        data_dir = "memo_cache/"
        data_file = data_dir + method.__name__+'.pkl'
        
        arg_names,varargs,keywords,default_values = getargspec(method)

        #exclude 'self' from the tracked arguments
        arg_names = arg_names
        given_values = args

        #the dict is first populated with the default values if there are any
        argname_2_value = {}
        try:
            #zip from the bottom up to properly align
            argname_2_value.update(apply(zip,map(reversed,[arg_names,default_values])))
        except:
            #default_values might be None
            pass
        
        #whether any are provided or not, then they are overwritten here
        argname_2_value.update(kwargs)

        #overwrite only happens for 
        argname_2_value.update(zip(arg_names,given_values))

        #if there are fewer than the expected number of arguments
        #call the wrapped function and let it handle the exception
        if len(argname_2_value) < len(arg_names):
            method(*args,**kwargs)

        if "trial" in argname_2_value:
            argname_2_value["trial"] = range(argname_2_value["trial"])
        ### get a dict of the args explicitly named in the init definition
        expected_args = named_args = multiRefOrderedDict((arg,argname_2_value[arg]) for arg in arg_names)

        #self.unexpected_args = {item for item in argname_2_value.items() if item not in arg_names}
        #self.__dict__.update(named_args)


        #identify items that are sequences
        dynamic_args = {key:val for key,val  in expected_args.items() if is_sequency(val)}
        static_args = {key:val for key,val in expected_args.items() if key not in dynamic_args}

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


        try:
            print data_file
            cache = pd.read_pickle(data_file)
        except:
            print "First time running"
            cache = pd.DataFrame(columns = all_arg_calls[0].keys())

        #print "Given the following arg_calls:",all_arg_calls
        uncomputed_arg_calls = []; append_call =  uncomputed_arg_calls.append
        for arg_call in all_arg_calls:
            print "query"
            a = dict_query(cache,arg_call)
            if not dict_query(cache,arg_call).empty:
                print "it's in there"
                print a
            else:
                print "it's not there"
                print a
            if dict_query(cache,arg_call).empty:
                append_call(arg_call)
                
        results = []
        number_of_calls = len(uncomputed_arg_calls)
        print "There are", number_of_calls, "uncomputed method calls out of",len(all_arg_calls)
        for n,arg_call in enumerate(uncomputed_arg_calls):
            print n+1,"/",number_of_calls
            print "current arg call:",arg_call
            
            if keywords:
                reference = copy(arg_call)
                arg_call["result"] = method(**updated(arg_call,{"expected args":reference}))
            else:
                arg_call["result"] = method(**arg_call)
            print arg_call["result"]
            results.append(arg_call)
            
        to_cache = pd.DataFrame(results)
        for result in results:
            try:
                assert not dict_query(to_cache,result).empty
            except:
                print result
                print to_cache
        new_cache = pd.concat([cache,to_cache])

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        new_cache.to_pickle(data_file)

        #print new_cache
        #print new_cache[new_cache['agent_type'] == ReciprocalAgent]
        return dict_query(new_cache,expected_args)
    return wrapper

class dummy(object):
    @multi_method_call
    def meth(self,a,b):
        return a+b

if __name__ == "__main__":
    print "hi"
    a = dummy()

    a.meth([1,2],3)

"end"
