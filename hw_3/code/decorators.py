

## DEBUG DECORATOR

def debug_transformer(fun):
    # Allow wrapper to receive arbitrary args
    def wrapper(*args, **kwargs):
        print(f'Function `{fun.__name__}` called')
        # And pass it to the original function
        res = fun(*args, **kwargs)
        print(f'Function `{fun.__name__}` finished')
        return res
        
    return wrapper


## TIMING DECORATOR

import time
import functools

def time_it(fun):
    @functools.wraps(fun)  # preserve name, id
    def wrapper(*args, **kwargs):
        start = time.time()
        res = fun(*args, **kwargs)
        end = time.time()
        print(f'Function {fun.__name__} took {end-start}s')
        
        return res
    
    return wrapper

# more serious debug function - print the args

def debug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           # 4
        return value
    return wrapper_debug