from collections import defaultdict
from contextlib import ContextDecorator, contextmanager, nullcontext
import heapq
import logging
from math import ceil
import multiprocessing as mp
import multiprocessing.pool
import os
from pathlib import Path
from time import time

import dill


###########
# LOGGING #
###########

def ntabs(n):
    return '\t' * n

class TabbedLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra):
        self.tabdepths = defaultdict(int)  # maps PIDs to tab depths
        self.default_level = logging.INFO  # default logging level
        super().__init__(logger, extra)
    def default_log(self, msg):
        """Logs msg at the default logging level."""
        self.log(self.default_level, msg)
    def reset(self):
        """Resets the tab depth to 0 on the current process."""
        self.tabdepths[os.getpid()] = 0
    def increment(self, n = 1):
        """Increments the level of tabbing on the current process."""
        self.tabdepths[os.getpid()] += n
    def decrement(self, n = 1):
        """Decrements the level of tabbing on the current process."""
        self.tabdepths[os.getpid()] -= n
    def process(self, msg, kwargs):
        return (ntabs(self.tabdepths[os.getpid()]) + msg, kwargs)

LOGGER = TabbedLoggerAdapter(logging.getLogger(__name__), {})
logging.basicConfig(level = logging.INFO, format = '%(message)s')

@contextmanager
def tabbed(n = 1):
    """Increments the global logger's tabdepth counter at the beginning of the context, then decrements it at the end."""
    LOGGER.increment(n)
    yield
    LOGGER.decrement(n)

@contextmanager
def loglevel(logger, level):
    """Sets the level of a logger during the context."""
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    yield
    logger.setLevel(old_level)


###############
# PARALLELISM #
###############

class NonDaemonicProcess(mp.Process):
    """Subclass of Process that makes it non-daemonic."""
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)  # type: ignore

class NonDaemonicPool(multiprocessing.pool.Pool):
    """Subclass of Pool that uses non-daemonic processes.
    We subclass multiprocessing.pool.Pool instead of multiprocessing.Pool because the latter is only a wrapper function, not a proper class."""
    Process = NonDaemonicProcess

def _apply_packed_function_for_map(tup):
    (dumped_fn, item, args, kwargs) = tup
    target_fn = dill.loads(dumped_fn)
    return target_fn(item, *args, **kwargs)

def _pack_function_for_map(target_fn, items, *args, **kwargs):
    dumped_fn = dill.dumps(target_fn)
    dumped_items = [(dumped_fn, item, args, kwargs) for item in items]
    return (_apply_packed_function_for_map, dumped_items)

def parallel_map(func, vals, *args, **kwargs):
    """Maps a function to an iterable of values in parallel.
    args and kwargs go to the function, except:
    _nthreads kwarg, which determines the number of threads to use, and
    _chunksize kwarg, which determines the number of jobs in a chunk."""
    nthreads = kwargs.pop('_nthreads', mp.cpu_count())
    chunksize = kwargs.pop('_chunksize', 1)
    nondaemonic = kwargs.pop('_nondaemonic', False)
    asynchronous = kwargs.pop('_async', False)
    pool = NonDaemonicPool(processes = nthreads) if nondaemonic else mp.Pool(processes = nthreads)
    mapfunc = pool.imap_unordered if asynchronous else pool.map
    gen = mapfunc(*_pack_function_for_map(func, vals, *args, **kwargs), chunksize = chunksize)
    def generate():
        yield from gen
        pool.close()
        pool.terminate()
    return generate()

def parallel_starmap(func, tuples, *args, **kwargs):
    """Starmaps a function to an iterable of tuples in parallel.
    args and kwargs go to the function, except:
    _nthreads: number of threads to use
    _chunksize: number of jobs in a chunk"""
    def func2(tup, *args, **kwargs):
        return func(*(tup + args), **kwargs)
    return parallel_map(func2, tuples, *args, **kwargs)

def parallel_array_map(func, arr, *args, **kwargs):
    """Given a function that applies to a numpy array, blocks the array into chunks, applies the function in parallel, then returns a list of the outputs of each chunk.
    _nthreads: determines the number of threads to use
    _nchunks: determines the number of chunks to divide the array into"""
    nthreads = kwargs.pop('_nthreads', mp.cpu_count())
    nchunks = kwargs.pop('_nchunks', nthreads)
    m = arr.shape[0]
    chunksize = int(ceil(m / nchunks))
    chunks = [arr[i * chunksize : (i + 1) * chunksize] for i in range(nchunks)]
    def newfunc(mat, *newargs, **newkwargs):
        return func(mat, *newargs, **newkwargs)
    return parallel_map(newfunc, chunks, *args, **kwargs)

##############
# DECORATORS #
##############

class Memodict(dict):
    """Memoization dictionary for a function taking one or more positional arguments."""
    def __init__(self, g):
        self.g = g
    def __call__(self, *args):
        return self[args]
    def __missing__(self, key):
        ret = self[key] = self.g(*key)
        return ret

def pack_args(*args, **kwargs):
    """Converts args and kwargs to a single tuple."""
    return (args, tuple(sorted(kwargs.items())))

def unpack_args(key):
    """Converts a tuple to args and kwargs."""
    return (key[0], dict(key[1]))

class KwargMemodict(Memodict):
    """Memoization dictionary for a function taking positional and/or keyword arguments."""
    def __call__(self, *args, **kwargs):
        return self[pack_args(*args, **kwargs)]
    def __missing__(self, key):
        (args, kwargs) = unpack_args(key)
        ret = self[key] = self.g(*args, **kwargs)
        return ret

def memoize(f):
    """Memoization decorator for functions taking one or more positional arguments. Also works on classes whose initializer has only positional arguments."""
    return Memodict(f)

def memoize2(f):
    """Memoization decorator for functions taking arbitrary hashable args and kwargs. Also works on methods.
    NOTE: there is some additional packing/unpacking overhead for this version, so use `memoize` when possible."""
    import inspect
    args = inspect.getfullargspec(f).args
    ismethod = (len(args) > 0) and (args[0] in ['self', 'cls'])
    if ismethod:
        memo = KwargMemodict(f)
        def f2(self, *args, **kwargs):
            return memo(self, *args, **kwargs)
        f2._memo = memo
        return f2
    else:
        return KwargMemodict(f)

def memproperty(f):
    return property(memoize2(f))

def user_confirm(message):
    """Decorator for user actions requesting yes/no confirmation after the supplied message."""
    def _action_with_confirmation(action):
        def _action(*args, **kwargs):
            prompt = message + ' [y/n]'
            while True:
                LOGGER.warning(prompt)
                choice = input().lower()
                if (len(choice) > 0):
                    if (choice[0] == 'y'):
                        return action(*args, **kwargs)
                    elif (choice[0] == 'n'):
                        return None
        return _action
    return _action_with_confirmation

class Timed(ContextDecorator):
    """Timing context, printing the runtime afterward."""
    def __init__(self, printer = LOGGER.default_log):
        self.printer = printer
    def __enter__(self):
        self.start = time()
        return self
    def __exit__(self, tp, value, traceback):
        self.end = time()
        total = self.end - self.start
        self.printer(f'{total:.3g} sec')

timed = Timed()

class Action(ContextDecorator):
    """Context that prints a message at the beginning and/or end, and (optionally) the total runtime."""
    def __init__(self, start_msg = None, end_msg = None, timed = True, printer = LOGGER.default_log):
        """start_msg is a message to print at the start
           end_msg is a message to print at the end
           timed indicates whether to time the context
           printer is a function that takes a message and prints it"""
        self.start_msg = start_msg
        self.end_msg = end_msg
        self.printer = printer
        self.time_context = Timed(printer = self.printer) if timed else nullcontext()
    def __enter__(self):
        if (self.start_msg is not None):
            self.printer(self.start_msg)
        self.time_context.__enter__()
        return self
    def __exit__(self, tp, value, traceback):
        self.time_context.__exit__(tp, value, traceback)
        if (self.end_msg is not None):
            self.printer(self.end_msg)


###################
# FILE MANAGEMENT #
###################

OPEN_KWARGS = {'buffering', 'encoding', 'errors', 'newline', 'closefd', 'opener'}

def count_lines(filename, encoding = 'utf-8'):
    """Counts the number of lines in a file."""
    with open(filename, 'r', encoding = encoding) as f:
        for (i, _) in enumerate(f):  # noqa
            pass
        return i + 1

def save_data(data, filename, file_type = None, mkdir = False, **kwargs):
    """Saves some data to a file. The file type may vary, and if it is unspecified, uses the filename extension to infer the type. If mkdir = True, creates the directory path to the file if it does not already exist; otherwise raises an error if the directory does not exist. User may pass additional kwargs to delegate to the saving function."""
    p = Path(filename)
    if mkdir and (not p.exists()):  # create the directory if it does not yet exist
        with tabbed():
            LOGGER.warning(f'Creating directory {p.parent}')
        p.parent.mkdir(parents = True, exist_ok = True)
    file_type = p.suffix[1:] if (file_type is None) else file_type
    file_type = file_type.lower()
    open_kwargs = {key : val for (key, val) in kwargs.items() if (key in OPEN_KWARGS)}
    other_kwargs = {key : val for (key, val) in kwargs.items() if (key not in OPEN_KWARGS)}
    with Action(f'Saving file to {filename}'):
        if (file_type == 'csv'):
            data.to_csv(filename, **kwargs)
        elif (file_type == 'dat'):  # tab-separated table file
            data.to_csv(filename, sep = '\t', **kwargs)
        elif (file_type == 'txt'):
            with open(filename, 'w', **open_kwargs) as f:
                f.write(data)
        elif (file_type == 'json'):
            import json
            with open(filename, 'w', **open_kwargs) as f:
                json.dump(data, f, **other_kwargs)
        else:  # default to pickle
            import pickle
            with open(filename, 'wb', **open_kwargs) as f:
                pickle.dump(data, f, **other_kwargs)

def save_table(data, filename, mkdir = False, **kwargs):
    save_data(data, filename, file_type = 'dat', mkdir = mkdir, **kwargs)

def load_data(filename, file_type = None, **kwargs):
    """Loads some data from a file. The file type may vary, and if it is unspecified, uses the filename extension to infer the type. User may pass additional kwargs to delegate to the loading function."""
    file_type = Path(filename).suffix[1:] if (file_type is None) else file_type
    open_kwargs = {key : val for (key, val) in kwargs.items() if (key in OPEN_KWARGS)}
    other_kwargs = {key : val for (key, val) in kwargs.items() if (key not in OPEN_KWARGS)}
    with Action(f'Loading {file_type} file from {filename}'):
        if (file_type == 'csv'):
            import pandas as pd
            data = pd.read_csv(filename, **kwargs)
        elif (file_type == 'dat'):  # tab-separated table file
            import pandas as pd
            data = pd.read_csv(filename, sep = '\t', **kwargs)
        elif (file_type == 'txt'):
            with open(filename, 'r', **open_kwargs) as f:
                data = f.read()
        elif (file_type == 'json'):
            import json
            with open(filename, 'r', **open_kwargs) as f:
                data = json.load(f, **other_kwargs)
        else:  # default to pickle
            import pickle
            with open(filename, 'rb', **open_kwargs) as f:
                data = pickle.load(f, **other_kwargs)
    return data

def load_table(filename, **kwargs):
    return load_data(filename, file_type = 'dat', **kwargs)


########
# MATH #
########

def ceildiv(a, b):
    """Ceiling division."""
    return -(-a // b)


###################
# DATA STRUCTURES #
###################

class TopNHeap(list):
    """Maintains the largest N items on a heap."""
    def __init__(self, N = None):
        super().__init__()
        self.N = N
    def empty(self):
        return (len(self) == 0)
    def top(self):
        if self.empty():
            raise ValueError("Heap is empty.")
        return heapq.nsmallest(1, self)
    def push(self, elt):
        if ((self.N is None) or (len(self) < self.N)):
            heapq.heappush(self, elt)
            return None
        else:
            return heapq.heappushpop(self, elt)
    def pop(self):
        return heapq.heappop(self)


############
# PLOTTING #
############

light_gray = (0.95, 0.95, 0.95)
