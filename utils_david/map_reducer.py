from tqdm import tqdm
import multiprocessing


def process_multi(func_, iterable_, num_executor=0, max_tasks_child=1, run_type='map'):
    results = None
    pool = multiprocessing.Pool(processes=num_executor, maxtasksperchild=max_tasks_child)
    if run_type == 'map_async':
        results = [i for i in tqdm(pool.map_async(func_, iterable_).get())]
    elif run_type == 'map':
        results = list(tqdm(pool.map(func_, iterable_), total=len(iterable_)))
    elif run_type == 'imap_unordered':
        results = list(tqdm(pool.imap_unordered(func_, iterable_), total=len(iterable_)))
    elif run_type == 'imap':
        results = list(tqdm(pool.imap(func_, iterable_), total=len(iterable_)))
    # star map are not in use right now
    elif run_type == 'startmap':
        results = list(tqdm(pool.starmap(func_, iterable_), total=len(iterable_)))
    elif run_type == 'starmap_async':
        results = list(tqdm(pool.starmap_async(func_, iterable_), total=len(iterable_)))
    else:
        pass
    return results


class Map:

    def __init__(self, func, data):
        self.func = func
        self.data = data

    def map_reduce(self):
        res = process_multi(self.func, self.data, run_type='map')
