import pickle
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from flow import logger
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from connect import OpenAI


def process_multi(func_, iterable_, num_executor=1, max_tasks_child=1, run_type='map'):
    results = None
    pool = multiprocessing.Pool(processes=1, maxtasksperchild=max_tasks_child)
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

def process_multi2(func_, iterable_, num_executor=1, max_tasks_child=1, run_type='map'):
    with Pool() as pool:
        # issue multiple tasks each with multiple arguments
        async_results = [pool.apply_async(func_, args=(i)) for i in iterable_]
        # retrieve the return value results
        results = [ar.get() for ar in async_results]


class Map:

    def __init__(self, data, q, *args):
        self.question = q
        self.client = q['client']
        self.arg = args
        self.data = data
        self.parse_arg()

    def map_reduce_yes_no_question(self):
        # res = process_multi(partial(self.func, question=self.question['question']), self.data, run_type='map')

        agg_results = []
        for idx, row in tqdm(self.data.iterrows()):
            tmp_res = self.client.yes_or_no(question=self.question['question'], text=row['body'])
            agg_results.append(tmp_res)
        # stay only with yes answer
        res = [i for i in agg_results if i=='yes']
        path_to_agg_answer = self.question['path'] /  self.question['id']
        Map.write_response(path_to_agg_answer, res)
        return path_to_agg_answer

    def map_reduce_mail_entities_question(self):
        # res = process_multi(self.func, self.data, run_type='map')
        agg_results = []
        for idx, row in self.data.iterrows():
            tmp_res = self.client.mail_entities(question=self.question['question'], text=row['body'])
            agg_results.append(tmp_res)
        final_res = ','.join(agg_results)
        path_to_agg_answer = self.question['path'] / self.question['id']
        Map.write_response(path_to_agg_answer, final_res)
    def parse_arg(self):
        if self.arg[0] is None:
            pass
        else:
            if isinstance(self.arg, Path):
                df = Map.read_response(self.arg)
                if isinstance(df, list):
                    self.data = self.data[df]
                if isinstance(df, str):
                    self.arg = df
            else:
                logger.info('parse_arg case not implemented')
                raise Exception
    @staticmethod
    def write_response(path, data):
        res = None
        try:
            if type(data).__name__ !='string':
                data = pickle.dumps(data)
            with open(path, 'wb') as w:
                res = w.write(data)
        except Exception as ex:
            logger.error(f'failed to write response, error {ex}')
        return res

    @staticmethod
    def read_response(path):
        res = None
        try:
            with open(path, 'rb') as r:
                res = r.read()
        except Exception as ex:
            logger.error(f'failed to write response, error {ex}')
        return pickle.loads(res)

