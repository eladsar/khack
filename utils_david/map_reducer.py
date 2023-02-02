import pickle
import time
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from flow import logger
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from connect import OpenAI
import threading

def process_multi(func_, iterable_, num_executor=1, max_tasks_child=1, run_type='map_async'):
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

    def __init__(self, data, q, *args):
        self.question = q
        self.client = q['client']
        self.arg = args
        self.data = data
        self.parse_arg()

    def map_reduce_yes_no_question(self):
        agg_results = []
        if "precedent_question" in self.data.columns:
            data_temp = self.data[self.data['precedent_question']==1]
        else:
            data_temp = self.data
        if data_temp.empty:
            logger.info(f'The Data is empty for question: {self.question["question"]}\nExit process')
            exit(0)
        for idx, row in tqdm(data_temp.iterrows()):
        # start_time = time.perf_counter()
        # tmp_res = process_multi(partial(self.client.yes_or_no, question=self.question['question']), self.data, num_executor=5)
        # end_time = time.perf_counter()
        # logger.info(f'took {end_time - start_time}seconds')
            if len(row['body_splited'])>0:
                split_res = []
                for split in tqdm(row['body_splited']):
                    tmp_res = self.client.yes_or_no(question=self.question['question'], text=split)
                    if tmp_res:
                        split_res.append(True)
                        break
                    split_res.append(tmp_res)
                if any(split_res):
                    agg_results.append(True)
                else:
                    agg_results.append(False)
            # agg_results.append(tmp_res)
        agg_results = [False if i is None else i for i in agg_results]
        if len(data_temp) != len(self.data):
            self.data.loc[self.data['precedent_question']==1, 'precedent_question'] = agg_results
        else:
            self.data['precedent_question'] = agg_results
        path_to_agg_answer = self.question['path'] /  (self.question['id'].hex + '.pkl')
        Map.write_response(path_to_agg_answer, self.data['precedent_question'].tolist())
        return path_to_agg_answer

    def map_reduce_mail_entities_question(self):
        if "precedent_question" in self.data.columns:
            data_temp = self.data[self.data['precedent_question']==1]
        else:
            data_temp = self.data
        if data_temp.empty:
            logger.info(f'The Data is empty for mail entity question\nExit process')
            exit(0)
        agg_results = []
        agg_results_mail_index = []
        for idx, row in data_temp.iterrows():
            if len(row['body_splited'])>0:
                found_entities = []
                interesting_mails = []
                for split  in tqdm(row['body_splited']):
                    format_question_test = f"from: {row['from']}\nto: {row['to']}\nbody: {split}"
                    tmp_res = self.client.mail_entities(text=format_question_test)
                    # logger.critical(f'{tmp_res} are suspects')
                    found_entities.append(tmp_res)
                    if idx not in interesting_mails:
                        interesting_mails.append(idx)
                agg_results.append(f"{found_entities}")
                agg_results_mail_index.append(*interesting_mails)
        final_res = [k for i in agg_results for j in i for k in j]
        # if len(data_temp) != len(self.data):
        #     self.data.loc[self.data['precedent_question']==1 , 'precedent_question'] = agg_results
        # else:
        #     self.data['precedent_question'] = agg_results
        path_to_agg_answer = self.question['path'] / (self.question['id'].hex + '.pkl')
        logger.warning(f'Those are mail entities you search {final_res}, and mail index to check {agg_results_mail_index}')
        Map.write_response(path_to_agg_answer, final_res)
    def parse_arg(self):
        if self.arg[0] is None:
            pass
        else:
            if isinstance(self.arg[0], Path):
                df = Map.read_response(self.arg[0])
                if isinstance(df, list):
                    if len(df) == 0:
                        return
                    self.data['precedent_question'] = df
                elif isinstance(df, str):
                    self.arg = df
            else:
                logger.error('parse_arg case not implemented')
                raise Exception
    @staticmethod
    def write_response(path, data):
        res = None
        try:
            if type(data).__name__ !='string':
                data = pickle.dumps(data)
            with open(path , 'wb') as w:
                w.write(data)
        except Exception as ex:
            logger.error(f'failed to write response, error {ex}')


    @staticmethod
    def read_response(path):
        res = None
        try:
            with open(path, 'rb') as r:
                res = r.read()
        except Exception as ex:
            logger.error(f'failed to write response, error {ex}')
        return pickle.loads(res)

