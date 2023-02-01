import pandas as pd
from itertools import chain
import numpy as np

def responses_to_flattened_lists(responses):
    responses_chained = []
    for resp in responses:
        responses_chained.append(list(chain.from_iterable([v for v in resp.values()])))
    return responses_chained

def responses_to_coocuurence_dataframe(responses):
    responces_flattend = responses_to_flattened_lists(responses)
    u = (pd.get_dummies(pd.DataFrame(responces_flattend), prefix='', prefix_sep='')
         .groupby(level=0, axis=1)
         .sum())

    v = u.T.dot(u)
    v.values[(np.r_[:len(v)],) * 2] = 0
    return v



