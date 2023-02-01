import pandas as pd
from itertools import chain
import numpy as np
import networkx as nx
from cdlib.algorithms import louvain

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

def type_dict_from_responses(responses):
    type_dict = {}
    for resp in responses:
        for key, values in resp.items():
            for v in values:
                if not v in type_dict.keys():
                    type_dict[v] = key
    return type_dict

def build_graph(responses):
    type_dict = type_dict_from_responses(responses)
    G = nx.from_pandas_adjacency(responses_to_coocuurence_dataframe(responses))
    nx.set_node_attributes(G, type_dict, 'node_type')

    communities = louvain(G).communities

    community_idx_dict = {}
    for node in G.nodes:
        for i, c in enumerate(communities):
            if node in c:
                community_idx_dict[node] = str(i)

    nx.set_node_attributes(G, community_idx_dict, 'community')
    return G
