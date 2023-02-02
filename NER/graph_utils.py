import pandas as pd
from itertools import chain
import numpy as np
import networkx as nx
from cdlib.algorithms import louvain
import difflib
from tqdm import tqdm

def responses_to_flattened_lists(responses, valid_keys):
    responses_chained = []
    for resp in responses:
        responses_chained.append(list(chain.from_iterable([v for k, v in resp.items() if k in valid_keys])))
    return responses_chained

def responses_to_coocuurence_dataframe(responses, valid_keys):
    responces_flattend = responses_to_flattened_lists(responses, valid_keys)
    u = (pd.get_dummies(pd.DataFrame(responces_flattend), prefix='', prefix_sep='')
         .groupby(level=0, axis=1)
         .sum())

    v = u.T.dot(u)
    v.values[(np.r_[:len(v)],) * 2] = 0
    return v

def type_dict_from_responses(responses, valid_keys):
    type_dict = {}
    for resp in responses:
        for key, values in resp.items():
            key = key.strip()
            if key in valid_keys:
                for v in values:
                    if not v in type_dict.keys():
                        type_dict[v] = key
    return type_dict

def build_graph(responses,valid_keys):
    type_dict = type_dict_from_responses(responses, valid_keys)
    G = nx.from_pandas_adjacency(responses_to_coocuurence_dataframe(responses, valid_keys))
    nx.set_node_attributes(G, type_dict, 'node_type')

    communities = louvain(G).communities

    community_idx_dict = {}
    for node in G.nodes:
        for i, c in enumerate(communities):
            if node in c:
                community_idx_dict[node] = str(i)

    nx.set_node_attributes(G, community_idx_dict, 'community')
    return G

def get_subgraph_by_node_names(G, node_names, community=True):
    additional_nodes = []
    if community:
        for nn in node_names:
            node = G.nodes.get(nn)
            if node:
                community = node['community']
                additional_nodes.extend( [x for x,y in G.nodes(data=True) if y['community']==community])
    sg_nodes = list(set(node_names+additional_nodes))
    return G.subgraph(sg_nodes)

def get_subgraph_by_node_type(G, node_type):
    sg_nodes = [x for x, y in G.nodes(data=True) if y['node_type'] == node_type]
    return G.subgraph(sg_nodes)

def remove_duplicates(G):
    nodes = G.nodes
    groups_to_contract = []
    nodes_to_contract = []
    for node in tqdm(nodes):
        if node not in nodes_to_contract:
            similar_nodes = difflib.get_close_matches(node, nodes)
            groups_to_contract.append(similar_nodes)
            nodes_to_contract.extend(similar_nodes)

    contracted_graph = G
    for g in tqdm(groups_to_contract):
        u = g[0]
        for v in g[1:]:
            if u in contracted_graph.nodes and v in contracted_graph.nodes:
                contracted_graph = nx.contracted_nodes(contracted_graph, u, v)

    return contracted_graph