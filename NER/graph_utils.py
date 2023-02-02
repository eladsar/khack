import pandas as pd
from itertools import chain
import numpy as np
import networkx as nx
from cdlib.algorithms import louvain
import difflib
from tqdm import tqdm
from bokeh.io import output_notebook, show, save
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, Scatter
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.models import ColumnDataSource, CategoricalColorMapper, CategoricalMarkerMapper
from bokeh.palettes import RdBu3

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

def draw_bokeh_graph(G, title):

    # Choose a title!
    communities = list(chain.from_iterable(louvain(G).communities))
    G = G.subgraph(communities)

    #     Establish which categories will appear when hovering over each node
    HOVER_TOOLTIPS = [("Name", "@index"), ("type", "@node_type")]

    # Create a plot — set dimensions, toolbar, and title
    plot = figure(tooltips=HOVER_TOOLTIPS,
                  tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
                  x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title)

    # Create a network graph object with spring layout
    # https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html
    network_graph = from_networkx(G, nx.spring_layout, scale=10, center=(0, 0))
    node_types = [G.nodes[node]['node_type'] for node in G.nodes]
    color_mapper = CategoricalColorMapper(factors=['0', '1', '2', '3', '4', '5', '6'],
                                          palette=['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'brown'])
    color = {'field': 'community', 'transform': color_mapper}
    marker_mapper = CategoricalMarkerMapper(factors=['People', 'Organizations', 'Locations'],
                                            markers=['circle', 'diamond', 'square'])
    marker = {'field': 'node_type', 'transform': marker_mapper}

    # Set node size and color
    network_graph.node_renderer.glyph = Scatter(size=15, marker=marker, fill_color=color)

    # Set edge opacity and width
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)

    # Add network graph to the plot
    plot.renderers.append(network_graph)

    save(plot, filename=f"{title}.html")