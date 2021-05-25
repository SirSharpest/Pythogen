import os

import networkx as nx
import numpy as np
import pandas as pd
from networkx.generators.lattice import (grid_2d_graph,
                                         hexagonal_lattice_graph,
                                         triangular_lattice_graph)

from .voronoi import generate_voronoi


def get_ego_graph(
    G,
    r=1,
    C=None,
):
    if C is None:
        C = get_centre_node(G)
    Gn = nx.generators.ego.ego_graph(G, C, radius=r, center=True)
    return Gn


def update_node_attribute(G, attr, new_attrs):
    for n, d in G.nodes(data=True):
        d[attr] = new_attrs[n]
    #nx.set_node_attributes(G, dict(zip(G.nodes(), new_attrs)), attr)


def get_centre_node_voronoi(G):
    # Sloppy implementation for now, could optimise
    # if I didnt have a review meeting tomorrow

    dists = []
    centre = np.array([0.5, 0.5])
    for n, d in G.nodes(data=True):
        b = np.array([d['x'], d['y']])
        dist = np.linalg.norm(centre - b)
        dists.append(dist)

    node = np.where(dists == np.amin(dists))[0][0]
    return node
    # nx.algorithms.distance_measures.center(G)[0]


def get_centre_node(G, voronoi=False):
    if voronoi:
        return get_centre_node_voronoi(G)
    Xs = np.array(list(nx.get_node_attributes(G, 'x').values()))
    Ys = np.array(list(nx.get_node_attributes(G, 'y').values()))
    centre = np.intersect1d(np.where(Xs == (max(Xs) + 1) // 2),
                            np.where(Ys == (max(Ys) + 1) // 2))
    return centre[0]


def set_edge_attribute(G, attr, new_attrs):
    nx.set_edge_attributes(
        G, {(u, v): va
            for (u, v, a), va in zip(G.edges(data=True), new_attrs)}, attr)


def custom_shape(df, cut_center=0):
    _, ext = os.path.splitext(df)
    if ext != '.json':
        raise NotImplementedError('Only json supported for now')
    df = pd.read_json(df).T.drop(0)

    df['neighbours'] = df.apply(
        lambda r: [x for x in r['neighbours'] if x > 0], axis=1)
    df['area'] = np.sqrt(df['area'].values.astype('float') / np.pi)
    df = df.rename(columns={
        'centroid_x': 'x',
        'centroid_y': 'y',
        'area': 'radius'
    })
    # Gotta make a network

    G = nx.Graph()
    for idx, row in df.iterrows():
        edges = [(idx, n) for n in row['neighbours']]
        if len(edges) == 0:
            continue
        G.add_edges_from(edges)
        for feature in ['x', 'y', 'radius']:
            G.nodes(data=True)[idx][feature] = row[feature]

    if cut_center > 0:
        X = np.max([x for k, x in nx.get_node_attributes(G, 'x').items()])
        Y = np.max([y for k, y in nx.get_node_attributes(G, 'y').items()])
        a = [X, Y]
        to_remove = []
        for idx, n in G.nodes(data=True):
            b = [n['x'], n['y']]
            dist = np.linalg.norm(np.array(a) - np.array(b))
            if dist > cut_center:
                to_remove.append(idx)
        G.remove_nodes_from(to_remove)

    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)

    G = G.subgraph(Gcc[0])

    return nx.convert_node_labels_to_integers(G)


def generate_shape(shape, n=1, m=1):
    func_dict = {
        'rectangle': grid_2d_graph,
        'hexagon': hexagonal_lattice_graph,
        'triangle': triangular_lattice_graph,
        'voronoi': generate_voronoi
    }

    f = func_dict[shape]
    G = f(m, n)
    if shape != 'voronoi':
        set_shape_xy(G)
    G = nx.convert_node_labels_to_integers(G)
    return G


def extract_graph_info(G, kind):
    A = nx.to_numpy_array(G)
    A[A > 0] = 1
    C = np.diag(np.array(get_concentration(G, kind)))
    return A, C


def get_concentration(G, kind):
    return list(nx.get_node_attributes(G, kind).values())


def set_shape_xy(G):
    for i, XY in enumerate(['x', 'y']):
        nx.set_node_attributes(G, {n: n[i]
                                   for (n, d) in G.nodes(data=True)}, XY)


def weights_to_A(G):
    A = nx.to_numpy_array(G)
    W = np.triu(A) + np.tril(A)
    return W


def attr_to_arr(G, attr):
    return np.array([n[attr] for k, n in G.nodes(data=True)])
