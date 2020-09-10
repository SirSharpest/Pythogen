import numpy as np
import networkx as nx
from networkx.generators.lattice import grid_2d_graph
from networkx.generators.lattice import hexagonal_lattice_graph
from networkx.generators.lattice import triangular_lattice_graph
from .voronoi import generate_voronoi


# DEFAULT_ATTR = 'weight'
# DEFAULT_C = 'C'
# DEFAULT_PATHOGEN = 'P'
# DEFAULT_EFFECTOR = 'E'


def get_ego_graph(G, r=1, C=None, ):
    if C is None:
        C = get_centre_node(G)
    Gn = nx.generators.ego.ego_graph(G, C, radius=r, center=True)
    return Gn


def update_node_attribute(G, attr, new_attrs):
    for n, d in G.nodes(data=True):
        d[attr] = new_attrs[n]


def get_centre_node_voronoi(G):
    # Sloppy implementation for now, could optimise
    # if I didnt have a review meeting tomorrow

    dists = []
    centre = np.array([0.5, 0.5])
    for n, d in G.nodes(data=True):
        b = np.array([d['x'], d['y']])
        dist = np.linalg.norm(centre-b)
        dists.append(dist)

    node = np.where(dists == np.amin(dists))[0][0]
    return node
    # nx.algorithms.distance_measures.center(G)[0]


def get_centre_node(G, voronoi=False):
    if voronoi:
        return get_centre_node_voronoi(G)
    Xs = np.array(list(nx.get_node_attributes(G, 'x').values()))
    Ys = np.array(list(nx.get_node_attributes(G, 'y').values()))
    centre = np.intersect1d(np.where(Xs == (max(Xs)+1)//2),
                            np.where(Ys == (max(Ys)+1)//2))
    return centre[0]


def set_edge_attribute(G, attr, new_attrs):
    nx.set_edge_attributes(G, {(u, v): va for (u, v, a), va in zip(
        G.edges(data=True), new_attrs)}, attr)


def generate_shape(shape, n=1, m=1):
    func_dict = {'rectangle': grid_2d_graph,
                 'hexagon': hexagonal_lattice_graph,
                 'triangle': triangular_lattice_graph,
                 'voronoi': generate_voronoi}

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
