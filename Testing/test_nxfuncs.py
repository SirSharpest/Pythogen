import pytest
import numpy as np
from CellNetwork.networking_nx import generate_shape, get_ego_graph, update_node_attribute, get_centre_node, set_edge_attribute, set_random_edge_weights, set_concentration, extract_graph_info, get_concentration, weights_to_A, get_weights, get_centre_c, get_concentration, weights_to_A


n = 5
m = 5

def make_G():
    G = generate_shape('rectangle', n=n, m=m)
    return G


def test_get_ego_graph():
    G = make_G()
    Gn = get_ego_graph(G, r=1)

    assert Gn.number_of_nodes() < G.number_of_nodes()


def test_update_node_attribute():
    G = make_G()
    c = get_centre_c(G)
    update_node_attribute(G, 'C', np.ones(n*m)*2)
    cn = get_centre_c(G)
    assert cn > c


def test_get_centre_node():
    G = make_G()
    update_node_attribute(G, 'C', np.ones(n*m)*2)


def test_get_centre_c():
    G = make_G()
    c = get_centre_c(G)
    assert c == 1


def test_generate_shape():
    n = 5
    m = 5
    G = generate_shape('rectangle', n=n, m=m)
    assert G.number_of_nodes() == m*n


def test_extract_graph_info():
    G = make_G()
    A,C = extract_graph_info(G)

    assert A.shape == (n*m, n*m)
    assert np.diag(C).shape == (n*m,)


def test_get_concentration():
    C = get_concentration(make_G())
    assert np.sum(C) == 1

def test_weights_to_A():
    w = weights_to_A(make_G())
    assert np.sum(w) > 0

