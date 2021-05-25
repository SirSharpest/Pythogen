import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def plot_network(model, attr, size_factor=1000, ax=None):
    pos = {k: (v['x'], v['y']) for k, v in model.G.nodes(data=True)}
    sizes = np.array([v[attr]
                      for k, v in model.G.nodes(data=True)]) * size_factor
    nx.draw(model.G, pos, node_size=sizes, ax=ax)
