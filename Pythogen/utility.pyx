import numpy as np
import pandas as pd


def stokes_einstein(kda):
    return ((1.38e-23 * 298.15) / (6 * np.pi * 8.9e-4 * kda))


def check_negative_values(A):
    if np.any(A < 0):
        raise ValueError(f"Matrix cannot contain negative values! {A}")


def enforce_matrix_shape(I, O):
    if type(I) != np.ndarray or I.shape != O.shape:
        I = (I * O)
    return I


def G_to_pd(G, shape):
    data = dict(G.nodes(data=True))
    for k, v in data.items():
        if 'pos' in data[k]:
            del data[k]['pos']
    df = pd.DataFrame(data).T
    df['shape'] = shape

    return df


def attr_to_arr(G, attr):
    return np.array([v[attr] for k, v in G.nodes(data=True)])
