import numpy as np
from .nx import extract_graph_info, update_node_attribute
from .narrow_escape import multi_escp


def calc_D_eff(r, D, N, ep):
    tau = multi_escp(r, D, N, ep)
    x2 = r**2
    Deff = x2 / (6*tau)
    return Deff

# TODO: Apply a realitive factor here for number of neighbours... Perhaps when calculating D_eff?


def diffuse(G, D, dt, dx, epochs, name, modifier_fs, signal):
    E, C = extract_graph_info(G, kind=name)
    for f in modifier_fs:
        f(G, E, signal)
    dx2 = dx**2
    q_hat = (E * D * dt)
    diag_C = np.diag(C)
    for i in range(epochs):
        E_hat = (diag_C/dx2) * q_hat
        diag_C = diag_C + (np.sum(E_hat, axis=1)-np.sum(E_hat, axis=0))
    diag_C[diag_C > 1] = 1
    diag_C[diag_C < 0] = 0
    update_node_attribute(G, name, diag_C)
