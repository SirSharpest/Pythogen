import numpy as np
from .nx import extract_graph_info, update_node_attribute
from .nx import weights_to_A, get_centre_node
from .narrow_escape import multi_escp


def apply_dead_cells(G, E):
    for cell in G.nodes(data=True):
        if 'deadcell' in cell[1] or cell[1]['pd_radius'] == 0:
            if cell[1]['deadcell'] or cell[1]['pd_radius'] == 0:
                E[:, cell[0]] = 0
                E[cell[0]] = 0


def calc_D_eff(r, D, N, ep, ignore_error=False):
    tau = multi_escp(r, D, N, ep)
    if tau == 0:
        return 0
    x2 = r**2
    Deff = x2 / (2*tau)
    return Deff


def diffuse(G, D, dt, dx, epochs, name):
    E, C = extract_graph_info(G, kind=name)
    dx2 = dx**2
    apply_dead_cells(G, E)
    q_hat = (E * D * dt)
    diag_C = np.diag(C)
    for i in range(epochs):
        E_hat = (diag_C/dx2) * q_hat
        diag_C = diag_C + (np.sum(E_hat, axis=1)-np.sum(E_hat, axis=0))
    diag_C[diag_C > 1] = 1
    diag_C[diag_C < 0] = 0
    update_node_attribute(G, name, diag_C)
