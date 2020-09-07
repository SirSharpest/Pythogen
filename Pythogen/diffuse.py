import numpy as np
from tqdm import tqdm
from .nx import extract_graph_info, update_node_attribute, weights_to_A, DEFAULT_C, get_centre_node
from .utility import enforce_matrix_shape, check_negative_values
from .narrow_escape import multi_escp


def apply_dead_cells(G, E):
    for cell in G.nodes(data=True):
        if 'deadcell' in cell[1]:
            if cell[1]['deadcell']:
                E[:, cell[0]] = 0
                E[cell[0]] = 0


def calc_D_eff(r, D, N, ep, ignore_error=False):
    tau = multi_escp(r, D, N, ep)
    x2 = r**2
    Deff = x2 / (2*tau)
    return Deff


def diffuse(G, D, dt, dx, epochs, deadcells=False, progress=True, bombardment=False, voronoi=False):
    E, C = extract_graph_info(G)
    dx2 = dx**2
    if deadcells:
        apply_dead_cells(G, E)
    q_hat = (E * D * dt)
    diag_C = np.diag(C)
    if progress:
        for i in tqdm(range(epochs)):
            E_hat = (diag_C/dx2) * q_hat
            diag_C = diag_C + (np.sum(E_hat, axis=1)-np.sum(E_hat, axis=0))
    else:
        for i in range(epochs):
            E_hat = (diag_C/dx2) * q_hat
            diag_C = diag_C + (np.sum(E_hat, axis=1)-np.sum(E_hat, axis=0))
            if bombardment:
                diag_C[get_centre_node(G, voronoi)] = 1
    update_node_attribute(G, DEFAULT_C, diag_C)


def diffuse_func(diag_C, dx2, q_hat):
    E_hat = (diag_C/dx2) * q_hat
    diag_C = diag_C + (np.sum(E_hat, axis=1)-np.sum(E_hat, axis=0))
    return diag_C


def diffuse_numba(G, D, dt, dx, epochs, deadcells=False, progress=True):
    E, C = extract_graph_info(G)
    dx2 = dx**2
    if deadcells:
        apply_dead_cells(G, E)
    q_hat = (E * D * dt)
    diag_C = np.diag(C)

    # compile func first
    for i in range(epochs):
        diag_C = diffuse_func(diag_C, dx2, q_hat)

    update_node_attribute(G, DEFAULT_C, diag_C)
