import numpy as np
from .nx import extract_graph_info, update_node_attribute
from .narrow_escape import multi_escp


def calc_cell_D_eff_NEP(cell, signal):
    return calc_D_eff(cell['radius'], signal.D,
                            cell['num_pd']/cell['num_neighbours'],
                            np.pi*(cell['radius_ep'])**2)

def calc_D_eff(r, D, N, ep):
    tau = multi_escp(r, D, N, ep)
    x2 = r**2
    Deff = x2 / (6*tau)
    return Deff

def calc_cell_D_eff_permiability(cell, signal):
    return calc_D_eff_permiability(signal.D, cell['q'],
                                   cell['radius']*2 )

def calc_D_eff_permiability(D,q,l):
    return (D*q*l)/(D+q*l)

def diffuse(G, D, dt, dx, epochs, name, modifier_fs, signal):
    E, C = extract_graph_info(G, kind=name)
    for f in modifier_fs:
        f(G, E, signal)
    dx2 = dx**2

    q_hat = np.zeros( (2, len(E), len(E)) )
    q_hat[0] = (D* dt *E) # Potential per connection! 
    q_hat[1] = (E.T * dt * D).T
    q_hat = np.min(q_hat, axis=0)
    diag_C = np.diag(C)

    for _ in range(epochs):
        E_hat = (diag_C/dx2) * q_hat
        diag_C = diag_C + (np.sum(E_hat, axis=1)-np.sum(E_hat, axis=0))
    #diag_C[diag_C > 1] = 1
    diag_C[diag_C < 0] = 0
    update_node_attribute(G, name, diag_C)
