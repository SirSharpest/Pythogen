import numpy as np

# For signals!


def apply_deadcells(G, E):
    for cell in G.nodes(data=True):
        if 'deadcell' in cell[1] or cell[1]['radius_ep'] == 0:
            if cell[1]['deadcell'] or cell[1]['radius_ep'] == 0:
                E[:, cell[0]] = 0
                E[cell[0]] = 0


def apply_asym_diffusion(G, E, x=10):
    for cell in G.nodes(data=True):
        # Find all centre cells and if they have non-centre
        # neighbours, reduce connectivity by factor of X
        if cell[1]['centre']:
            for n in cell[1]['neighbours']:
                if G.nodes(data=True)[n]['centre'] == False:
                    E[cell[0], n] = E[cell[0], n]/x


# For Cells - refactor
def apply_deadcells(G, deadCellPC=0):
    for i, cell in G.nodes(data=True):
        cell['deadcell'] = np.random.choice(
            [True, False], p=[deadCellPC, 1-deadCellPC])
