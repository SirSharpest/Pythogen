import numpy as np
import networkx as nx
from .diffuse import calc_cell_D_eff_NEP, diffuse


class Signal:
    def __init__(self, D, initial_value, name, producesSelf=False,
                 decays=False,
                 deff_f=calc_cell_D_eff_NEP,
                 productionThreshold=0,
                 productionRate=0,
                 DoesntDiffuse=False, decayRate=0):

        self.D = D
        self.name = name
        self.producesSelf = producesSelf
        self.productionThreshold = productionThreshold
        self.productionRate = productionRate
        self.initial_value = initial_value
        self.decays = decays
        self.deff_f = deff_f
        self.Deff = None
        self.interactions = []
        self.interaction_names = []
        self.onAdds = []
        self.onAdd_names = []
        self.DoesntDiffuse = DoesntDiffuse
        self.variables = {}
        self.decayRate = decayRate
        self.diffusion_fs = []

    def run_decay(self, G):
        if self.decays:
            if self.decayRate > 0:
                for k, c in G.nodes(data=True):
                    if c[self.name] > 0:
                        c[self.name] *= 1-self.decayRate

    def run_production(self, G):
        if self.producesSelf:
            if self.productionRate > 0:
                for k, c in G.nodes(data=True):
                    if c[self.name] > self.productionThreshold:
                        c[self.name] *= 1+self.productionRate

    def flatten(self, G):
        for k, c in G.nodes(data=True):
            if c[self.name] < 1e-6:
                c[self.name] = 0
            elif c[self.name] > 1:
                c[self.name] = 1

    def set_Deff(self, G):
        if self.DoesntDiffuse:
            self.Deff = np.zeros(G.number_of_nodes())
            return 0
        self.Deff = np.zeros(G.number_of_nodes())
        for idx, (k, c) in enumerate(G.nodes(data=True)):
            if c['radius_ep'] <= 0:
                continue

            self.Deff[idx] = self.deff_f(c, self)


    def run_diffuse(self, G, dt, dx, epochs):
        if not self.DoesntDiffuse:
            diffuse(G, self.Deff, dt, dx, epochs,
                    self.name, self.diffusion_fs, self)
        for k, c in G.nodes(data=True):
            if c[self.name] > 1:
                c[self.name] = 1

    def add_to_cells(self, G):
        for k, c in G.nodes(data=True):
            c[self.name] = 0

    def add_interaction_function(self, f, names=[]):
        self.interactions.append(f)
        self.interaction_names.append(names)

    def add_onAdd_function(self, f, names=[]):
        self.onAdds.append(f)
        self.onAdd_names.append(names)

    def interact(self, G):
        for f, names in zip(self.interactions, self.interaction_names):
            f(self, G, names)

    def onAdd(self, G):
        self.add_to_cells(G)
        for f, names in zip(self.onAdds, self.onAdd_names):
            f(self, G, names)

    def add_diffusion_modifier(self, f):
        self.diffusion_fs.append(f)
