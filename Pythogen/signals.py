import numpy as np
from .diffuse import calc_D_eff, diffuse


class Signal:
    def __init__(self, D, initial_value, name, producesSelf=False,
                 productionThreshold=0,
                 productionRate=0, DeffEq='NEP',
                 DoesntDiffuse=False, decayRate=0):

        self.D = D
        self.name = name
        self.producesSelf = producesSelf
        self.productionThreshold = productionThreshold
        self.productionRate = productionRate
        self.initial_value = initial_value
        self.DeffEq = DeffEq
        self.Deff = None
        self.interactions = []
        self.interaction_names = []
        self.onAdds = []
        self.onAdd_names = []
        self.DoesntDiffuse = DoesntDiffuse
        self.variables = {}
        self.decayRate = decayRate

    def run_decay(self, G):
        if self.decayRate > 0:
            for k, c in G.nodes(data=True):
                if c[self.name] > 0:
                    c[self.name] *= 1-self.decayRate

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
        Deff = np.ones(G.number_of_nodes())
        if self.DeffEq == 'NEP':
            for idx, (k, c) in enumerate(G.nodes(data=True)):
                Deff[idx] = calc_D_eff(c['radius'], self.D,
                                       c['num_pd'],
                                       np.pi*(c['pd_radius'])**2)
        else:
            Deff = Deff * self.D
        self.Deff = Deff

    def run_diffuse(self, G, dt, dx, epochs):
        if not self.DoesntDiffuse:
            diffuse(G, self.Deff, dt, dx, epochs, self.name)
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
