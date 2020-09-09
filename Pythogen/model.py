import numpy as np
from networkx import shortest_path_length
from .nx import set_concentration
from .nx import generate_shape
from .nx import get_centre_node, attr_to_arr
from .utility import G_to_pd


class Model:
    def __init__(self, shape, NCells=100, NCellsY=None, NCellsX=None):
        self.shape = shape
        if NCellsY is not None and NCellsX is not None:
            self.G = generate_shape(self.shape, n=NCellsX, m=NCellsY)
        else:
            self.G = generate_shape(self.shape, n=int(
                np.sqrt(NCells)), m=int(np.sqrt(NCells)))
        self.apply_dist_from_centre()
        self.Cells = None
        self.signals = []

    def add_cell_features(self, Cells):
        self.apply_cell_radii(Cells)
        Cells.apply_deadcells(self.G)
        self.Cells = Cells

    def add_signal(self, signal):
        self.signals.append(signal)

        signal.onAdd(self.G)

    def run(self, seconds, dt=1, seconds_per_update=1):
        dfs = []
        epochs = int(seconds_per_update/dt)
        dx = attr_to_arr(self.G, 'radius')
        for update in range(int(seconds/seconds_per_update)):
            self.apply_effective_diffusion(self.Cells)
            for signal in self.signals:
                signal.run_diffuse(self.G, dt, dx, epochs)
            for signal in self.signals:
                signal.interact(self.G)
            df = self.to_pd()
            df['time'] = (update+1)*seconds_per_update
            dfs.append(df)

    def apply_effective_diffusion(self, Cells):
        for signal in self.signals:
            signal.set_Deff(self.G)

    def apply_cell_radii(self, Cells):
        for k, c in self.G.nodes(data=True):
            r_noise = np.random.normal(Cells.meanCellRadius,
                                       Cells.cellRadiusVariationPC *
                                       Cells.meanCellRadius)

            # TODO: need to do a minY maxX normalisation for voronoi plots
            r = r_noise * (Cells.cellSizeGradient * (c['y']) + 1)

            pdr = abs(np.random.normal(Cells.meanPDRadius,
                                       Cells.PDRadiusVariationPC *
                                       Cells.meanPDRadius))

            num_pd = abs(np.random.normal(Cells.meanPDNum,
                                          Cells.meanPDNum*Cells.PDNumVariationPC))

            c['radius'] = r
            c['pd_radius_original'] = pdr
            c['pd_radius'] = pdr
            c['num_pd'] = num_pd

    def apply_dist_from_centre(self):
        centre = get_centre_node(self.G, voronoi=self.shape)
        for n, d in self.G.nodes(data=True):
            d['distCentre'] = shortest_path_length(self.G, n, centre)

    def to_pd(self):
        return G_to_pd(self.G, self.shape)
