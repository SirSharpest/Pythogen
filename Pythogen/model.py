import numpy as np
from tqdm import tqdm
from networkx import shortest_path_length
from .nx import set_concentration
from .nx import set_default_edge_weights
from .nx import generate_shape
from .nx import get_centre_node
from .diffuse import calc_D_eff, diffuse, diffuse_numba
from .utility import G_to_pd


class LatticeNetworkModel:
    def __init__(self, shape, sizeN, sizeM, IC='default', quiet=False):
        self.G = generate_shape(shape, n=sizeN,
                                m=sizeM)

        self.quiet = quiet
        self.voronoi = True if 'voronoi' in shape else False
        self.apply_dist_from_centre()
        self.shape = shape
        set_default_edge_weights(self.G)
        self.N = np.array([len([_ for _ in self.G.neighbors(n)])
                           for n in self.G.nodes()])

        self.IC = IC
        self.reset_IC()

    def reset_IC(self):
        if self.IC == 'default':
            set_concentration(self.G, voronoi=self.voronoi)
        else:
            set_concentration(self.G, self.IC, voronoi=self.voronoi)
        self.totalTime = 0

    def set_model_parameters(self, D, avgCellR=50, PDR=5e-3, PDN=1e3,
                             cellSigmaPC=0, yGradientPC=0, deadCellPC=0,
                             bombardedDCPC=None,  DeffEq="NEP", q=1):
        self.DeffEq = DeffEq
        self.q = q
        self.D = D
        self.cellSigmaPC = cellSigmaPC
        self.yGradientPC = yGradientPC
        self.deadCellPC = deadCellPC
        self.bombardedDCPC = bombardedDCPC if bombardedDCPC is not None else deadCellPC
        self.avgCellR = avgCellR
        self.PDR = PDR
        self.PDN = PDN
        self.PDArea = np.pi*(self.PDR**2)
        self.avgCellSA = (4*np.pi*(self.avgCellR**2))
        self.PD_per_um2 = PDN / self.avgCellSA
        self.apply_deadcells()
        self.apply_radius()

    def apply_deadcells(self):
        centre = get_centre_node(self.G, self.voronoi)

        for i, cell in self.G.nodes(data=True):
            cell['deadcell'] = np.random.choice(
                [True, False], p=[self.deadCellPC, 1-self.deadCellPC])
            if i == centre:
                cell['deadcell'] = np.random.choice(
                    [True, False], p=[self.bombardedDCPC, 1-self.bombardedDCPC])

    def apply_radius_G(self, upperLim=200, lowerLim=1):
        # Removed these two lines, they don't do anything???
        # _, centreY = (self.G.nodes()[get_centre_node(self.G, self.voronoi)]['x'],
        #               self.G.nodes()[get_centre_node(self.G, self.voronoi)]['y'])
        for k, v in self.G.nodes(data=True):
            noisy_size = np.random.normal(
                self.avgCellR, self.cellSigmaPC*self.avgCellR)
            r = noisy_size * (self.yGradientPC * (v['y']+1))
            if r < lowerLim or r > upperLim:
                r = self.avgCellR
            v['r'] = r

    def apply_radius(self):
        self.apply_radius_G()
        self.Rn = np.array([v['r'] for k, v in self.G.nodes(data=True)])
        self.PD_per_cell = np.around(self.PD_per_um2 * (4*np.pi*(self.Rn**2)))
        self.Ep = self.PD_per_cell * self.PDArea
        self.Eps = self.Ep/self.PD_per_cell
        self.set_effective_diffusion()

    def run(self, seconds, dt=1e-4, reapply_randomDC=False, reapply_randomR=False, reset_time=True, reset_IC=True,  bombardment=False):
        self.epochs = int(seconds/dt)
        if reset_IC:
            self.reset_IC()
        if reset_time:
            self.totalTime = seconds
        else:
            self.totalTime += seconds
        if reapply_randomR:
            self.apply_radius()
        if reapply_randomDC:
            self.apply_deadcells()

        diffuse(self.G, self.Deff, dt, self.Rn,
                self.epochs, deadcells=True, progress=(not self.quiet),
                bombardment=bombardment, voronoi=self.voronoi)

    def get_df(self, rep=1, apply_cutoff=None):

        centreX, centreY = self.G.nodes[get_centre_node(
            self.G, self.voronoi)]['x'], self.G.nodes[get_centre_node(self.G, self.voronoi)]['y']

        for n, d in self.G.nodes(data=True):
            x, y = d['x'], d['y']
            if x < centreX:
                if y > centreY:
                    q = 1
                else:
                    q = 2
            else:
                if y > centreY:
                    q = 3
                else:
                    q = 4
            if x == centreX and y == centreY:
                q = 0
            d['quadrant'] = q
            d['half'] = 1 if q < 3 else 2
            d['num_neighbours'] = self.G.degree[n]
            d['neighbours'] = [n for n in self.G.neighbors(n)]
        df = G_to_pd(self.G, self.shape, self.Deff, rep)
        df['sigma'] = self.cellSigmaPC
        df['gradient'] = self.yGradientPC
        df['DC'] = self.deadCellPC
        df['time'] = self.totalTime
        df['C'] = df['C'].astype('float64')
        return df

    def apply_dist_from_centre(self):
        centre = get_centre_node(self.G, voronoi=self.voronoi)
        if self.quiet:
            for n, d in self.G.nodes(data=True):
                d['distCentre'] = shortest_path_length(self.G, n, centre)
        else:
            print('Calculating distances...')
            for n, d in tqdm(self.G.nodes(data=True)):
                d['distCentre'] = shortest_path_length(self.G, n, centre)

    def set_effective_diffusion(self):
        self.N = np.array([len([_ for _ in self.G.neighbors(n)])
                           for n in self.G.nodes()])

        if self.DeffEq == "NEP":
            self.Deff = np.array([calc_D_eff(r, self.D, n, ep)
                                  for r, ep, n in zip(self.Rn,
                                                      self.Eps,
                                                      self.PD_per_cell)])
        elif self.DeffEq == "Deinum":
            def Deinum(D, q, l): return (D*q*l)/(D+q*l)
            self.Deff = np.array([Deinum(self.D, self.q, r*2)
                                  for r in self.Rn])

        elif self.DeffEq == 'D':
            self.Deff = self.D
