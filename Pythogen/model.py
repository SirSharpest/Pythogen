import numpy as np
from tqdm import tqdm
from networkx import shortest_path_length
from .nx import set_concentration
from .nx import set_default_edge_weights
from .nx import generate_shape
from .nx import get_centre_node
from .diffuse import calc_D_eff, diffuse
from .utility import G_to_pd


class NetworkModel:
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

    def reset_IC(self):
        if self.IC == 'default':
            set_concentration(self.G, voronoi=self.voronoi,
                              IC_value=self.IC_value)
            set_concentration(self.G, voronoi=self.voronoi,
                              pathogen=True, IC_value=self.pathogen_IC)
        else:
            set_concentration(self.G, self.IC, voronoi=self.voronoi)
        self.totalTime = 0

    def set_model_parameters(self, D, avgCellR=50, PDR=5e-3, PDN=1e3,
                             cellSigmaPC=0, PDSigmaPC=0, PDRSigmaPC=0,
                             yGradientPC=0, deadCellPC=0,
                             DeffEq="NEP", q=1, effectorD=None, productionPC=None, pathoProductionPC=None,
                             IC_value=0.1, pathogen_IC=0.1):
        self.IC_value = IC_value
        self.pathogen_IC = pathogen_IC
        self.reset_IC()
        self.DeffEq = DeffEq
        self.q = q
        self.D = D
        self.effectorD = effectorD
        self.cellSigmaPC = cellSigmaPC
        self.PDSigmaPC = PDSigmaPC
        self.PDRSigmaPC = PDRSigmaPC
        self.yGradientPC = yGradientPC
        self.deadCellPC = deadCellPC
        self.avgCellR = avgCellR
        self.PDR = PDR * np.ones(self.G.number_of_nodes())
        if PDRSigmaPC > 0:
            self.PDR *= (np.random.randint(-int(self.PDRSigmaPC*100),
                                           int(self.PDRSigmaPC*100),
                                           self.G.number_of_nodes()) / 100) + 1
        self.PDN = PDN
        self.PDArea = np.pi*(self.PDR**2)
        self.avgCellSA = (4*np.pi*(self.avgCellR**2))
        self.PD_per_um2 = PDN / self.avgCellSA
        self.apply_deadcells()
        self.apply_radius()
        self.productionPC = productionPC
        self.pathoProductionPC = pathoProductionPC

    def apply_deadcells(self):
        centre = get_centre_node(self.G, self.voronoi)

        for i, cell in self.G.nodes(data=True):
            cell['deadcell'] = np.random.choice(
                [True, False], p=[self.deadCellPC, 1-self.deadCellPC])

    def apply_radius_G(self, upperLim=200, lowerLim=1):
        for PDR, (k, v) in zip(self.PDR, self.G.nodes(data=True)):
            noisy_size = np.random.normal(
                self.avgCellR, self.cellSigmaPC*self.avgCellR)
            r = noisy_size * (self.yGradientPC * (v['y']+1))
            if r < lowerLim or r > upperLim:
                r = self.avgCellR
            v['r'] = r
            v['PDR_orig'] = PDR
            v['PDR'] = PDR

    def apply_radius(self):
        self.apply_radius_G()
        self.Rn = np.array([v['r'] for k, v in self.G.nodes(data=True)])
        self.PD_per_cell = (
            np.around(self.PD_per_um2 * (4*np.pi*(self.Rn**2))))
        if self.PDSigmaPC > 0:
            self.PD_per_cell *= (np.random.randint(-int(self.PDSigmaPC*100),
                                                   int(self.PDSigmaPC*100),
                                                   self.Rn.shape) / 100) + 1

        self.Ep = self.PD_per_cell * self.PDArea
        self.Eps = self.Ep/self.PD_per_cell
        self.set_effective_diffusion()

    def update_mid_run(self):
        # Need to define closure rate and conditions
        self.PDR = np.zeros(self.G.number_of_nodes())
        for idx, (k, v) in enumerate(self.G.nodes(data=True)):
            # here could add effector interactions to widen...
            v['PDR'] = v['PDR_orig'] * (1 - v['C'])
            self.PDR[idx] = v['PDR']

        self.PDArea = np.pi*(self.PDR**2)
        self.Ep = self.PD_per_cell * self.PDArea
        self.Eps = self.Ep/self.PD_per_cell
        self.set_effective_diffusion()

    def run(self, seconds, dt=1e-2, reapply_randomDC=False, reapply_randomR=False, reset_time=True, reset_IC=True, constMax=False, pathogen=False, dynamic=False, time_per_update=1, perTurnData=False):
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

        if pathogen:
            dynamic = True

        if dynamic:
            n_epochs_per_update = int(time_per_update/dt)
            num_runs = int(self.epochs / n_epochs_per_update)
            for _ in range(num_runs):
                for p in [True, False]:
                    # Update everything...
                    self.update_mid_run()
                    diffuse(self.G, (self.effectorDeff if p else self.Deff), dt, self.Rn,
                            n_epochs_per_update, deadcells=True, progress=(not self.quiet),
                            constMax=constMax, voronoi=self.voronoi,
                            pathogen=p, productionPC=(self.pathoProductionPC if p else self.productionPC))
                    if perTurnData:
                        yield self.get_df()

        else:
            diffuse(self.G, self.Deff, dt, self.Rn,
                    self.epochs, deadcells=True, progress=(not self.quiet),
                    constMax=constMax, voronoi=self.voronoi, pathogen=pathogen, productionPC=self.productionPC)

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
            if self.effectorD is not None:
                self.effectorDeff = np.array([calc_D_eff(r, self.effectorD, n, ep)
                                              for r, ep, n in zip(self.Rn,
                                                                  self.Eps,
                                                                  self.PD_per_cell)])

        elif self.DeffEq == "Deinum":
            def Deinum(D, q, l): return (D*q*l)/(D+q*l)
            self.Deff = np.array([Deinum(self.D, self.q, r*2)
                                  for r in self.Rn])

            if self.effectorD is not None:
                self.effectorDeff = np.array([Deinum(self.effectorD, self.q, r*2)
                                              for r in self.Rn])

        elif self.DeffEq == 'D':
            self.Deff = self.D
            if self.effectorD is not None:
                self.effectorDeff = self.effectorD
