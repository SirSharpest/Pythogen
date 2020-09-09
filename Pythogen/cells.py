import numpy as np


class Cells:
    def __init__(self, meanCellRadius, meanPDRadius, meanPDNum=1e3,
                 cellRadiusVariationPC=0, PDRadiusVariationPC=0,
                 PDNumVariationPC=0, deadCellPC=0,
                 cellSizeGradientPC=0):

        self.meanCellRadius = meanCellRadius
        self.meanPDRadius = meanPDRadius
        self.meanPDNum = meanPDNum
        self.cellRadiusVariationPC = cellRadiusVariationPC
        self.PDRadiusVariationPC = PDRadiusVariationPC
        self.PDNumVariationPC = PDNumVariationPC
        self.deadCellPC = deadCellPC
        self.cellSizeGradientPC = cellSizeGradientPC

    def apply_deadcells(self, G):
        for i, cell in G.nodes(data=True):
            cell['deadcell'] = np.random.choice(
                [True, False], p=[self.deadCellPC, 1-self.deadCellPC])
