import numpy as np


class Cells:
    def __init__(self, meanCellRadius, meanPDRadius, meanPDNum=1e3,
                 cellRadiusVariationPC=0, PDRadiusVariationPC=0,
                 PDNumVariationPC=0,
                 cellSizeGradientPC=0):

        self.meanCellRadius = meanCellRadius
        self.meanPDRadius = meanPDRadius
        self.meanPDNum = meanPDNum
        self.cellRadiusVariationPC = cellRadiusVariationPC
        self.PDRadiusVariationPC = PDRadiusVariationPC
        self.PDNumVariationPC = PDNumVariationPC
        self.cellSizeGradientPC = cellSizeGradientPC
        self.onAdds = []

    def run_onAdd(self, G):
        for f in self.onAdds:
            f(G)

    def add_onAdd(self, f):
        self.onAdds.append(f)
