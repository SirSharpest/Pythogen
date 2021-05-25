import numpy as np
from time import time 
from Pythogen import cells, model, signals

t1 = time()
meanCellRadius, meanPDRadius = 25, 5e-3
np.random.seed()
cell_params = cells.Cells(meanCellRadius, meanPDRadius, cellSizeGradientPC=0.5)

defSignal = signals.Signal(300, 0, 'defenceSignal')


def init_func(signal, G, names):
    idx = np.random.randint(G.number_of_nodes())
    G.nodes(data=True)[idx]['defenceSignal'] = 1


defSignal.add_onAdd_function(init_func)

mdl = model.Model('voronoi')
mdl.add_cell_features(cell_params)
mdl.add_signal(defSignal)

mdl.run(60*60*10)

df = mdl.to_pd()
print(df.sort_values(by='radius', ascending=False).head())
print(f"Took {time()-t1} seconds")
