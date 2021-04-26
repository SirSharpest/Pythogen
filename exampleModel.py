import numpy as np

from Pythogen import cells, model, signals

meanCellRadius, meanPDRadius = 25, 5e-3

cell_params = cells.Cells(meanCellRadius, meanPDRadius, cellSizeGradientPC=0.5)

defSignal = signals.Signal(300, 0, 'defenceSignal')


def init_func(signal, G, names):
    idx = np.random.randint(G.number_of_nodes())
    G.nodes(data=True)[idx]['defenceSignal'] = 1


defSignal.add_onAdd_function(init_func)

mdl = model.Model('voronoi')
mdl.add_cell_features(cell_params)
mdl.add_signal(defSignal)

mdl.run(10)

df = mdl.to_pd()
print(df.sort_values(by='radius', ascending=False).head())
