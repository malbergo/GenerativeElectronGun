from geant_dataloader import rebin_histos
import numpy as np
import matplotlib.pyplot as plt


old_size = 64
new_size = 32
filename ="geant4Data_20000Events_64ImageSize_1800MeV_ScintiAbsoThickness75_8.npz"

rebin_histos(filename, new_size = 32)


test_load = np.load("/home/chris/G4Builds/Geant4HadronCalorimeter/numpy_data/geant4Data_20000Events_32ImageSize_1800MeV_ScintiAbsoThickness75_8.npz")['array']

print "COMPLETE:", test_load.shape

