import ROOT
import numpy as np
#import root_numpy
#import root_pandas
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize, LogNorm
import torch
from torch.autograd import Variable, grad
from root_numpy import root2array
import os
import logistics
from tqdm import tqdm
from rebin import rebin


#ROOT.gSystem.Load("/usr/local/bin/Delphes-3.3.3/external/ExRootAnalysis/libExRootAnalysis.so")
# ROOT.gROOT.ProcessLine('.include /usr/local/bin/Delphes-3.4.1')
ROOT.gSystem.Load("/usr/local/bin/Delphes-3.3.3/libDelphes.so")



def load_G4electron_data_single_event(root_file, Normalized=True, edep_min = 100, event_num = 0, scale = 'normal'):
    f_root = ROOT.TFile(root_file);
    chain_name = str("Event" + str(event_num))
    chain = ROOT.TChain(chain_name);
    chain.Add(root_file);
    myTree = f_root.Get(chain_name);
    #print f_root

    # eta restricted and PT restricted > 10 Mev and -2.5 < eta < 2.5 by Delphes card
    x_list = []
    y_list = []
    edep_list = []
    counter = 0
    for entry in myTree:
        #for particle in event.Electron:
            #print("Electron eta phi: ", particle.Eta, particle.Phi)
            #if particle.PT < edep_min:
            #    continue
            #else:
        x_list.append(np.float64(entry.XScinti))
        y_list.append(np.float64(entry.YScinti))
        edep_list.append(np.float64(entry.EdepScinti))


    #print type(x_list[0])
    x_array = np.array(x_list)
    y_array = np.array(y_list)
    edep_array = np.array(edep_list)
    g4_data = np.column_stack((x_array,y_array,edep_array))


    # if Normalized:
    #     data, means_stds = feature_normalize(g4_data, scale = scale)
    # else:
    #     data = g4_data
    #     means_stds = [g4_data.mean(), g4_data.std()]

    #returns [eta, phi, PT] for electron
    return g4_data#, means_stds



def load_G4electron_data(root_file, Normalized=True, edep_min = 100, scale = 'normal', num_events = 10):
    

    f_root = ROOT.TFile(root_file);

    
    image_list = []
    full_images = []
    counter = 0
    for i in range(num_events):
        if i % 20 == 0:
            print "Loading event: ", i
        chain_name = "Event" + str(i)
        chain = ROOT.TChain(chain_name);
        chain.Add(root_file);
        myTree = f_root.Get(chain_name); 

        x_list = []
        y_list = []
        edep_list = []

        for entry in myTree:

            x_list.append(np.float64(entry.XScinti))
            y_list.append(np.float64(entry.YScinti))
            edep_list.append(np.float64(entry.EdepScinti))
            
        counter +=1
        x_array = np.array(x_list)
        y_array = np.array(y_list)
        edep_array = np.array(edep_list)
        full_image = np.column_stack((x_array,y_array,edep_array))
        xedges, yedges = np.linspace(-50, 50, 25), np.linspace(-50,50,25)
        image_binned, _ , _  = np.histogram2d(x_array, y_array, bins = (xedges,yedges), weights=edep_array )
        if counter < 2:
            print image_binned.shape
        #image_binned = submatsum(full_image, 64, 64)
        image_list.append(image_binned)
        full_images.append(full_image)
        #plt.imshow(image_binned)
        #plt.show()


    print "X: ", full_image[:10,0]
    print "Y: ", full_image[:10,1]
    print "E: ", full_image[:10,2]

    #g4_data = np.array(full_images)
    g4_data = full_images

    # if Normalized:
    #     data, means_stds = feature_normalize(g4_data, scale = scale)
    # else:
    #     data = g4_data
    #     means_stds = [g4_data.mean(), g4_data.std()]

    #returns [eta, phi, PT] for electron
    return image_list, g4_data#, means_stds

#g4_data, g4_means_stds = load_G4electron_data("B4_10k.root", Normalized=False)

#print g4_data.shape



def rtnpy_load_data(filename, Normalized = True, from_numpy=True, num_events = 10, image_size = [32,64], Save = True):


    if from_numpy == True:
        image_array = np.load(filename)
        nans_list = []
        for i in range(image_array.shape[0]):
            if np.isnan(image_array[i][0]).any():
                nans_list.append(i)

        image_array_new = np.delete(image_array,nans_list,axis=0)

        return image_array_new

    else:
        image_lists = []
        for size in image_size:
            new_list = []
            image_lists.append(new_list)

        filename = "build/" + filename

        f = ROOT.TFile( filename)
        dirlist = f.GetListOfKeys()
        iter = dirlist.MakeIterator()
        key = iter.Next()
        names = []
        td = None
        for i in range(1000000):
            if key == None:
                print "Total # Events in ROOT file:", len(names)
                break
            name = key.ReadObj().GetName()
            if "Event" in name:
                names.append(name)
            key = iter.Next()

        if num_events > len(names):
            raise ValueError("The number of events requested (" +str(num_events) +") is greater than the number in the ROOT file (" + str(len(names)) + ").")


        for i in tqdm(range(num_events)):
            event_name = "Event" + str(i)
            if i % 250 == 0:
                print "Processing", event_name
            x_tree = root2array(filename, event_name, ['XScinti'])#
            x_tree_float = x_tree.astype('float64')
            y_tree = root2array(filename, event_name, ['YScinti'])
            y_tree_float = y_tree.astype('float64')
            e_tree = root2array(filename, event_name, ['EdepScinti'])
            e_tree_float = e_tree.astype('float64')


            #print type(x_tree[0])
            for i in range(len(image_lists)):
                size = image_size[i]
              #  print(size)
                xedges, yedges = np.linspace(-50, 50, size+1), np.linspace(-50,50,size+1)
                image_binned, _ , _  = np.histogram2d(x_tree_float, y_tree_float, bins = (xedges,yedges), weights=e_tree_float )
                image_lists[i].append(image_binned)


        image_arrays = []
        for i in range(len(image_lists)):
            image_array = np.dstack(image_lists[i]).T
            image_array = image_array[:,np.newaxis,:,:]
            image_arrays.append(image_array)

        if Save == True:
            print("SAVING FILES")
            directory = os.getcwd() + "/numpy_data/"
            #print(directory)
            for i in range(len(image_lists)):
                filename = "geant4Data_" +str(num_events) + "Events_" + str(image_size[i]) + "ImageSize_800MeV_ScintiAbsoThickness75_8"
                np.save(directory+filename, image_arrays[i])

        return image_arrays


def rebin_histos(filename, new_size, num_events = None, Save = True):

    data = np.load("/home/chris/G4Builds/Geant4HadronCalorimeter/numpy_data/"+filename)
    data_to_use = np.squeeze(data)
    old_size = data_to_use.shape[1]
    print("OLD SIZE:", old_size)
    print("NEW SIZE:", new_size)

    if num_events != None:
        data_to_use = data_to_use[:num_events]
    else:
        num_events = data_to_use.shape[0]

    print("NUM EVENTS:", num_events)

    rebinned_images = []
    x1, y1 = np.linspace(-100, 100, old_size+1), np.linspace(-100,100,old_size+1)
    xedges, yedges = np.linspace(-100, 100, new_size+1), np.linspace(-100,100,new_size+1)
    for image in tqdm(data_to_use):
        new_image = rebin.rebin2d(x1,y1,image,xedges,yedges).T
        rebinned_images.append(new_image)


    rebinned_images_array = np.dstack(rebinned_images).T
    rebinned_images_array = rebinned_images_array[:,np.newaxis,:,:]

    directory = os.getcwd() + "/numpy_data/"
    filename = "rebinnedGeant4Data_" +str(num_events) + "Events_" + str(old_size) + "OldSize_" + str(new_size) + "NewSize"
    np.save(directory+filename, rebinned_images_array)

    return 


#rtnpy_load_data("B4_20k.root", num_events = 200, image_size=32, from_numpy = False)
#load_G4electron_data("B4.root", num_events = 1)


#test_data = np.load("/home/chris/G4Builds/Geant4HadronCalorimeter/numpy_data/geant4Data_12Events_64ImageSize.npy")
#print(test_data.shape)


def submatsum(data,n,m):
    # return a matrix of shape (n,m)
    bs = data.shape[0]//n,data.shape[1]//m  # blocksize averaged over
    return np.reshape(np.array([np.sum(data[k1*bs[0]:(k1+1)*bs[0],k2*bs[1]:(k2+1)*bs[1]]) for k1 in range(n) for k2 in range(m)]),(n,m))
