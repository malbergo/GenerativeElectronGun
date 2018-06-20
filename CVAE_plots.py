
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import gridspec
import torch
import random
import datetime
import os
from logistics import *
import torch.nn as nn
import torch.nn.functional as F
#from CVAEElectronGun  import *




plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.figsize'] = (8.0, 8.0)


def make_samples(model, avg, conditions, scale, norm_scale, zdim, device, imageSize, normed_array, n_events=500, real=True, fake = True):
    
    sample = torch.randn(n_events, zdim).to(device)
    #print sample.shape
    energies = torch.tensor(np.random.choice(np.squeeze(conditions), size=n_events)).type(torch.FloatTensor).to(device).unsqueeze(1)
    #print energies.shape
    sample_conds = torch.cat((sample,energies),1)
    gen_sample = model.decode(sample_conds).cpu()
    sample_new = gen_sample.view(n_events, 1, imageSize, imageSize).detach().numpy().squeeze()
    fake_images_array = unnormalize(sample_new, scale = scale, norm_scale = norm_scale)
    
    
    randints = np.random.randint(low = 0,high = normed_array.shape[0], size = n_events)
    real_images_array = unnormalize(normed_array[randints].squeeze(),norm_scale = norm_scale, scale = scale)
    
    
    # if you want the average image, take the mean. if not, will return a whole set of images
    if avg == True:
        fake_images_array = np.sum(fake_images_array,axis=0) / n_events
        real_images_array = np.sum(real_images_array, axis=0) / n_events
    
    if real == True and fake == True:
        return [real_images_array,fake_images_array], n_events
    elif (real == False) and (fake == True):
        return fake_images_array
    else:
        return real_images_array
    
    



def plot_avg(data, n_events, epoch, imageSize, last_decode_act, withMarginals=True, save_dir = None ):

    test_noNans = np.copy(data)
    test_unnormed = data
    if last_decode_act == F.sigmoid:
        test_unnormed[test_unnormed < 0.001] = np.nan
    else:
        test_unnormed[test_unnormed <= 0.0] = np.nan
    
    fig=plt.figure(figsize=(6, 6))
    xran = (-50,50)
    yran = (-50,50)
    extent = xran + yran
    cmap = sns.cubehelix_palette(dark = 0.4, light=0.93, gamma = 2.5, hue = 1, start =0, as_cmap=True)
    color_list = sns.cubehelix_palette(dark=0.4, light=0.93, gamma=2.5, hue=1).as_hex()
    
    if withMarginals == False:
        marginals_str = 'woMarginals'
        plt.hist2d(fake.to('cpu').detach().numpy()[0][0][:,0], fake.to('cpu').detach().numpy()[0][0][:,1])
        im = ax.imshow(test_unnormed, vmin = 0, extent=extent, origin='lower', cmap=cmap)
        cbar = plt.colorbar(im, fraction=0.05, pad=0.05)
        cbar.set_label(r'Pixel $E_{dep}$ (MeV)', y=0.85)
        if save_dir != None:
            #directory = "/home/chris/Documents/MPhilProjects/ForViewing/Geant4/SingleLayerEGun/AverageImage/"
            filename = "CVAE_AvgEdepOver" + str(n_events) + "Events_Epochs" + str(epoch) +" .png" 
            plt.savefig(directory + filename)
        
    else:
        marginals_str = 'withMarginals'
        img=test_noNans
        t = np.arange(-50,50, 100/float(imageSize))
        #t = np.arange(img.shape[0])
        f = np.arange(-50,50, 100/float(imageSize))
        #f = np.arange(img.shape[1])
        flim = (f.min(), f.max())
        tlim = (t.min(), t.max())

        gs = gridspec.GridSpec(2, 2, width_ratios=[1,5], height_ratios=[1,5])
        gs.update(hspace=0, wspace=0)

        ax = fig.add_subplot(gs[1,1])
        im = ax.imshow(test_unnormed, vmin = 0, extent = extent, origin = 'lower', cmap = cmap)
        cbaxes = fig.add_axes([0.97, 0.18, 0.03, 0.55]) 
        cbar = plt.colorbar(mappable=im, ticks = None, cax=cbaxes, use_gridspec=True)
        ax.yaxis.set_ticks_position('right')
        cbar.set_label(r'Pixel $E_{dep}$ (MeV)', y=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines['left'].set_visible(False)

        axl = fig.add_subplot(gs[1,0], sharey=ax)
        axl.fill_between(img.mean(1), f, alpha = 0.7, color = color_list[1])
        axl.invert_xaxis()
        axb = fig.add_subplot(gs[0,1], sharex=ax)
        axb.fill_between(t, img.mean(0), alpha =0.7, color= color_list[1])

        plt.setp(axl.get_yticklabels(), visible=False)
        plt.setp(axb.get_xticklabels(), visible=False)
        plt.setp(axl.get_xticklabels(), visible=False)
        plt.setp(axb.get_yticklabels(), visible=False)

        axl.yaxis.set_ticks_position('none')
        axb.xaxis.set_ticks_position('none')
        axl.xaxis.set_ticks_position('none')
        axb.yaxis.set_ticks_position('none')


        axl.spines["top"].set_visible(False)
        axl.spines['right'].set_visible(False)
        axl.spines['left'].set_visible(False)
        axl.spines['bottom'].set_visible(False)
        axb.spines["top"].set_visible(False)
        axb.spines["right"].set_visible(False)
        axb.spines["left"].set_visible(False)
        axb.spines["bottom"].set_visible(False)
        ax.set_xlim(tlim)
        ax.set_xlabel(r"$\mathit{x}$", fontsize = 12)
        ax.xaxis.set_label_coords(0.02,-0.05)
        ax.set_ylim(tlim)
        ax.set_ylabel(r"$\mathit{y}$", fontsize = 12, rotation = 0)
        ax.yaxis.set_label_coords(1.07,0.98)
        if save_dir != None:
            #learning_rate = '%.0E' % Decimal(lr)
            #directory = "/home/chris/Documents/MPhilProjects/ForViewing/plots/Geant4/SingleLayerEGun/VAE/"
            filename = "CVAE_AvgEdep"+str(marginals_str) + "Over" + str(n_events) + "Events_Epoch" + str(epoch) + ".png"
            plt.savefig(save_dir + filename, bbox_inches='tight')
        plt.show()
    return


# def plot_sns(n_events = 500, n_samples = 3, save_dir = None, epoch = epoch, batchSize=batchSize, imageSize=imageSize, lr=lr, norm_scale = norm_scale):
    
#     fig, axn = plt.subplots(1, n_samples, figsize=(12,4), sharex=True, sharey=True)
#     xran = (-50,50)
#     yran = (-50,50)
#     extent = xran + yran

#     #xticks = [-50,-25,0,25,50]
#     #yticks = [-50,-25,0,25,50]
#     for i, ax in enumerate(axn.flat):
#         noise = torch.randn(batchSize, nz, 1, 1, device=device)
#         fake = netG(noise)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(True)
#         if i != 0:
#             ax.spines['left'].set_visible(False)
#         if i == 0:
#             ax.set_xlabel(r"$\mathit{x}$", fontsize=13, x = 0.96)
#             ax.set_ylabel(r"$\mathit{y}$", fontsize=13, rotation=0, y = .95)
#         ax.set_title("Ex " +str(i), x =0.85, y = 0.88, alpha=0.6, fontweight='heavy', fontsize=11)
#         test_image = fake.to('cpu').detach().numpy()[0][0]
#         test_unnormed =  unnormalize(test_image, scale)
#         #test_unnormed = logistic_unnormalize(test_image, scale)
#         test_unnormed[test_unnormed < 0.1] = np.nan
#         cmap = sns.cubehelix_palette(dark = 0.4, light=0.99, gamma = 2.5, hue = 1, start =0, as_cmap=True)
#         #sns.heatmap(test_unnormed, ax=ax, cmap=cmap,
#         #            cbar=i == 0, cbar_ax=None if i else cbar_ax, square=True,
#         #            vmin = 0, vmax = 80, xticklabels = xticks, yticklabels=xticks)
#         im = ax.imshow(test_unnormed, vmin = 0, vmax=10, extent=extent, origin='lower', cmap=cmap)
#         ax.tick_params(axis=u'both', which=u'both',length=0)

#     fig.tight_layout(rect=[0, 0, .9, 1])
#     cbar_ax = fig.add_axes([0.9, 0.25, 0.025, 0.5])
#     fig.colorbar(im, cax=cbar_ax)
#     fig.subplots_adjust(wspace=0.1, hspace=0)
#     fig.suptitle("Samples of Generated Electron Gun Energy Depositions",x=0.5,y=0.99)
#     if save_dir != None:
#         learning_rate = '%.0E' % Decimal(lr)
#         #directory = "/home/chris/Documents/MPhilProjects/ForViewing/plots/Geant4/SingleLayerEGun/VAE/"
#         filename = "VAE_3SampleEdepOver" + str(n_events) + "Events_" + str(imageSize) + "x" +str(imageSize) \
#                     + "Image_Epoch" + str(epoch) + "_" + norm_scale + "Normalized_" + str(batchSize) + "batchSize_" +  str(learning_rate) + "lr.png"
#         plt.savefig(save_dir + filename, bbox_inches='tight')
#     plt.show()
#     return



def plot_avg_both(real_data, fake_data, n_events, epoch, imageSize,withMarginals=True, save_dir=None):
    fig=plt.figure(figsize=(10,5))
    xran = (-50,50)
    yran = (-50,50)
    extent = xran + yran

    #real_data = real_image
    img = real_data
    test_unnormed = fake_data
    test_noNans = np.copy(test_unnormed)
    img2 = test_noNans
    t = np.arange(-50,50, 100/float(imageSize))
    #t = np.arange(img.shape[0])
    f = np.arange(-50,50, 100/float(imageSize))
    #f = np.arange(img.shape[1])
    flim = (f.min(), f.max())
    tlim = (t.min(), t.max())


    gs = gridspec.GridSpec(2, 4, width_ratios=[5,1,1,5], height_ratios=[1,5])
    gs.update(hspace=0, wspace=0)
    ax1 = fig.add_subplot(gs[1,0])
    axl = fig.add_subplot(gs[1,1], sharey=ax1)
    axb = fig.add_subplot(gs[0,0], sharex=ax1)
    ax2 = fig.add_subplot(gs[1,3])
    axl2= fig.add_subplot(gs[1,2], sharey=ax2)
    axb2= fig.add_subplot(gs[0,3], sharex=ax2)
    plt.setp(axl.get_yticklabels(), visible=False)
    plt.setp(axb.get_xticklabels(), visible=False)
    plt.setp(axl.get_xticklabels(), visible=False)
    plt.setp(axb.get_yticklabels(), visible=False)
    plt.setp(axl2.get_yticklabels(), visible=False)
    plt.setp(axb2.get_xticklabels(), visible=False)
    plt.setp(axl2.get_xticklabels(), visible=False)
    plt.setp(axb2.get_yticklabels(), visible=False)
    axl.yaxis.set_ticks_position('none')
    axb.xaxis.set_ticks_position('none')
    axl.xaxis.set_ticks_position('none')
    axb.yaxis.set_ticks_position('none')
    axl2.yaxis.set_ticks_position('none')
    axb2.xaxis.set_ticks_position('none')
    axl2.xaxis.set_ticks_position('none')
    axb2.yaxis.set_ticks_position('none')
    cmap = sns.cubehelix_palette(dark = 0.4, light=0.915, gamma = 2.5, hue = 1, start =0, as_cmap=True)
    im = ax1.imshow(real_data, vmin = 0, extent =extent, origin='lower', cmap=cmap)
    ax1.spines["top"].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    color_list = sns.cubehelix_palette(dark=0.4, light=0.915, gamma=2.5, hue=1).as_hex()
    axl.fill_between(img.mean(1), f, alpha = 0.7, color = color_list[1])
    axl.spines["top"].set_visible(False)
    axl.spines['right'].set_visible(False)
    axl.spines['left'].set_visible(False)
    axl.spines['bottom'].set_visible(False)
    axb.fill_between(t, img.mean(0), alpha =0.7, color= color_list[1])
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)
    axb.spines["left"].set_visible(False)
    axb.spines["bottom"].set_visible(False)
    ax1.set_xlim(tlim)
    
    ax1.set_ylim(tlim)


    #RECONSIDER TAKING OUT THE LESS THAN 0 VALUES
    #real_data[real_data < 0.0] = np.nan 


    #test_unnormed[test_unnormed < 0.0] = np.nan
    print test_unnormed.max()
    im = ax2.imshow(test_unnormed, vmin = 0, extent=extent, origin='lower', cmap=cmap)
    ax2.spines["top"].set_visible(False)
    ax2.spines['left'].set_visible(False)
    axl2.fill_between(img2.mean(1), f, alpha = 0.7, color = color_list[1])
    axl2.invert_xaxis()
    axl2.spines["top"].set_visible(False)
    axl2.spines['right'].set_visible(False)
    axl2.spines['left'].set_visible(False)
    axl2.spines['bottom'].set_visible(False)
    axb2.fill_between(t, img2.mean(0), alpha =0.7, color= color_list[1])
    axb2.spines["top"].set_visible(False)
    axb2.spines["right"].set_visible(False)
    axb2.spines["left"].set_visible(False)
    axb2.spines["bottom"].set_visible(False)
    ax2.set_xlim(tlim)
    ax2.set_ylim(tlim)

    ax1.set_xlabel("Real", fontsize=11)  
    ax2.set_xlabel("VAE", fontsize=11)
    ax1.set_ylabel(r"$\mathit{y}$", fontsize = 12, rotation = 0)
    ax1.yaxis.set_label_coords(-0.07,0.98)
    fig.tight_layout(rect=[0, 0, .9, 1])
    cbar_ax = fig.add_axes([0.9, 0.25, 0.025, 0.45])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(r'$E_{dep}$ (MeV)', y =0.85)
    plt.figtext(0.05,0.060,r"$\mathit{x}$", fontsize = 12)
    fig.subplots_adjust(wspace=0.1, hspace=0)
    fig.suptitle(r" CVAE Avg $E_{dep}$ Over " + str(n_events) + " Events, "
                 + str(imageSize) + "x" +str(imageSize) + " \n Image Size, " + str(epoch) 
                 + " Epochs", x = 0.46, y = 0.02)
    if save_dir != None:
        #learning_rate = '%.0E' % Decimal(lr)
        #directory = "/home/chris/Documents/MPhilProjects/ForViewing/plots/Geant4/SingleLayerEGun/VAE/"
        filename = "CVAE_RealandFakeAvgEdep"+str(withMarginals) + "Over" + str(n_events) + "Events_Epoch" +  str(epoch) + ".png"
        plt.savefig(save_dir + filename, bbox_inches='tight')
    plt.show()
    return



def samples(source, epoch, conds, zdim,  beta, norm_scale , scale, imageSize, last_decode_act, device, save_dir=None, rows=2, columns=5 ):
    xran = (-50,50)
    yran = (-50,50)
    extent = xran + yran

    #rows  = 2
    #columns = 5

    if rows <= 3 and columns <=5:
        scale_factor = 3.5
    else:
        scale_factor = 2

    #real = False
    if type(source).__module__ == 'numpy':
        real = True
        randints = np.random.randint(low = 0,high = normed_array.shape[0], size = 64)
        data = unnormalize(source[randints].squeeze(),norm_scale = norm_scale, scale = scale)
        
        #data= next(iter(test_loader))[0].detach().numpy().squeeze()
    else:
        real = False
        sample = torch.randn(64, zdim).to(device)
        energies = torch.tensor(np.random.choice(np.squeeze(conds), size=64)).type(torch.FloatTensor).to(device).unsqueeze(1)
        sample_conds = torch.cat((sample,energies),1)
        gen_sample = source.decode(sample_conds).cpu()
        #sample = model.decode(sample).cpu()
        data = unnormalize(gen_sample.view(64, 1, imageSize, imageSize).detach().numpy().squeeze(), norm_scale = norm_scale, scale = scale)
    
        
    cmap = sns.cubehelix_palette(dark = 0.4, light=0.965, gamma = 2.5, hue = 1, start =0, as_cmap=True)
    
    if (rows == 1) and (columns == 1):

        fig, ax = plt.subplots(rows, columns, figsize=(5,5))
        image_array = data[0]
        if last_decode_act == F.sigmoid:
            image_array[image_array < 0.0001] = np.nan
        else:
            image_array[image_array <= 0] = np.nan
        im = ax.imshow(image_array, cmap=cmap, extent = extent, origin='lower')
        ax.set_xlabel(r"$\mathit{x}$", fontsize=13, x = 0.96)
        ax.set_ylabel(r"$\mathit{y}$", fontsize=13, rotation=0, y = .95)
    else:
        fig, axes =plt.subplots(rows,columns, figsize=(scale_factor*columns, scale_factor*rows), sharex=True,sharey=True)
        #print(len(axes))
        event_number = 0
        plt.locator_params(axis='y', nbins=1)
        plt.locator_params(axis='x', nbins=2)
        #fig.xticks(rotation=45)
        for i in range(rows):
            for j in range(columns):

                #print(range(rows))
                image_array = data[event_number]
            
                if last_decode_act == F.sigmoid:
                    image_array[image_array < 0.0001] = np.nan
                else:
                    image_array[image_array <= 0] = np.nan
                #cmap = sns.light_palette((210, 95, 30), input='husl', as_cmap=True)
                #cmap = sns.dark_palette('muted purple', as_cmap=True, input='xkcd')
                #cmap = sns.color_palette("BrBG",7)
                #img = ax.imshow(image_array[0], vmin = 0, extent=extent, origin='lower', cmap=cmap)
                #axes[i,j].set_aspect('equal')
                plt.axis('on')
                if rows == 1:
                    #axes[j] = plt.subplot(gs1[i,j])
                    if j != 0:
                        axes[j].spines['left'].set_visible(False)
                    else:
                        axes[j].set_xlabel(r"$\mathit{x}$", fontsize=13, x = 0.96)
                        axes[j].set_ylabel(r"$\mathit{y}$", fontsize=13, rotation=0, y = .95)
                    axes[j].spines['top'].set_visible(False)
                    axes[j].spines['right'].set_visible(False)
                    #axes[j].spines['bottom'].set_visible(False)
                    #axes[j].spines['left'].set_visible(False)
                    #axes[j].set_aspect('equal')
                    im =axes[j].imshow(image_array, vmin = 0, extent=extent, origin='lower', cmap=cmap)
                    #axes[i,j].set_xticklabels([-100,0,100],rotation=90)
                    axes[j].set_xticklabels([])
                    axes[j].set_yticklabels([])
                    axes[j].xaxis.set_ticks_position('none')
                    axes[j].yaxis.set_ticks_position('none')
                    event_number +=1
                else:
                    #axes[i,j].spines['top'].set_visible(False)
                    axes[i,j].spines['bottom'].set_visible(True)
                    if j!= 0:
                        axes[i,j].spines['left'].set_visible(False)
                    if i == range(rows)[-1] and j == 0:
                        axes[i,j].set_xlabel(r"$\mathit{x}$", fontsize=13, x = 0.96)
                        axes[i,j].set_ylabel(r"$\mathit{y}$", fontsize=13, rotation=0, y = .95)

                    if j == range(columns)[-1]:
                        axes[i,j].spines['right'].set_visible(True)
                    else:
                        axes[i,j].spines['right'].set_visible(False)
                    im =axes[i,j].imshow(image_array, vmin = 0, extent=extent, origin='lower', cmap=cmap)
                    #axes[i,j].set_xticklabels([-100,0,100],rotation=90)
                    axes[i,j].set_xticklabels([])
                    axes[i,j].set_yticklabels([])
                    axes[i,j].xaxis.set_ticks_position('none')
                    axes[i,j].yaxis.set_ticks_position('none')
                    event_number +=1

    if rows==1 and columns ==1:
        if imageSize == 64:
            cbar_ax = fig.add_axes([0.9, 0.18, 0.02, 0.66])
        elif imageSize == 32:
            cbar_ax = fig.add_axes([0.92, 0.18, 0.02, 0.66])
    else:
        cbar_ax = fig.add_axes([0.92, 0.18, 0.02, 0.66])
    # fig.text(0.5,0.04, "Some very long and even longer xlabel", ha="center", va="center")
    #fig.text(0.05,0.5, "Some quite extensive ylabel", ha="center", va="center", rotation=90)

    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(r'$E_{dep}$ (MeV)', y =0.85)
    #plt.tight_layout()
    fig.subplots_adjust(wspace=-0.085, hspace=0.00)
    if real == True:
        fig.suptitle("Samples of Geant4 Electron Gun Energy Depositions", y = 0.95)
    else:
        if rows==1 and columns ==1:
            fig.suptitle(r"$\beta$-CVAE,$\quad$ $\beta=$"+str(beta), y=0.95)
        else:
            fig.suptitle("CVAE Samples of Generated Electron Gun Energy Depositions, Epoch " + str(epoch), y=0.95)

    num_samples = rows * columns


    if save_dir != None:
        if real == True:
            type_string = "real"
        else:
            type_string = "fake"

        #learning_rate = '%.0E' % Decimal(lr)
        #directory = "/home/chris/Documents/MPhilProjects/ForViewing/plots/Geant4/SingleLayerEGun/VAE/"
        filename = "CVAE"+str(num_samples) + type_string+ "SamplesEdepOver_Epoch" + str(epoch)  + ".png"
        #print(save_dir)
        plt.savefig(save_dir + filename, bbox_inches='tight')


    plt.show()
    return



def calc_sum_difference(real_fake_list):
    real = real_fake_list[0]
    fake = real_fake_list[1]
   # data = (real - fake) 
    sum_diff = (real.sum() - fake.sum()) 
    return sum_diff


def plot_difference(real_fake_list, epoch, imageSize, n_events = 500, save_dir = None ):
#mean_normalizer = data_test[1][data_test[1] > 0.0].max()

    real = real_fake_list[0]
    fake = real_fake_list[1]
    data = (real - fake) 
    
    #sum_diff = (real.sum() - fake.sum()) / data.size()
#plot_avg(data_diff, n_events = 5000, save_dir=None)
    test_noNans = np.copy(data)
    test_unnormed = data
#test_unnormed[test_unnormed < ] = np.nan

    xran = (-50,50)
    yran = (-50,50)
    extent = xran + yran
    
    if imageSize == 64:
        vmin = -4
        vmax = 4
    elif imageSize == 32:
        vmin = -8
        vmax = 8
    
    t = np.arange(-50,50, 100/float(imageSize))
    #t = np.arange(img.shape[0])
    f = np.arange(-50,50, 100/float(imageSize))
    #f = np.arange(img.shape[1])
    flim = (f.min(), f.max())
    tlim = (t.min(), t.max())
    #cmap = sns.cubehelix_palette(dark = 0.4, light=0.93, gamma = 2.5, hue = 1, start =2, as_cmap=True)
    #cmap = sns.light_palette((210, 95, 30), input='husl', as_cmap=True)
    #cmap = sns.color_palette("BrBG", 7, as_cmap=True)
    cmap = sns.diverging_palette(30, 270, s=80, l=55, n=11, as_cmap = True)
    color_list = sns.cubehelix_palette(dark=0.4, light=0.93, gamma=2.5, hue=1).as_hex()


    fig, ax=plt.subplots(figsize=(6, 6))
    #plt.hist2d(fake.to('cpu').detach().numpy()[0][0][:,0], fake.to('cpu').detach().numpy()[0][0][:,1])
    im = ax.imshow(test_unnormed, vmin = vmin,vmax = vmax, extent=extent, origin='lower', cmap='PuOr', label ='VAE')
    cbar = plt.colorbar(im, fraction=0.05, pad=0.05)
    cbar.set_label(r'Pixel $E_{dep}$ (MeV)', y=0.85)
    plt.title(r"$E_{avg}^{G4} - E_{avg}^{CVAE}$, Epoch " + str(epoch))
    ax.text(-45, 39, 'VAE', color='black', 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    #plt.legend(loc = 'best')
    ax.set_xlim(tlim)
    ax.set_xlabel(r"$\mathit{x}$", fontsize = 12)
    ax.xaxis.set_label_coords(0.51,-0.08)
    ax.set_ylim(tlim)
    ax.set_ylabel(r"$\mathit{y}$", fontsize = 12, rotation = 0)
    ax.yaxis.set_label_coords(-.1,0.48)
    ax.spines["top"].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    if save_dir != None:
        #learning_rate = '%.0E' % Decimal(lr)

        filename = "CVAE_EdepDifferenceOver" + str(n_events) + "Events_Epoch" + str(epoch) + ".png"
        plt.savefig(save_dir + filename, bbox_inches='tight')

    plt.show()
    return

# plot all the sum differences, appended into a list by calc_sum_difference function
def plot_sum_difference(sum_diffs, epochs, n_epochs, save_dir = None):
    filename =  "SummedDifferenceComp_Egun_Edep_CVAE_Epoch" + str(len(epochs)) + ".png"
    fig = plt.figure(figsize=(5,5))
    plt.scatter(epochs, sum_diffs, alpha = 0.8)
    #plt.scatter(epochs, gen_FWHMs, alpha = 0.8,  label='VAE')
    plt.xlabel("Epoch", fontsize = 12.5)
    plt.ylabel("Difference", fontsize = 12.5)
    #plt.ylim(0,50)
    plt.xlim(0,n_epochs)
    plt.title(r" Avg Summed $E_{dep}$ diff", fontsize=11)
   # plt.legend(loc='best')
    if save_dir != None: 
        plt.savefig(save_dir + filename)
    plt.show()

    return 



def plot_FWHMs(gen_FWHMs, real_FWHMs, epochs, n_epochs, save_dir = None):



    print(type(bs))
    FWHM_comp =  "FWHMComp_Egun_Edep_CVAE_Epoch" + str(len(epochs)) + ".png"
    fig = plt.figure(figsize=(5,5))
    plt.scatter(epochs, real_FWHMs, alpha = 0.8, label='Geant4')
    plt.scatter(epochs, gen_FWHMs, alpha = 0.8,  label='CVAE')
    plt.xlabel("Epoch", fontsize = 12.5)
    plt.ylabel("FWHM", fontsize = 12.5)
    #plt.ylim(0,50)
    plt.xlim(0,n_epochs)
    plt.title(r" FWHM  for $E_{dep}$ diff", fontsize=12)
    plt.legend(loc='best')
    if save_dir != None:
        
        plt.savefig(save_dir + FWHM_comp)
    plt.show()

    return 

def hist_width(hist_data):

    mean = hist_data.mean()
    std = hist_data.std()
    FWHM = 2.*np.sqrt(2.*np.log(2.))*std
    #line2 = FWHM - (FHWM / 2)
    #line1 = line2 - FWHM

    return FWHM

def plot_all_metrics(gen_FWHMs, g4_FWHMs, gen_means, g4_means, epochs, n_epochs, save_dir =None):

#     if gamma == 0.1:
#         gamma_val = str('01')
#     elif gamma == 0.5:
#         gamma_val = str('05')
#     elif gamma == 1.0:
#         gamma_val = str('1')

    #gen_means = np.array(gen_means)
    #gen_means = gen_means
    #gen_kurts = gen_means[:,1]
    #g4 = np.array(g4_means)
    #g4_means = g4_means
    #g4_kurts = g4_means[:,1]

    #if Kurts == True:
    #    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,4), sharey = False, sharex = True)

    #else:

    #print(type(gen_FWHMs), type)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(9,4), sharey = False, sharex = True)

    ax1.scatter(epochs, g4_FWHMs, alpha = 0.75, label = 'geant4');
    ax1.scatter(epochs, gen_FWHMs, alpha = 0.70, label = 'CVAE');
    ax1.set_title("FWHMs")
    if len(epochs) < n_epochs/2:
        ax1.set_xlim(0,n_epochs /2)
    else:
        ax1.set_xlim(0,n_epochs)
    fig.suptitle(r" Distribution Metrics for $E_{dep}$", x=0.5, y = 1.02, fontsize = 13) 
    #plt.xlabel(r"Difference in $P_{T}$ compared to Ground Truth")
    #ax1.set_xlabel(r"$P_{T} - P_{T}^{sim} $ (MeV)", fontsize = 16)
    #ax1.set_ylabel("FWHM", fontsize = 14)
    #ax1.set_ylim(0,9000)

    ax2.scatter(epochs, g4_means, alpha = 0.75, label = 'geant4');
    ax2.scatter(epochs, gen_means, alpha = 0.70, label = 'CVAE');
    ax2.set_title("Means")
    if len(epochs) < n_epochs/2:
        ax2.set_xlim(0,n_epochs /2)
    else:
        ax2.set_xlim(0,n_epochs)

#     if Kurts == True:
#         ax3.scatter(epochs, delphes_kurts,alpha = 0.75, label = 'geant4');
#         ax3.scatter(epochs, gen_kurts, alpha = 0.70, label = 'VAE');
#         ax3.set_title("Kurtoses")
#         ax3.set_ylim(0,5)

    fig.text(0.5, -0.01, "Epoch", ha = 'center', fontsize = 15)
    ax1.legend(loc='best')
    #if Kurts == True:
    #    ax1.legend(bbox_to_anchor=[3.94, 0.5], loc='center right')
    #else:
    ax1.legend(bbox_to_anchor=[2.64, 0.5], loc='center right')


    if save_dir != None:
        FWHMs_Means_comp = "FHWMsMeans_Edep_EGunCVAE_Epoch" + str(len(epochs)) + ".png"

        #file_path = "/home/chris/Documents/MPhilProjects/ForViewing/plots/eGunPTSmearing/FWHMsMeans/"
        plt.savefig(save_dir + FWHMs_Means_comp, bbox_inches="tight")

    plt.show()

    return



def plot_losses(train_losses, test_losses, epochs , n_epochs, save_dir = None):



    print(type(bs))
    FWHM_comp =  "LossesComp_Egun_Edep_CVAE_Epoch" + str(len(epochs)) + ".png"
    fig = plt.figure(figsize=(6,4))
    plt.scatter(epochs, train_losses, alpha = 0.8, label='train')
    plt.scatter(epochs, test_losses, alpha = 0.8,  label='test')
    plt.xlabel("Epoch", fontsize = 12.5)
    plt.ylabel("Loss", fontsize = 12.5)
    #plt.ylim(0,50)
    plt.xlim(0,n_epochs)
    plt.title(r" Losses  for learning $e^{-}$ gun $E_{dep}$", fontsize=11)
    plt.legend(loc='best')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    if save_dir != None:
        
        plt.savefig(save_dir + FWHM_comp)
    plt.show()

    return 

