#! /usr/bin/env python

from numpy.linalg import inv
from numpy import identity
from numpy import matrix, diag, random, array, zeros, block, mod, reshape
import math
import matplotlib.pyplot as plt
import numpy as np

""" Plots """
# Set fonts properties
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
titleFontSize = 20
defaultFontSize = 12
markerSize = 3

###################### Robot X and Y ######################
def plotXY(simX, simY, linPosX, linPosY, estMapX, estMapY):  
    trialTitleString = r'Map Frame \textit{XY} Position'

    # Plot settings
    plt.figure(1)
    plt.gcf().canvas.mpl_connect('key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None]) # press esc key to exit figure
    plt.plot(simX, simY, color = 'k', linestyle = '--', linewidth = 1, markersize = markerSize, label = 'simulated ground truth')
    plt.plot(linPosX, linPosY, color = 'r', linestyle = '-', marker = 'o', fillstyle = 'none', linewidth = 1, markersize = markerSize, label = 'simulated measurement')    
    plt.plot(estMapX, estMapY, color = 'b', linestyle = '--', marker = '.', linewidth = 1, markersize = markerSize, label = 'estimator output')
    plt.title(trialTitleString,fontsize = titleFontSize)
    plt.xlabel(r'$x_m$ [m]',fontsize = defaultFontSize)
    plt.ylabel(r'$y_m$ [m]',fontsize = defaultFontSize)
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
        
    # # Set figure size in pixels
    # fig = plt.gcf()
    # DPI = fig.get_dpi()
    # fig.set_size_inches(1920.0/float(DPI),1280.0/float(DPI))

def plotMap(t_N, endTime, simX, simY, simZ, estMapX, estMapY, estMapZ, linPosX, linPosY):
    plt.figure(2)
    plt.gcf().canvas.mpl_connect('key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None]) # press esc key to exit figure

    tt = np.linspace(0, endTime, 48001, endpoint=True)
    ttDvl = np.linspace(0, endTime, 3, endpoint=True)
    ttImu = np.linspace(0, endTime, 1000, endpoint=True)
    # ttBar = np.linspace(0, endTime, 3, endpoint=True)
    ttEst = np.linspace(0, endTime, 48001, endpoint=True)

    xm = plt.subplot(331)
    plt.plot(tt[:-1], simX, color = 'k', linestyle = '--', linewidth = 1, markersize = markerSize, label = 'simulated ground truth')
    # plt.plot(ttDvl[:-1], linPosX, color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize, label = 'sensor measurement')
    plt.plot(ttEst[:-1], estMapX, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize, label = 'estimated trajectory')
    plt.title(r'Map Position',fontsize = defaultFontSize)
    plt.xlabel(r'time [s]',fontsize = defaultFontSize)
    plt.ylabel(r'$x_m$ [m]',fontsize = defaultFontSize)

    ym = plt.subplot(332)
    plt.plot(tt[:-1], simY, color = 'k', linestyle = '--', linewidth = 1, markersize = markerSize, label = 'simulated ground truth')
    # plt.plot(ttDvl[:-1], linPosY, color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize, label = 'sensor measurement')
    plt.plot(ttEst[:-1], estMapY, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize, label = 'estimated trajectory')
    plt.title(r'Map Position',fontsize = defaultFontSize)    
    plt.xlabel(r'time [s]',fontsize = defaultFontSize)
    plt.ylabel(r'$y_m$ [m]',fontsize = defaultFontSize)

    zm = plt.subplot(333)
    plt.plot(tt[:-1], simZ, color = 'k', linestyle = '--', linewidth = 1, markersize = markerSize, label = 'simulated ground truth')
    # plt.plot(ttDvl[:-1], linPosY, color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize, label = 'sensor measurement')
    plt.plot(ttEst[:-1], estMapZ, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize, label = 'estimated trajectory')
    plt.title(r'Map Position',fontsize = defaultFontSize)
    plt.xlabel(r'time [s]',fontsize = defaultFontSize)
    plt.ylabel(r'$z_m$ [m]',fontsize = defaultFontSize)

# def plotRbt():



