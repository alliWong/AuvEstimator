#! /usr/bin/env python

from numpy.linalg import inv
from numpy import identity
from numpy import matrix, diag, random, array, zeros, block, mod, reshape
import math
import matplotlib.pyplot as plt
import numpy as np

""" Plots """
###################### Robot X and Y ######################
def plotXY(simX, simY, linPosX, linPosY, estMapX, estMapY):  
    # Set fonts properties
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    titleFontSize = 20
    defaultFontSize = 12
    markerSize = 3
    trialTitleString = r'Map Frame \textit{XY} Position'

    # Plot settings
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
        
    # Set figure size in pixels
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(1920.0/float(DPI),1280.0/float(DPI))
    plt.show()