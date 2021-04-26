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
titleFontSize = 16
defaultFontSize = 10
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

    # Set figure size in pixels
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(1920.0/float(DPI),1280.0/float(DPI))

def plotMap(t_N, endTime, simX, simY, simZ, estMapX, estMapY, estMapZ, linPosX, linPosY, linPosZ):
    # Plot settings    
    plt.figure(2)
    plt.gcf().canvas.mpl_connect('key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None]) # press esc key to exit figure
    tt = np.linspace(0, endTime, 48001, endpoint=True)
    ttDvl = np.linspace(0, endTime, 48001, endpoint=True)
    ttImu = np.linspace(0, endTime, 48001, endpoint=True)
    ttBar = np.linspace(0, endTime, 48001, endpoint=True)
    ttEst = np.linspace(0, endTime, 48001, endpoint=True)

    xm = plt.subplot(331)
    plt.plot(tt[:-1], simX, color = 'k', linestyle = '--', linewidth = 1, markersize = markerSize/3, label = 'simulated ground truth')
    plt.plot(ttDvl[:-1], linPosX, color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize/2, fillstyle = 'none', label = 'sensor measurement')
    plt.plot(ttEst[:-1], estMapX, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize/2, label = 'estimated trajectory')
    plt.title(r'Map Position',fontsize = defaultFontSize)
    plt.xlabel(r'time [s]',fontsize = defaultFontSize)
    plt.ylabel(r'$x_m$ [m]',fontsize = defaultFontSize)

    ym = plt.subplot(332)
    plt.plot(tt[:-1], simY, color = 'k', linestyle = '--', linewidth = 1, markersize = markerSize/2, label = 'simulated ground truth')
    plt.plot(ttDvl[:-1], linPosY, color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize/2, fillstyle = 'none', label = 'sensor measurement')
    plt.plot(ttEst[:-1], estMapY, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize/2, label = 'estimated trajectory')
    plt.title(r'Map Position',fontsize = defaultFontSize)    
    plt.xlabel(r'time [s]',fontsize = defaultFontSize)
    plt.ylabel(r'$y_m$ [m]',fontsize = defaultFontSize)

    zm = plt.subplot(333)
    plt.plot(tt[:-1], simZ, color = 'k', linestyle = '--', linewidth = 1, markersize = markerSize/2, label = 'simulated ground truth')
    plt.plot(ttBar[:-1], linPosZ, color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize/2, fillstyle = 'none', label = 'sensor measurement')
    plt.plot(ttEst[:-1], estMapZ, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize/2, label = 'estimated trajectory')
    plt.title(r'Map Position',fontsize = defaultFontSize)
    plt.xlabel(r'time [s]',fontsize = defaultFontSize)
    plt.ylabel(r'$z_m$ [m]',fontsize = defaultFontSize)

    plt.grid(True)
    plt.axis('equal')
    # plt.legend()

    # Set figure size in pixels
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(1920.0/float(DPI),1280.0/float(DPI))

def plotRbt(t_N, endTime, simRoll, simPitch, simYaw, simVelX, simVelY, simVelZ, linVelXRbt, linVelYRbt, linVelZRbt, roll, pitch, yaw, acBiasX, acBiasY, acBiasZ, estAcBiasX, estAcBiasY, estAcBiasZ, estVelXRbt, estVelYRbt, estVelZRbt):
    # Plot settings
    plt.figure(3)
    plt.gcf().canvas.mpl_connect('key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None]) # press esc key to exit figure
    tt = np.linspace(0, endTime, 48001, endpoint=True)
    ttDvl = np.linspace(0, endTime, 48001, endpoint=True)
    ttImu = np.linspace(0, endTime, 48001, endpoint=True)
    ttBar = np.linspace(0, endTime, 48001, endpoint=True)
    ttEst = np.linspace(0, endTime, 48001, endpoint=True)

    # Robot frame plot rotation states
    rollm = plt.subplot(331)
    plt.subplots_adjust(hspace = 1)
    plt.plot(tt[:-1], np.rad2deg(simRoll), color = 'k', linestyle = '--', linewidth = 1, markersize = markerSize, label = 'simulated ground truth')
    plt.plot(ttImu[:-1], np.rad2deg(roll), color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize/5, label = 'sensor measurement')
    # plt.plot(ttEst[:-1], estMapZ, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize, label = 'estimated trajectory')
    plt.title(r'Robot body rotation',fontsize = defaultFontSize)
    plt.xlabel(r'time [s]',fontsize = defaultFontSize)
    plt.ylabel(r'$roll_m$ [deg]',fontsize = defaultFontSize)

    pitchm = plt.subplot(332)
    plt.plot(tt[:-1], np.rad2deg(simPitch), color = 'k', linestyle = '--', linewidth = 1, markersize = markerSize, label = 'simulated ground truth')
    plt.plot(ttImu[:-1], np.rad2deg(pitch), color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize/5, label = 'sensor measurement')
    # plt.plot(ttEst[:-1], estMapZ, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize, label = 'estimated trajectory')
    plt.title(r'Robot body rotation',fontsize = defaultFontSize)
    plt.xlabel(r'time [s]',fontsize = defaultFontSize)
    plt.ylabel(r'$pitch_m$ [deg]',fontsize = defaultFontSize)

    yawm = plt.subplot(333)
    plt.plot(tt[:-1], np.rad2deg(simYaw), color = 'k', linestyle = '--', linewidth = 1, markersize = markerSize, label = 'simulated ground truth')
    plt.plot(ttImu[:-1], np.rad2deg(yaw), color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize/5, label = 'sensor measurement')
    # plt.plot(ttEst[:-1], estMapZ, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize, label = 'estimated trajectory')
    plt.title(r'Robot body rotation',fontsize = defaultFontSize)
    plt.xlabel(r'time [s]',fontsize = defaultFontSize)
    plt.ylabel(r'$yaw_m$ [deg]',fontsize = defaultFontSize)

    # Robot frame plot velocity states
    velxr = plt.subplot(334)
    plt.plot(tt[:-1], simVelX, color = 'k', linestyle = '--', linewidth = 1, markersize = markerSize, label = 'simulated ground truth')
    plt.plot(ttDvl[:-1], linVelXRbt, color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize/5, label = 'sensor measurement')
    plt.plot(ttEst[:-1], estVelXRbt, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize, label = 'estimated trajectory')
    plt.title(r'Robot body velocity',fontsize = defaultFontSize)
    plt.xlabel(r'time [s]',fontsize = defaultFontSize)
    plt.ylabel(r'$velocity_x$ [m/s]',fontsize = defaultFontSize)

    velyr = plt.subplot(335)
    plt.plot(tt[:-1], simVelY, color = 'k', linestyle = '--', linewidth = 1, markersize = markerSize, label = 'simulated ground truth')
    plt.plot(ttDvl[:-1], linVelYRbt, color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize/5, label = 'sensor measurement')
    plt.plot(ttEst[:-1], estVelYRbt, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize, label = 'estimated trajectory')
    plt.title(r'Robot body velocity',fontsize = defaultFontSize)
    plt.xlabel(r'time [s]',fontsize = defaultFontSize)
    plt.ylabel(r'$velocity_y$ [m/s]',fontsize = defaultFontSize)

    velzr = plt.subplot(336)
    plt.plot(tt[:-1], simVelZ, color = 'k', linestyle = '--', linewidth = 1, markersize = markerSize, label = 'simulated ground truth')
    plt.plot(ttDvl[:-1], linVelZRbt, color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize/5, label = 'sensor measurement')
    plt.plot(ttEst[:-1], estVelZRbt, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize, label = 'estimated trajectory')
    plt.title(r'Robot body velocity',fontsize = defaultFontSize)
    plt.xlabel(r'time [s]',fontsize = defaultFontSize)
    plt.ylabel(r'$velocity_z$ [m/s]',fontsize = defaultFontSize)

    # Acceleration bias
    accbiasxm = plt.subplot(337)
    plt.plot(ttImu[:-1], acBiasX, color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize/2, fillstyle = 'none', label = 'sensor measurement')
    plt.plot(ttEst[:-1], estAcBiasX, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize/2, label = 'estimated trajectory')
    plt.title(r'Robot body accel bias',fontsize = defaultFontSize)
    plt.xlabel(r'time [s]',fontsize = defaultFontSize)
    plt.ylabel(r'$bias_x$ [$m/s^2$]',fontsize = defaultFontSize)

    accbiasym = plt.subplot(338)
    plt.plot(ttImu[:-1], acBiasY, color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize/2, fillstyle = 'none', label = 'sensor measurement')
    plt.plot(ttEst[:-1], estAcBiasY, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize/2, label = 'estimated trajectory')
    plt.title(r'Robot body accel bias',fontsize = defaultFontSize)    
    plt.xlabel(r'time [s]',fontsize = defaultFontSize)
    plt.ylabel(r'$bias_y$ [$m/s^2$]',fontsize = defaultFontSize)

    accbiaszm = plt.subplot(339)
    plt.plot(ttImu[:-1], acBiasZ, color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize/2, fillstyle = 'none', label = 'sensor measurement')
    plt.plot(ttEst[:-1], estAcBiasZ, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize/2, label = 'estimated trajectory')
    plt.title(r'Robot body accel bias',fontsize = defaultFontSize)
    plt.xlabel(r'time [s]',fontsize = defaultFontSize)
    plt.ylabel(r'$bias_z$ [$m/s^2$]',fontsize = defaultFontSize)

    plt.grid(True)
    plt.axis('equal')
    # legend = plt.legend()
    # legend.set_draggable(state=True)