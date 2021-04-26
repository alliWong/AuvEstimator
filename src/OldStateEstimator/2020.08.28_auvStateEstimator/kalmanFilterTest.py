#! /usr/bin/env python

"""
TASKS:
1) Generate estimated trajectory
2) Change sensor type
3) Create 3D Kalman Filter
4) Create 3D EKF
"""

from numpy.linalg import inv
from numpy import identity
from numpy import matrix, diag, random, array, zeros, block, mod, reshape
import math
import matplotlib.pyplot as plt
import numpy as np

""" Simulation User Inputs """
# USER INPUT: time properties
dt = 0.001              # simulation time step [s]
startTime = 0           # simulation start time [s]
endTime = 60            # simulation end time [s]

# USER INPUT: underwater vehicle lumped parameters
sim_m = 225     # mass/inertia [kg]
sim_I = 100     # rotational moment of inertia [kg*m^2]
sim_bxr = 40    # drag in the surge (robot frame along x) direction [N*s/m]
sim_byr = 400   # drag in the sway (robot frame along y) direction [N*s/m]
sim_bpr = 300   # rotational drag in the yaw (robot frame about z) direction [N*s/m]

# USER INPUT: sensor setup
gnss_rr = 1             # refresh rate of GPS sensor [Hz]
gnss_linPosVar = 1      # variance of the GNSS sensor data [m]
imu_rr = 20             # refresh rate of IMU sensor [Hz]
imu_angPosVar = 0.05    # variance of the IMU angular position sensor data [rad]
imu_angVelVar = 0.05    # variance of the IMU angular velocity sensor data [rad/s]
imu_linAccVar = 0.1     # variance of the IMU linear acceleration sensor data [m/s^2]

""" Time Properties """
t_N = round(endTime/dt) # total number of time steps [s]
t_now = startTime       # current time [s]

""" Simulation Setup """
# State, output, and input vectors
sim_xr = matrix(zeros((6, t_N)))    # simulation state vector in robot reference frame
sim_yr = matrix(zeros((9, t_N)))    # simulation output vector in robot reference frame
sim_xm = matrix(zeros((3, t_N)))    # simulation state vector in map reference frame
sim_u = matrix(zeros((3, t_N)))     # input vector (control inputs act on robot frame)

# Create simulation state matrices
sim_A = matrix([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, -sim_bxr/sim_m, 0, 0],
    [0, 0, 0, 0, -sim_byr/sim_m, 0],
    [0, 0, 0, 0, 0, -sim_bpr/sim_I]])   # state matrix [6x6]
sim_B = matrix([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1/sim_m, 0, 0],
    [0, 1/sim_m, 0],
    [0, 0, 1/sim_I]])   # input matrix [6x3]
sim_C = matrix(block([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [sim_A[3:6, :], zeros((3,3))]]))   # output matrix [9x9]
sim_D = matrix(block([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],  
    [sim_B[3:6, :]]]))   # feedthrough matrix [9x3] 

""" Sensor Setup """
gnss_linPos = matrix(zeros((2, endTime*gnss_rr)))   # linear position data from GNSS
gnss_linPosRbt = matrix(zeros((2,endTime*gnss_rr))) # linear position data from GNSS, converted to the robot frame
gnss_angPos = matrix(zeros((2,endTime*gnss_rr)))    # angular position data from IMU, references to GNSS timestamp
imu_angPos = matrix(zeros((1,endTime*imu_rr)))  # angular position data from IMU
imu_angVel = matrix(zeros((1,endTime*imu_rr)))  # angular velocity data from IMU
imu_linAcc = matrix(zeros((2,endTime*imu_rr)))  # linear acceleration data from IMU

""" Reset Sensor Tracker """
# Sensor update tracker
gnss_update = 0 # gnss update tracker (boolean)
imu_update = 0  # imu update tracker (boolean)
gnss_k = 1  # gnss iteration tracker
imu_k = 1   # imu iteration tracker
gnss_lastUpdateTime = 0 # gnss last update time tracker [s]
imu_lastUpdateTime = 0  # imu last update time tracker [s]

""" Frame Transformations """
def rbt2map(xrf,yrf,xr0,yr0,psi0,xm0,ym0):
    # Converts pose in the robot frame to pose in the map frame

    # Calculate translations and rotations in robot frame
    Txr = xrf-xr0
    Tyr = yrf-yr0

    # Calculate intermediate length and angle
    li = math.sqrt(Txr**2+Tyr**2)
    psii = math.atan2(yrf-yr0, xrf-xr0)

    # Calculate translations and rotations in map frame
    Txm = math.cos(psii+psi0)*li
    Tym = math.sin(psii+psi0)*li

    # Calculate individual components in the map frame
    xmf = xm0+Txm
    ymf = ym0+Tym

    return xmf, ymf

def map2rbt(xmf,ymf,xm0,ym0,psi0,xr0,yr0):
    # Converts pose in the map frame to pose in the robot frame

    # Calculate translations and rotations in robot frame
    Txm = xmf-xm0
    Tym = ymf-ym0

    # Calculate intermediate length and angle
    li = math.sqrt(Txm**2+Tym**2)
    psii = math.atan2(ymf-ym0, xmf-xm0)
    
    # Calculate translations and rotations in robot frame
    Txr = math.cos(psii-psi0)*li
    Tyr = math.sin(psii-psi0)*li

    # Calculate individual components in the robot frame
    xrf = xr0+Txr
    yrf = yr0+Tyr

    return xrf, yrf

""" Controllers """
def controller():
    if (k == 0):
        sim_u[0,k] = 0     # force input in surge direction
        sim_u[1,k] = 0     # force input in sway direction
        sim_u[2,k] = 0     # torque input in yaw direction
    elif (k > 1 and k < 15000):
        sim_u[0,k] = 300   # force input in surge direction
        sim_u[1,k] = 150     # force input in sway direction
        sim_u[2,k] = 50   # torque input in yaw direction
    elif (k > 1 and k < 30000):
        sim_u[0,k] = -300   # force input in surge direction
        sim_u[1,k] = -150     # force input in sway direction
        sim_u[2,k] = -50   # torque input in yaw direction
    elif (k > 1 and k < 45000):
        sim_u[0,k] = 300  # force input in surge direction
        sim_u[1,k] = 150     # force input in sway direction
        sim_u[2,k] = 50   # torque input in yaw direction
    else:
        sim_u[0,k] = -300     # force input in surge direction
        sim_u[1,k] = 150     # force input in sway direction
        sim_u[2,k] = -50     # torque input in yaw direction

""" Run simulation """
def runSim():
    # Create discrete state matrices for Euler integration
    F = identity(6)+sim_A*dt    # discrete state matrix
    G = sim_B*dt    # discrete input matrix
    H = sim_C   # discrete output matrix
    J = sim_D   # discrete feedthrough matrix 

    if k != t_N:
        # Simulate plant using discrete Euler integration
        sim_xr[:, k+1] = F*sim_xr[:, k]+G*matrix([[sim_u[0, k]], [sim_u[1, k]], [sim_u[2, k]]]) #  state matrix integration solution 
        sim_yr[:, k+1] = H*block([[sim_xr[:, k+1]], [zeros((3,1))]])+J*matrix([[sim_u[0, k]], [sim_u[1, k]], [sim_u[2, k]]])    # state observer matrix integration solution

        # Convert pose in robot frame to pose in map frame
        [sim_xm[0,k+1], sim_xm[1, k+1]] = rbt2map(sim_xr[0, k+1], sim_xr[1, k+1], sim_xr[0, k], sim_xr[1, k], sim_xr[2, k], sim_xm[0, k], sim_xm[1, k])
        sim_xm[2, k+1] = sim_xr[2, k+1]

""" Estimater User Inputs """
# USER INPUT: time step properties
est_dt = 0.025  # estimator time step [s]

# USER INPUT: Number of past sensor updates to store for calculating the incremental moving mean of sensor data
est_linPos_n = 2    # linear position mean array size (must be 2 or more)
est_angPos_n = 2    # angular position mean array size (must be 2 or more)
est_angVel_n = 2    # angular velocity mean array size (must be 2 or more)
est_linAcc_n = 2    # linear acceleration mean array size (must be 2 or more)

# USER INPUT: decide which sensors to use for pseudo linear velcotiy calculation
est_useGnssLinVel = 1   # set to 1 to use GNSS forward difference, set to 0 to ignore
est_useImuLinVel = 1    # set to 1 to use IMU trapezoidal method, set to 0 to ignore
est_linVelBias = 1.35   # bias (multiplier) on linear velocity output

# USER INPUT: robot frame estimator measurement noise covariance
est_rbt_R_linPos = 2       # linear position [m]
est_rbt_R_angPos = 0.075   # angular position [rad]
est_rbt_R_linVel = 2       # linear velocity [m/s]
est_rbt_R_angVel = 0.075   # angular velocity [rad/s]
est_rbt_R_linAcc = 0.25    # linear acceleration [m/s^s]
est_rbt_R_angAcc = 1       # angular acceleration [rad/s^2] (not measured, so this is actually irrelevant)

# USER INPUT: robot frame estimator process noise covariance
est_rbt_Q_linPos = est_rbt_R_linPos*1e1    # linear position [m]
est_rbt_Q_angPos = est_rbt_R_angPos*1e1    # angular position [rad]
est_rbt_Q_linVel = est_rbt_R_linVel*1e1    # linear velocity [m/s]
est_rbt_Q_angVel = est_rbt_R_angVel*1e-1   # angular velocity [rad/s]
est_rbt_Q_linAcc = est_rbt_R_linAcc*1e-1   # linear acceleration [m/s^s]
est_rbt_Q_angAcc = est_rbt_R_angAcc*1e-1   # angular acceleration [rad/s^2] (not measured, so this is actually irrelevant)

# USER INPUT: robot frame estimator observation/observability
est_H_linPos = 1       # linear position (measured)
est_H_angPos = 1       # angular position (measured)
est_H_linVel = 1       # linear velocity (pseudo-measured)
est_H_angVel = 1       # angular velocity (measured)
est_H_linAcc = 1       # linear acceleration (measured)
est_H_angAcc = 0       # angular acceleration (NOT measured)
            
# USER INPUT: map frame estimator measurement noise covariance
est_map_R_linPos = 1000       # linear position [m]
est_map_R_angPos = 0.075   # angular position [rad]
est_map_R_linVel = 1       # linear velocity [m/s]
est_map_R_angVel = 0.075   # angular velocity [rad/s]
            
# USER INPUT: map frame estimator process noise covariance          
est_map_Q_linPos = est_map_R_linPos*1e1    # linear position [m]
est_map_Q_angPos = est_map_R_angPos*1e5    # angular position [rad]
est_map_Q_linVel = est_map_R_linVel*1e5    # linear velocity [m/s]
est_map_Q_angVel = est_map_R_angVel*1e5    # angular velocity [rad/s]
            
# USER INPUT: map frame estimator observation/observability
est_H_linPos = 1       # linear position (measured)
est_H_angPos = 1       # angular position (measured)
est_H_linVel = 1       # linear velocity (pseudo-measured)
est_H_angVel = 1       # angular velocity (measured)      

""" Estimator Setup """
# Instantiate measurement variables and arrays
est_k = 1                                      # iteration tracking variable
est_gnssLastUpdateTime = 0                     # last GNSS update time [s]
est_imuLastUpdateTime = 0                      # last IMU update time [s]
est_linPosMapArray = matrix(zeros((2,est_linPos_n)))     # linear position directly from GNSS in map frame measurement array
est_linPosRbtArray = matrix(zeros((2,est_linPos_n)))     # linear position directly from GNSS in robot frame measurement array
est_angPosArray = matrix(zeros((1,est_angPos_n)))        # angular position directly from IMU in map and robot frame measurement array
est_linVelGnssAprxRbt = matrix(zeros((2,1)))             # approximated linear velocity in robot frame from forward-difference of GNSS
est_linVelImuAprxRbt = matrix(zeros((2,1)))              # approximated linear velocity in robot frame from trapezoidal method of IMU
est_angVelArray = matrix(zeros((1,est_angVel_n)))        # angular velocity directly from IMU in map and robot frame measurement array
est_linAccRbtArray = matrix(zeros((2,est_linAcc_n)))     # linear acceleration directly from IMU in robot frame measurement array
            
# Instantiate estimator matrices
est_rbt_m = matrix(zeros((9,int(endTime/est_dt))))              # robot frame estimator measurement matrix
est_rbt_x = matrix(zeros((9,int(endTime/est_dt))))              # robot frame estimator state estimate
est_rbt_L = array(zeros((9,9,int(endTime/est_dt))))            # robot frame estimator kalman gain matrix
est_rbt_P = array(zeros((9,9,int(endTime/est_dt))))            # robot frame estimator covariance matrix
est_rbt_s = array(zeros((9,9,int(endTime/est_dt))))            # robot frame sliding covariance gain matrix            
est_map_m = matrix(zeros((6,int(endTime/est_dt))))              # map frame estimator measurement matrix
est_map_x = matrix(zeros((6,int(endTime/est_dt))))              # map frame estimator state estimate
est_map_L = array(zeros((6,6,int(endTime/est_dt))))            # map frame estimator kalman gain matrix
est_map_P = array(zeros((6,6,int(endTime/est_dt))))            # map frame estimator covariance matrix


# Robot frame covariance, obvervation, state and input matrices
est_rbt_Q = matrix([
    [est_rbt_Q_linPos, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, est_rbt_Q_linPos, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, est_rbt_Q_angPos, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, est_rbt_Q_linVel, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, est_rbt_Q_linVel, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, est_rbt_Q_angVel, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, est_rbt_Q_linAcc, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, est_rbt_Q_linAcc, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, est_rbt_Q_angAcc]])      # process noise covariance matrix    
est_rbt_R = matrix([
    [est_rbt_R_linPos, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, est_rbt_R_linPos, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, est_rbt_R_angPos, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, est_rbt_R_linVel, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, est_rbt_R_linVel, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, est_rbt_R_angVel, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, est_rbt_R_linAcc, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, est_rbt_R_linAcc, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, est_rbt_R_angAcc]])      # measurement noise covariance matrix
est_rbt_H = matrix([
    [est_H_linPos, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, est_H_linPos, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, est_H_angPos, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, est_H_linVel, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, est_H_linVel, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, est_H_angVel, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, est_H_linAcc, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, est_H_linAcc, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, est_H_angAcc]])       # obervation matrix
est_rbt_A = matrix([
    [1, 0, 0, est_dt, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, est_dt, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, est_dt, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]])                  # state matrix
est_rbt_B = matrix(zeros((9,3)))                   # input matrix            
            
# Map frame covariance, obvervation, state and input matrices
est_map_Q = matrix([
    [est_map_Q_linPos, 0, 0, 0, 0, 0],
    [0, est_map_Q_linPos, 0, 0, 0, 0],
    [0, 0, 1e5, 0, 0, 0],
    [0, 0, 0, 1e5, 0, 0],
    [0, 0, 0, 0, 1e5, 0],
    [0, 0, 0, 0, 0, 1e5]])                 # process noise covariance matrix    
est_map_R = matrix([
    [est_map_R_linPos, 0, 0, 0, 0, 0],
    [0, est_map_R_linPos, 0, 0, 0, 0],
    [0, 0, est_map_R_angPos, 0, 0, 0],
    [0, 0, 0, est_map_R_linVel, 0, 0],
    [0, 0, 0, 0, est_map_R_linVel, 0],
    [0, 0, 0, 0, 0, est_map_R_angVel]])    # measurement noise covariance matrix
est_map_H = matrix([
    [est_H_linPos, 0, 0, 0, 0, 0],
    [0, est_H_linPos, 0, 0, 0, 0],
    [0, 0, est_H_angPos, 0, 0, 0], 
    [0, 0, 0, est_H_linVel, 0, 0],
    [0, 0, 0, 0, est_H_linVel, 0],
    [0, 0, 0, 0, 0, est_H_angVel]])        # obervation matrix 
est_map_A = lambda x: matrix([
    [1, 0, 0, math.cos(x[2])*est_dt, -math.sin(x[2])*est_dt, 0],
    [0, 1, 0, math.sin(x[2])*est_dt, math.cos(x[2])*est_dt, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]])                   # state matrix
est_map_B = matrix(zeros((6,3)))           # input matrix

""" Kalman Filter """
def KalmanFilter(x, P, z, u, A, B, Q, R, H):

    # Prediction for state vector and covariance
    x = A * x + B * u
    P = A * P * (A.T) + Q

    # Compute Kalman Gain
    K = P * (H.T) * inv(H * P * (H.T) + R)

    # Correction based on observation
    x += K * (z - H * x)
    P = P - K * H * P

    return x, P, z, u, A, B, Q, R, H

""" Plots """
def plotXY():  
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ###################### Robot X and Y ######################
    # Set fonts properties
    titleFontSize = 32
    defaultFontSize = 12
    markerSize = 10
    trialTitleString = r'Map Frame \textit{XY} Position'

    # Plot settings
    plt.gcf().canvas.mpl_connect('key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None]) # press esc key to exit figure
    plt.plot(simX, simY, color = 'k', linestyle = '--', linewidth = 1, markersize = markerSize, label = 'simulated ground truth')
    plt.plot(linPosX, linPosY, color = 'r', linestyle = '-', marker = 'o', fillstyle = 'none', linewidth = 1, markersize = markerSize*0.25, label = 'simulated measurement')    
    plt.plot(estMapX, estMapY, color = 'b', linestyle = '--', marker = '.', linewidth = 1, markersize = markerSize)
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

""" Store data """
# Store data for plotting
simX = []
simY = []
linPosX = []
linPosY = []
estMapX = []
estMapY = []

"""  RUN LOOP """
for k in range (0, t_N, 1):
    if k < 58000:

        """ Simulation """
        # Controller
        controller()

        # Simulate robot 
        runSim()

        ### Check sensor update ###
        if k != t_N:
            # Increment GNSS and set update flag to true if GNSS update should occur
            if mod(k*dt, 1/gnss_rr) == 0:   # update sensor reading at rr
                gnss_k = gnss_k+1   # increment sensor
                gnss_update = 1     # set update flag to true
                gnss_lastUpdatetime = t_now # set the last time the sensor updated
            else:
                gnss_update = 0     # set update flag to false
                
            #  Increment IMU and set update flag to true if IMU update should occur
            if mod(k*dt, 1/imu_rr) == 0:    # update sensor reading at rr    
                imu_k = imu_k+1     # increment sensor
                imu_update = 1      # set update flag to true
                imu_lastUpdatetime = t_now  # set the last time the sensor updated
            else:
                imu_update = 0  # set update flag to false
            
        ### Update sensors ### 
        if k != t_N:          
            # Add noise to GNSS states to simulate sensor data if GNSS update should occur
            if gnss_update == 1:
                gnss_linPos[0, gnss_k] = sim_xm[0, k]+np.random.rand(1)*math.sqrt(gnss_linPosVar)
                gnss_linPos[1, gnss_k] = sim_xm[1, k]+np.random.rand(1)*math.sqrt(gnss_linPosVar)
                gnss_angPos[0, gnss_k] = imu_angPos[0, imu_k-1]
            
            # Add noise to IMU states to simulate sensor data if IMU update should occur
            if imu_update == 1:
                imu_angPos[0, imu_k] = sim_yr[2, k]+np.random.rand(1)*math.sqrt(imu_angPosVar)
                imu_angVel[0, imu_k] = sim_yr[5, k]+np.random.rand(1)*math.sqrt(imu_angVelVar)
                imu_linAcc[0, imu_k] = sim_yr[6, k]+np.random.rand(1)*math.sqrt(imu_linAccVar)
                imu_linAcc[1, imu_k] = sim_yr[7, k]+np.random.rand(1)*math.sqrt(imu_linAccVar)
        
        # Convert GNSS to robot frame (this MUST occur after previous steps)
        if gnss_update == 1:

            if gnss_k != 1:

                # Convert GNSS pose in map frame ot pose in true robot frame
                [xrf, yrf] = map2rbt(
                    gnss_linPos[0, gnss_k],
                    gnss_linPos[1, gnss_k],
                    gnss_linPos[0, gnss_k-1],
                    gnss_linPos[1, gnss_k-1],
                    gnss_angPos[0, gnss_k],
                    gnss_linPosRbt[0, gnss_k-1],
                    gnss_linPosRbt[1, gnss_k-1]) 

                # Save robot frame conversion  
                gnss_linPosRbt[0, gnss_k] = xrf
                gnss_linPosRbt[1, gnss_k] = yrf
            else:
                gnss_linPosRbt[0, gnss_k] = 0
                gnss_linPosRbt[1, gnss_k] = 0
        
        """ Estimator """
        if mod(k*dt, est_dt) == 0:

            ### Estimator Measurement Arrays ###
            # If GNSS updated, save data to measurement arrays
            if gnss_update == 1:

                # Set estimator GNSS last known sensor update to the sensor last known update
                est_gnssLastUpdateTime = gnss_lastUpdateTime

                # Update the linear position in map frame measurement array
                est_linPosMapArray[:, 1:] = est_linPosMapArray[:, :-1]
                est_linPosMapArray[:, 0] = gnss_linPos[0:2, gnss_k]

                # Update the linear position in the robot frame measurement array
                est_linPosRbtArray[:, 1:] = est_linPosRbtArray[:, :-1]
                est_linPosRbtArray[:, 0] = gnss_linPosRbt[0:2, gnss_k]
            
            # If IMU updated, save data to measurement arrays
            if imu_update == 1:

                # Set estimator last known sensor update to last known sensor update
                est_imuLastUpdateTime = imu_lastUpdateTime

                # Update the angular position (in map and robot frame) measurement array
                est_angPosArray[0, 1:] = est_angPosArray[0, :-1]
                est_angPosArray[0, 0] = imu_angPos[0, imu_k]

                # Update the angular velocity (in map and robot frame) measurement array
                est_angVelArray[0, 1:] = est_angVelArray[0, :-1]
                est_angVelArray[0,0] = imu_angVel[0, imu_k]
                    
                # Update the linear acceleration in robot frame measurement array
                est_linAccRbtArray[:,1:] = est_linAccRbtArray[:, :-1]
                est_linAccRbtArray[:,0] = imu_linAcc[0:2, imu_k] 

            
            ### Estimate Robot Linear Velocity ###
            if (est_useGnssLinVel == 1 and est_useImuLinVel == 0):                
                est_linVelGnssAprxRbt[:, 0] =   \
                    (1/((1/gnss_rr)*est_linPos_n))*(est_linPosRbtArray[0:2,0]-est_linPosRbtArray[0:2,-1])              

                vardxG =(math.sqrt(2)/((1/gnss_rr)**2))*est_rbt_R[0,0]         
                vardyG = (math.sqrt(2)/((1/gnss_rr)**2))*est_rbt_R[1,1]

                est_rbt_m[3:5,est_k] = est_linVelGnssAprxRbt[:,0]*est_linVelBias                             
                

            elif (est_useGnssLinVel == 0 and est_useImuLinVel == 1):  
                if (est_k-1 == 0):
                    est_linVelImuAprxRbt[:,0] = \
                        (1/2)*(est_linAccRbtArray[0:2,1]+est_linAccRbtArray[0:2,0])*(1/imu_rr)                         
                else:
                     est_linVelImuAprxRbt[:,0] = \
                        est_rbt_x[3:5,est_k-1] + (1/2)*(est_linAccRbtArray[0:2,1]+est_linAccRbtArray[0:2,0])*(1/imu_rr)                        

                vardxI = 1                                     
                vardyI = 1                                     
                
                est_rbt_m[3:5,est_k] = est_linVelImuAprxRbt[:,0]*est_linVelBias

            elif (est_useGnssLinVel == 1 and est_useImuLinVel == 1):
                est_linVelGnssAprxRbt[:,0] = \
                    (1/((1/gnss_rr)*est_linPos_n))*(est_linPosRbtArray[0:2,0]-est_linPosRbtArray[0:2,-1])               
                if (est_k-1 == 0):
                    est_linVelImuAprxRbt[:,0] = \
                        (1/2)*(est_linAccRbtArray[0:2,1]+est_linAccRbtArray[0:2,0])*(1/imu_rr)                         
                else:
                     est_linVelImuAprxRbt[:,0] = \
                        est_rbt_x[3:5,est_k-1]+(1/2)*(est_linAccRbtArray[0:2,1] + est_linAccRbtArray[0:2,0])*(1/imu_rr)                         

                vardxG =(math.sqrt(2)/((1/gnss_rr)**2))*est_rbt_R[0,0]         
                vardyG = (math.sqrt(2)/((1/gnss_rr)**2))*est_rbt_R[1,1]
                vardx = (vardxG**2)/(2*vardxG)               
                vardy = (vardyG**2)/(2*vardyG)                 
                
                est_rbt_m[3,est_k] = \
                    (est_linVelGnssAprxRbt[0,0]+est_linVelImuAprxRbt[0,0])/2*est_linVelBias
                est_rbt_m[4,est_k] = \
                    (est_linVelGnssAprxRbt[1,0]+est_linVelImuAprxRbt[1,0])/2*est_linVelBias
            else:
                print('Error in assigning mcaEst.useGnssLinVel and mcaEst.useImuLinVel variables')
                est_rbt_m[3:5,est_k] = matrix([[0], [0]])
            
            ### estimator robot measurement ###         
            # Linear position measurements              
            est_rbt_m[0:2,est_k] = est_linPosRbtArray[0:2, 0]   # linear positions in robot frame            
            
            # Angular position measurements
            est_rbt_m[2,est_k] = est_angPosArray[0,0]   # angular position in robot frame  
          
            # Angular velocity measurements
            est_rbt_m[5,est_k] = est_angVelArray[0,0]

            # Linear acceleration measurements
            est_rbt_m[6:8,est_k] = est_linAccRbtArray[0:2, 0] 
  
            
            ### Estimate map measurement ### 
            # Linear position measurements
            est_map_m[0:2, est_k] = est_linPosMapArray[0:2,0]   # linear position in map frame                  
                
            # Angular position measurements
            if k != t_N:
                est_map_m[2,est_k] = est_rbt_x[2,est_k+1]   # angular position in map frame      

            # Robot frame linear velocity pseudo measurements  
            if k != t_N:
                est_map_m[3:5,est_k] = est_rbt_x[3:5, est_k+1] 

            # Angular velocity measurements   
            if k != t_N:
                est_map_m[5,est_k] = est_rbt_x[5, est_k+1]  # angular velocity in map frame  

        # ### Save data ###
        simX.append(sim_xm[0, k])
        simY.append(sim_xm[1, k])
        linPosX.append(gnss_linPos[0, gnss_k])
        linPosY.append(gnss_linPos[1, gnss_k])
        
        ### Increment time ###
        t_now = t_now+dt


### Plots ###
# Robot x vs y
plotXY()

