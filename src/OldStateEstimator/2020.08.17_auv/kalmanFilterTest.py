#! /usr/bin/env python

"""
TASKS:
1) Create 2D Kalman Filter
2) Test 2D Kalman Filter
3) Generate simulation of USV
4) Generate estimated trajectory
5) Create 3D Kalman Filter
6) Create 3D EKF

"""

from numpy.linalg import inv
from numpy import identity
from numpy import matrix, diag, random, array, zeros, block, mod, reshape
import math
import matplotlib.pyplot as plt
import numpy as np


################## Simulation user inputs ##################
# Time properties
dt = 0.001      # simulation time step [s]
startTime = 0
endTime = 50    # simulation end time [s]
t_N = round(endTime/dt) # total number of time steps [s]
t_now = startTime

# USV lumped parameters
sim_m = 225
sim_I = 100
sim_bxr = 40
sim_byr = 400
sim_bpr = 300

# Sensor setup
gnss_rr = 1      # refresh rate of GPS sensor [Hz]
gnss_linPosVar = 1
imu_rr = 20     # refresh rate of IMU sensor [Hz]
imu_angPosVar = 0.05
imu_angVelVar = 0.05
imu_linAccVar = 0.1

################## Simulation Setup ##################
# State, output, and input vectors
sim_xr = zeros((6, t_N))
sim_yr = zeros((9, t_N))
sim_xm = zeros((3, t_N))
sim_u = zeros((3, t_N))

# Convert into matrix
sim_xr = matrix(sim_xr)
sim_yr = matrix(sim_yr)
sim_xm = matrix(sim_xm)
sim_u = matrix(sim_u)

sim_A = matrix([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, -sim_bxr/sim_m, 0, 0],
    [0, 0, 0, 0, -sim_byr/sim_m, 0],
    [0, 0, 0, 0, 0, -sim_bpr/sim_I]         
])

sim_B = matrix([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1/sim_m, 0, 0],
    [0, 1/sim_m, 0],
    [0, 0, 1/sim_I]
    ])

sim_C = block([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [sim_A[3:6, :], zeros((3,3))]
])
sim_C = matrix(sim_C)

sim_D = block([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],  
    [sim_B[3:6, :]] 
])
sim_D = matrix(sim_D)

######################  Sensor Setup #####################################
gnss_linPos = zeros((2, endTime*gnss_rr))
gnss_linPosRbt = zeros((2,endTime*gnss_rr))
gnss_angPos = zeros((2,endTime*gnss_rr))
imu_angPos = zeros((1,endTime*imu_rr))
imu_angVel = zeros((1,endTime*imu_rr))
imu_linAcc = zeros((2,endTime*imu_rr))

# Convert into matrix
gnss_linPos = matrix(gnss_linPos)
gnss_linPosRbt = matrix(gnss_linPosRbt)
gnss_angPos = matrix(gnss_angPos)
imu_angPos = matrix(imu_angPos)
imu_angVel = matrix(imu_angVel)
imu_linAcc = matrix(imu_linAcc)

# print(gnss_linPos.shape)
# print(gnss_linPosRbt.shape)
# print(gnss_angPos.shape)
# print(imu_angPos.shape)
# print(imu_angVel.shape)
# print(imu_linAcc.shape)

###################### Reset Sensor Tracker #####################################
gnss_update = 0
imu_update = 0
gnss_k = 1
imu_k = 1
gnss_lastUpdatetime = 0
imu_lastUpdateTime = 0

###################### RBT 2 MAP #####################################
def rbt2map(xrf,yrf,xr0,yr0,psi0,xm0,ym0):
    Txr = xrf-xr0
    Tyr = yrf-yr0

    li = math.sqrt(Txr**2+Tyr**2)
    psii = math.atan2(yrf-yr0, xrf-xr0)

    Txm = math.cos(psii+psi0)*li
    Tym = math.sin(psii+psi0)*li

    xmf = xm0+Txm
    ymf = ym0+Tym

    return xmf, ymf

###################### MAP 2 RBT #####################################
def map2rbt(xmf,ymf,xm0,ym0,psi0,xr0,yr0):
    Txm = xmf-xm0
    Tym = ymf-ym0

    li = math.sqrt(Txm**2+Tym**2)
    psii = math.atan2(ymf-ym0, xmf-xm0)
    
    Txr = math.cos(psii-psi0)*li
    Tyr = math.sin(psii-psi0)*li

    xrf = xr0+Txr
    yrf = yr0+Tyr

    return xrf, yrf

##################### PLOT #######################
simPlot = True
senPlot = False
caEstPlot = False
ddEstPlot = False

################### STORING RESULTS #######################
resultsX = []
resultsY = []


#####################  LOOP #######################
for k in range (0, t_N, 1):
    # print(sim_xm[1, k])
    print(' IN LOOP')
    if k< 48000:
        ### controller ###
        # print(sim_xm[1, k])
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

        ### run sim ###
        F = identity(6)+sim_A*dt
        G = sim_B*dt
        H = sim_C
        J = sim_D

        if k != t_N:
            sim_xr[:, k+1] = F*sim_xr[:, k]+G*matrix([[sim_u[0, k]], [sim_u[1, k]], [sim_u[2, k]]])
            sim_yr[:, k+1] = H*block([[sim_xr[:, k+1]], [zeros((3,1))]])+J*matrix([[sim_u[0, k]], [sim_u[1, k]], [sim_u[2, k]]])

            [sim_xm[0,k+1], sim_xm[1, k+1]] = rbt2map(sim_xr[0, k+1], sim_xr[1, k+1], sim_xr[0, k], sim_xr[1, k], sim_xr[2, k], sim_xm[0, k], sim_xm[1, k])
            sim_xm[2, k+1] = sim_xr[2, k+1]

        ### check sensor update
        if k != t_N:
            if mod(k*dt, 1/gnss_rr) == 0:
                gnss_k = gnss_k+1
                gnss_update = 1
                gnss_lastUpdatetime = t_now
            else:
                gnss_update = 0
                
            if mod(k*dt, 1/imu_rr) == 0:
                imu_k = imu_k+1
                imu_update = 1
                imu_lastUpdatetime = t_now
            else:
                imu_update = 0
            
        ### update sensors
        if k != t_N:
            if gnss_update == 1:
                gnss_linPos[0, gnss_k] = sim_xm[0, k]+np.random.rand(1)*math.sqrt(gnss_linPosVar)
                gnss_linPos[1, gnss_k] = sim_xm[1, k]+np.random.rand(1)*np.sqrt(gnss_linPosVar)
                gnss_angPos[0, gnss_k] = imu_angPos[0, imu_k-1]
                
            if imu_update == 1:
                imu_angPos[0, imu_k] = sim_yr[2, k]+np.random.rand(1)*np.sqrt(imu_angPosVar)
                imu_angVel[0, imu_k] = sim_yr[5, k]+np.random.rand(1)*np.sqrt(imu_angVelVar)
                imu_linAcc[0, imu_k] = sim_yr[6, k]+np.random.rand(1)*np.sqrt(imu_linAccVar)
                imu_linAcc[1, imu_k] = sim_yr[7, k]+np.random.rand(1)*np.sqrt(imu_linAccVar)
                
        if gnss_update == 1:
            if gnss_k != 1:
                [xrf, yrf] = map2rbt(
                    gnss_linPos[0, gnss_k],
                    gnss_linPos[1, gnss_k],
                    gnss_linPos[0, gnss_k-1],
                    gnss_linPos[1, gnss_k-1],
                    gnss_angPos[0, gnss_k],
                    gnss_linPosRbt[0, gnss_k-1],
                    gnss_linPosRbt[1, gnss_k-1])   
                gnss_linPosRbt[0, gnss_k] = xrf
                gnss_linPosRbt[1, gnss_k] = yrf
            else:
                gnss_linPosRbt[0, gnss_k] = 0
                gnss_linPosRbt[1, gnss_k] = 0

        resultsX.append(sim_xm[0, k])
        resultsY.append([sim_xm[1, k]])
        # result_x= np.hstack(resultsX)
        # result_y= np.hstack(resultsY)


        # # # increment time
        t_now = t_now+dt
        # print(t_now)
        # print(sim_xr[0, :])

print('OUTSIDE OF LOOP')
if simPlot == True:
    # titleFontSize = 32
    # defaultFontSize = 28
    markerSize = 10
    linewidth = 2

    # plt.figure()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect('key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    plt.plot(resultsX, resultsY, color = 'k', linestyle = '--', linewidth = linewidth, markersize = markerSize)
    plt.title('Map Frame xy Position')
    plt.xlabel('x_m')
    plt.ylabel('y_m')
    plt.grid(True)
    plt.axis('equal')
    plt.show()



###################### Kalman Filter Class ######################
class KalmanFilter:
    """
    Simple Kalman filter

    Control term has been omitted for now
    """
    def __init__(self, x, P, z, u, A, B, Q, R, H):
        self.x = x  # estimator state estimate
        self.P = P  # covariance matrx
        self.z = z  # measurement matrix
        self.u = u  # control input m atrix
        self.A = A  # state matrix
        self.B = B  # input matrix
        self.Q = Q  # process noise covariance matrix
        self.R = R  # measurement noise covariance matrix
        self.H = H  # observation matrix


    def predict(self, x, P, u):
        # Prediction for state vector and covariance
        x = self.A * x + self.B * u
        P = self.A * P * (self.A.T) + self.Q
        return(x, P)

    def update(self, x, P, z):
        # Compute Kalman Gain
        K = P * (self.H.T) * inv(self.H * P * (self.H.T) + self.R)

        # Correction based on observation
        x += K * (z - self.H * x)
        P = P - K * self.H * P
        return (x, P)

# ####################################################################
# """
# Simulation user input
# """
# # Time properties
# dt = 0.001      # simulation time step [s]
# endTime = 50    # simulation end time [s]
# t_N = round(endTime/dt) # total number of time steps [s]
# dt = 0.001
# # Standard deviation of random accelerations
# sigma_a = 0.2
# # Standard deviation of observations
# sigma_z = 0.2
# # State vector: [[Position], [velocity]]
# x = array([[0.0], [0.0]])
# # Initial state covariance
# P = diag((0.0, 0.0))
# # Acceleration model
# G = array([[(dt ** 2) / 2], [dt]])
# # State transition model
# A = array([[1, dt], [0, 1]])
# # Observation vector
# z = array([[0.0], [0.0]])
# # Observation model
# H = array([[1, 0], [0, 0]])
# # Observation covariance
# R = array([[sigma_z ** 2, 0], [0, 1]])
# # Process noise covariance matrix
# Q = G * (G.T) * sigma_a ** 2

# B = 0
# u = 0
# # Initialise the filter
# kf = KalmanFilter(x, P, z, u, A, B, Q, R, H)

# # Set the actual position equal to the starting position
# A = x

# # Create log for generating plots
# log = Logger()
# log.new_log('measurement')
# log.new_log('estimate')
# log.new_log('actual')
# log.new_log('time')
# log.new_log('covariance')
# log.new_log('moving average')

# moving_avg = MovingAverage(15)

# for i in range(0, t_N):
#     # Generate a random acceleration
#     # w = matrix(random.multivariate_normal([0.0, 0.0], Q)).T
#     # Predict
#     (x, P) = kf.predict(x, P, u)
#     # Update
#     (x, P) = kf.update(x, P, z)
#     # Update the actual position
#     A = A + B*u
#     # Synthesise a new noisy measurement distributed around the real position
#     Z = matrix([[random.normal(A[0, 0], sigma_z)], [0.0]])
#     # Update the moving average with the latest measured position
#     moving_avg.update(Z[0, 0])
#     # Update the log for plotting later
#     log.log('measurement', z[0, 0])
#     log.log('estimate', x[0, 0])
#     log.log('actual', A[0, 0])
#     log.log('time', i * dt)
#     log.log('covariance', P[0, 0])
#     log.log('moving average', moving_avg.getAvg())

# # Plot the system behaviour
# plotter = KalmanPlotter()
# plotter.plot_kalman_data(log)