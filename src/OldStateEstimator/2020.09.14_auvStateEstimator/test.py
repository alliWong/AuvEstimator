#! /usr/bin/env python

"""
TASKS:
1) Implement EKF 
2) Change simulation state space into 3D
2) Add pressure sensor
3) Rotation treament (gyro bias)

0 x
1 y
2 z
3 ang
4 xvel
5 yvel
6 zvel
7 angvel
8 x acc
9 y acc
10 z acc
11 ang acc
"""

from numpy.linalg import inv
from numpy import identity
from numpy import matrix, diag, random, array, zeros, block, mod, reshape
from math import sqrt, cos, sin, atan2
import math
import matplotlib.pyplot as plt
import numpy as np
import genPlots
from kf import KalmanFilter
from ekf import ExtendedKalmanFilter

""" Simulation User Inputs """
# USER INPUT: time properties
dt = 0.001      # simulation time step [s]
startTime = 0   # simulation start time [s]
endTime = 50    # simulation end time [s]

# USER INPUT: underwater vehicle lumped parameters
sim_m = 225     # mass/inertia [kg]
sim_I = 100     # rotational moment of inertia [kg*m^2]
sim_bxr = 40    # drag in the surge (robot frame along x) direction [N*s/m]
sim_byr = 400   # drag in the sway (robot frame along y) direction [N*s/m]
sim_bpr = 300   # rotational drag in the yaw (robot frame about z) direction [N*s/m]

sim_bzr = 100   # drag in heave (robot frame along z) direction [N*s/m]

# USER INPUT: sensor setup
dvl_rr = 20             # refresh rate of DVL sensor [Hz]
dvl_linVelVar = 1      # variance of the DVL sensor data [m/s]
imu_rr = 50            # refresh rate of IMU sensor [Hz]
imu_angRotVar = 0.05   # variance of the IMU angular position sensor data [rad]
imu_linAccVar = 0.1   # variance of the IMU linear acceleration sensor data [m/s^2]
# imu_angVelVar = 0.05   # variance of the IMU angular velocity sensor data [rad/s]

# Barometer
bar_rr = 10
bar_linPosVar = 0.1

""" Time Properties """
t_N = round(endTime/dt) # total number of time steps []
t_now = startTime       # current time [s]

""" Simulation Setup """
# State, output, and input vectors
sim_xr = matrix(zeros((8, t_N)))    # simulation state vector in robot reference frame
sim_yr = matrix(zeros((12, t_N)))    # simulation output vector in robot reference frame
sim_xm = matrix(zeros((4, t_N)))    # simulation state vector in map reference frame
sim_u = matrix(zeros((4, t_N)))     # input vector (control inputs act on robot frame)

# Create simulation state matrices
sim_A = matrix([
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, -sim_bxr/sim_m, 0, 0, 0],
    [0, 0, 0, 0, 0, -sim_byr/sim_m, 0, 0],
    [0, 0, 0, 0, 0, 0, -sim_bzr/sim_m, 0],
    [0, 0, 0, 0, 0, 0, 0, -sim_bpr/sim_I]])   # state matrix [6x6]
sim_B = matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1/sim_m, 0, 0, 0],
    [0, 1/sim_m, 0, 0],
    [0, 0, 1/sim_m, 0],
    [0, 0, 0, 1/sim_I]])   # input matrix [6x3]
sim_C = matrix(block([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [sim_A[4:8, :], zeros((4,4))]]))   # output matrix [9x9]
sim_D = matrix(block([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0], 
    [0, 0, 0, 0],  
    [0, 0, 0, 0],   
    [sim_B[4:8, :]]]))   # feedthrough matrix [9x3] 

""" Sensor Setup """
dvl_aprxLinPosRbt = matrix(zeros((3, endTime*dvl_rr)))   # approximated linear position data from DVL using trapezoidal integration method in robot frame
dvl_aprxLinPosMap = matrix(zeros((3, endTime*dvl_rr)))   # approximated linear position data from DVL using trapezoidal integration method in map frame
dvl_linVel = matrix(zeros((3,endTime*dvl_rr)))  # linear velocity data from DVL
dvl_angRot = matrix(zeros((3,endTime*dvl_rr)))  # angular position data from IMU, references to DVL timestamp
imu_angRot = matrix(zeros((3,endTime*imu_rr)))  # angular position data from IMU
imu_linAcc = matrix(zeros((3,endTime*imu_rr)))  # linear acceleration data from IMU
imu_acBias = matrix(zeros((3,endTime*imu_rr)))  # acceleration bias

# Barometer sensor
bar_linPos = matrix(zeros((1,endTime*bar_rr)))  # linear z position from barometer sensor

""" Reset Sensor Tracker """
dvl_update = 0  # DVL update tracker (boolean)
imu_update = 0  # IMU update tracker (boolean)
dvl_k = 1   # DVL iteration tracker
imu_k = 1   # IMU iteration tracker
dvl_lastUpdateTime = 0  # DVL last update time tracker [s]
imu_lastUpdateTime = 0  # IMU last update time tracker [s]

# Barometer
bar_update = 0  # barometer update tracker (boolean)
bar_k = 1   # barometer sensor iteration tracker
bar_lastUpdateTime = 0  # barometer last update tome tracker

""" Frame Transformations """
def rbt2map(xrf,yrf,xr0,yr0,psi0,xm0,ym0):
    # Converts pose in the robot frame to pose in the map frame

    # Calculate translations and rotations in robot frame
    Txr = xrf-xr0
    Tyr = yrf-yr0

    # Calculate intermediate length and angle
    li = sqrt(Txr**2+Tyr**2)
    psii = atan2(yrf-yr0, xrf-xr0)  

    # Calculate translations and rotations in map frame
    Txm = cos(psii+psi0)*li
    Tym = sin(psii+psi0)*li

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
    li = sqrt(Txm**2+Tym**2)
    psii = atan2(ymf-ym0, xmf-xm0)
    
    # Calculate translations and rotations in robot frame
    Txr = cos(psii-psi0)*li
    Tyr = sin(psii-psi0)*li

    # Calculate individual components in the robot frame
    xrf = xr0+Txr
    yrf = yr0+Tyr

    return xrf, yrf

""" Controllers """
# Funky controllerss
def controller():
    if (k == 0):
        sim_u[0,k] = 0     # force input in surge direction
        sim_u[1,k] = 0     # force input in sway direction
        sim_u[2,k] = 0     # torque input in heave direction
        sim_u[3,k] = 0     # torque input in yaw direction
    elif (k > 1 and k < 15000):
        sim_u[0,k] = 300    # force input in surge direction
        sim_u[1,k] = 150    # force input in sway direction
        sim_u[2,k] = 150    # force input in heave direction
        sim_u[3,k] = 50     # torque input in yaw direction
    elif (k > 1 and k < 30000):
        sim_u[0,k] = -300   # force input in surge direction
        sim_u[1,k] = -150   # force input in sway direction
        sim_u[2,k] = -150    # force input in heave direction
        sim_u[3,k] = -50    # torque input in yaw direction
    elif (k > 1 and k < 45000):
        sim_u[0,k] = 300  # force input in surge direction
        sim_u[1,k] = 150  # force input in sway direction
        sim_u[2,k] = 150  # force input in heave direction
        sim_u[3,k] = 50   # torque input in yaw direction
    else:
        sim_u[0,k] = -300   # force input in surge direction
        sim_u[1,k] = 150    # force input in sway direction
        sim_u[2,k] = 150    # force input in heave direction
        sim_u[3,k] = -50    # torque input in yaw direction

""" Run simulation """
def runSim():
    # Create discrete state matrices for Euler integration
    F = identity(8)+sim_A*dt    # discrete state matrix
    G = sim_B*dt    # discrete input matrix
    H = sim_C   # discrete output matrix
    J = sim_D   # discrete feedthrough matrix 

    t = matrix([[sim_u[0, k]], [sim_u[1, k]], [sim_u[2, k]], [sim_u[3, k]]]) 
    print(F.shape)
    print(t.shape)
    print(G.shape)

    if k != t_N:
        # Simulate plant using discrete Euler integration
        sim_xr[:, k+1] = F*sim_xr[:, k]+G*matrix([[sim_u[0, k]], [sim_u[1, k]], [sim_u[2, k]], [sim_u[3, k]]]) #  state matrix integration solution 
        sim_yr[:, k+1] = H*block([[sim_xr[:, k+1]], [zeros((4,1))]])+J*matrix([[sim_u[0, k]], [sim_u[1, k]], [sim_u[2, k]], [sim_u[3, k]]])    # state observer matrix integration solution

        # Convert pose in robot frame to pose in map frame
        [sim_xm[0,k+1], sim_xm[1, k+1]] = rbt2map(sim_xr[0, k+1], sim_xr[1, k+1], sim_xr[0, k], sim_xr[1, k], sim_xr[3, k], sim_xm[0, k], sim_xm[1, k])
        sim_xm[3, k+1] = sim_xr[3, k+1]

def updateSensors():
    if k != t_N:
        # Add noise to DVL states to simulate sensor data if DVL update should occur
        if dvl_update == 1:
            dvl_linVel[0, dvl_k] = sim_yr[4, k]+np.random.normal(1)*sqrt(dvl_linVelVar)
            dvl_linVel[1, dvl_k] = sim_yr[5, k]+np.random.normal(1)*sqrt(dvl_linVelVar)
            dvl_linVel[2, dvl_k] = sim_yr[6, k]+np.random.normal(1)*sqrt(dvl_linVelVar)
            dvl_angRot[0, dvl_k] = imu_angRot[0, imu_k-1]

        # Add noise to IMU states to simulate sensor data if IMU update should occur
        if imu_update == 1:
            imu_angRot[0, imu_k] = sim_yr[3, k]+np.random.normal(1)*sqrt(imu_angRotVar)
            imu_linAcc[0, imu_k] = sim_yr[8, k]+np.random.normal(1)*sqrt(imu_linAccVar)
            imu_linAcc[1, imu_k] = sim_yr[9, k]+np.random.normal(1)*sqrt(imu_linAccVar)
            imu_linAcc[2, imu_k] = sim_yr[10, k]+np.random.normal(1)*sqrt(imu_linAccVar)
            
            # Remove bias from inputs (F=ma -> a = F/m)
            imu_acBias[0, imu_k] = sim_u[0, k]/sim_m 
            imu_acBias[1, imu_k] = sim_u[1, k]/sim_m 
            imu_acBias[2, imu_k] = sim_u[2, k]/sim_m 

            # imu_angVel[0, imu_k] = sim_yr[7, k]+np.random.normal(1)*sqrt(imu_angVelVar)
            
        # Add noise to barometer states to simulate sensor data if barometer update should occur
        if bar_update ==1:
            bar_linPos[:, bar_k] = sim_xm[2, k]+np.random.normal(1)*sqrt(bar_linPosVar)

        # DVL approximated robot frame position and convert to map frame (this must occur after previous state)
        if dvl_update == 1:

            if dvl_k != 1:

                # Approximate position in robot frame using trapzeoidal integration method
                if (dvl_k == 1):
                    dvl_aprxLinPosRbt[:,dvl_k] = (1/2)*(dvl_linVel[:,dvl_k] + dvl_linVel[:,dvl_k-1])*(1/dvl_rr)
                else:
                    dvl_aprxLinPosRbt[:,dvl_k] = sim_xr[0:3,k-1]+(1/2)*(dvl_linVel[:,dvl_k]+dvl_linVel[:,dvl_k-1])*(1/dvl_rr)    

                # Convert DVL pose in robot frame to map frames
                [xmf, ymf] = rbt2map(
                    dvl_aprxLinPosRbt[0, dvl_k],
                    dvl_aprxLinPosRbt[1, dvl_k],
                    dvl_aprxLinPosRbt[0, dvl_k-1],
                    dvl_aprxLinPosRbt[1, dvl_k-1],
                    dvl_angRot[0, dvl_k],
                    dvl_aprxLinPosMap[0, dvl_k-1],
                    dvl_aprxLinPosMap[1, dvl_k-1]
                )

                # Save map frame conversion
                dvl_aprxLinPosMap[0, dvl_k] = xmf
                dvl_aprxLinPosMap[1, dvl_k] = ymf
            else:
                dvl_aprxLinPosMap[0, dvl_k] = 0
                dvl_aprxLinPosMap[1, dvl_k] = 0


""" Store data """
# Store data for plotting
simX = []
simY = []
linPosX = []
linPosY = []
estMapX = []
estMapY = []
rbtLinPosX = []
rbtLinPosY = []

imu_counter = 1
dvl_counter = 1
bar_counter = 1

# time = np.linspace(0, endTime, t_N)
"""  RUN LOOP """
for k in range(0, t_N, 1):
    if k< 48000:
        """ Simulation """
        # Controller
        controller()

        # Simulate robot 
        runSim()

        # Check sensor updates
        if k != t_N:
            # Increment GNSS and set update flag to true if GNSS update should occur
            if k*dt > (1/dvl_rr)*dvl_counter:
                dvl_k = dvl_k+1   # increment sensor
                dvl_update = 1     # set update flag to true
                dvl_lastUpdateTime = t_now # set the last time the sensor updated
                dvl_counter = dvl_counter+1
            else:
                dvl_update = 0     # set update flag to false
                    
            #  Increment IMU and set update flag to true if IMU update should occur
            if k*dt > (1/imu_rr)*imu_counter:
                imu_k = imu_k+1     # increment sensor
                imu_update = 1      # set update flag to true
                imu_lastUpdateTime = t_now  # set the last time the sensor updated
                imu_counter = imu_counter+1
            else:
                imu_update = 0  # set update flag to false
            
            #  Increment IMU and set update flag to true if IMU update should occur
            if k*dt > (1/bar_rr)*bar_counter:
                bar_k = bar_k+1     # increment sensor
                bar_update = 1      # set update flag to true
                bar_lastUpdateTime = t_now  # set the last time the sensor updated
                bar_counter = bar_counter+1
            else:
                bar_update = 0  # set update flag to false

        # Update sensors
        updateSensors()

        # """ Estimator """
        # if k*dt > est_dt*est_counter:   

        #     # Estimator measurement array
        #     estMeasArrays()       

        #     # Estimator robot measurement            
        #     estRbtMeas()  

        #     # Linear velocity
        #     rbtLinVel()   

        #     # Map linear position
        #     # mapLinPos()

        #     # Run robot kalman filter
        #     if k != t_N:
        #         # if k*dt > est_dt*est_counter:

        #         x = est_rbt_x[:, est_k]    # last state estimate vector from robot frame
        #         P = est_rbt_P[:,:,est_k]   # last covariance matrix
        #         Q = est_rbt_Q              # process noise covariance matrix
        #         R = est_rbt_R              # measurement noise covariance
        #         H = est_rbt_H              # observation matrix
        #         z = est_rbt_m[:,est_k]     # measurement vector
        #         u = matrix([[0], [0], [0]])  # control input vector (don't give kalman filter knowledge about thruster inputs)

        #         A = est_rbt_A
        #         B = est_rbt_B
                    
        #         state = KalmanFilter(x, P, z, u, A, B, Q, R, H)
        #         x, P = state.predict(x, P, u)
        #         x, K, P = state.update(x, P, z)

        #             # print('x', x)
        #             # # print('P', P)
        #             # # print('Q', Q)
        #             # # print('R', R)
        #             # # print('H', H)
        #             # # print('z', z)
        #             # # print('u', u)
        #             # # print('K', K)

        #         est_rbt_x[:, est_k+1] = x
        #         est_rbt_L[:,:, est_k+1] = K
        #         est_rbt_P[:,:, est_k+1] = P         
                
        #     # Estimator map measurement
        #     estMapMeas()  

        #     # Map linear position
        #     # mapLinPos()
                
        #     # Run map kalman filter
        #     if k != t_N:
                    
        #         # if k*dt > est_dt*est_counter:
                        
        #         x = est_map_x[:,est_k]       # last estimate from robot frame
        #         P = est_map_P[:,:,est_k]     # last covariance matrix
        #         Q = est_map_Q                # process noise covariance
        #         R = est_map_R                # measurement noise covariance
        #         H = est_map_H                # measurement matrix
        #         z = est_map_m[:,est_k]       # measurement
        #         u = matrix([[0], [0], [0]])  # control input vector (don't give kalman filter knowledge about thruster inputs)
                        
        #         A = est_map_A(est_map_m[:,est_k])
        #         B = est_map_B
                        
        #         state = KalmanFilter(x, P, z, u, A, B, Q, R, H)
        #         x, P = state.predict(x, P, u)
        #         x, K, P = state.update(x, P, z)

        #         est_map_x[:,est_k+1] = x
        #         est_map_L[:,:,est_k+1] = K
        #         est_map_P[:,:,est_k+1] = P
                        
        #         # print('x', x)
        #         # print('P', P)
        #         # print('Q', Q)
        #         # print('R', R)
        #         # print('H', H)
        #         # print('z', z)
        #         # print('u', u)
        #         # print('K', K)

        #         # Increment tracking k variable
        #         est_k = est_k+1
                
        #     # increment counter, end of loop
        #     est_counter = est_counter+1
            
        # # ### Save data ###
        # simX.append(sim_xm[0, k])
        # simY.append(sim_xm[1, k])
        # linPosX.append(dvl_aprxLinPosMap[0, dvl_k])
        # linPosY.append(dvl_aprxLinPosMap[1, dvl_k])
        # estMapX.append(est_map_x[0, est_k])
        # estMapY.append(est_map_x[1, est_k])
            
        ### Increment time ###
        t_now = t_now + dt

""" Plots """
# Robot x vs y
# genPlots.plotXY(simX, simY, linPosX, linPosY, estMapX, estMapY)
# genPlots.plotMap(t_N, endTime, simX, simY, estMapX, estMapY, linPosX, linPosY)
# plt.show()
