#! /usr/bin/env python

"""
TASKS:
3) Create 3D Kalman Filter
4) Create 3D EKF


states (MAP FRAME) 
x pos   0
y pos   1
ang pos 2
x vel   3
y vel   4
ang vel 5
x acc   6
y acc   7
ang acc 8

"""

from numpy.linalg import inv
from numpy import identity
from numpy import matrix, diag, random, array, zeros, block, mod, reshape
from math import sqrt, cos, sin, atan2
import math
import matplotlib.pyplot as plt
import numpy as np
import genPlots
from kalmanFilter import KalmanFilter



""" Simulation User Inputs """
# USER INPUT: time properties
dt = 0.001              # simulation time step [s]
startTime = 0           # simulation start time [s]
endTime = 50            # simulation end time [s]

# USER INPUT: underwater vehicle lumped parameters
sim_m = 225     # mass/inertia [kg]
sim_I = 100     # rotational moment of inertia [kg*m^2]
sim_bxr = 40    # drag in the surge (robot frame along x) direction [N*s/m]
sim_byr = 400   # drag in the sway (robot frame along y) direction [N*s/m]
sim_bpr = 300   # rotational drag in the yaw (robot frame about z) direction [N*s/m]

# USER INPUT: sensor setup
dvl_rr = 50             # refresh rate of DVL sensor [Hz]
dvl_linVelVar = 0.05       # variance of the DVL sensor data [m/s]
imu_rr = 30             # refresh rate of IMU sensor [Hz]
imu_angPosVar = 0.05    # variance of the IMU angular position sensor data [rad]
imu_angVelVar = 0.05    # variance of the IMU angular velocity sensor data [rad/s]
imu_linAccVar = 0.05     # variance of the IMU linear acceleration sensor data [m/s^2]

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
dvl_linVel = matrix(zeros((2, endTime*dvl_rr)))   # linear velocity data from DVL
dvl_angPos = matrix(zeros((2,endTime*dvl_rr)))    # angular position data from IMU, references to DVL timestamp
imu_angPos = matrix(zeros((1,endTime*imu_rr)))  # angular position data from IMU
imu_angVel = matrix(zeros((1,endTime*imu_rr)))  # angular velocity data from IMU
imu_linAcc = matrix(zeros((2,endTime*imu_rr)))  # linear acceleration data from IMU

dvl_aprxLinPosRbt = matrix(zeros((2, endTime*dvl_rr)))   # approximated linear position data from DVL using integration
dvl_aprxLinPosMap = matrix(zeros((2, endTime*dvl_rr)))   # approximated linear position data from DVL using integration


""" Reset Sensor Tracker """
dvl_update = 0  # DVL update tracker (boolean)
imu_update = 0  # imu update tracker (boolean)
dvl_k = 1           # DVL iteration tracker
imu_k = 1           # imu iteration tracker
dvl_lastUpdateTime = 0 # DVL last update time tracker [s]
imu_lastUpdateTime = 0  # imu last update time tracker [s]

""" Frame Transformations """
def rbt2map(xrf,yrf,xr0,yr0,psi0,xm0,ym0):
    # Converts pose in the robot frame to pose in the map frame

    # Calculate translations and rotations in robot frame
    Txr = xrf-xr0
    Tyr = yrf-yr0

    # Calculate intermediate length and angle
    li = sqrt(Txr**2+Tyr**2)
    psii = atan2(yrf-yr0, xrf-xr0)  # atan or atan2

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

        # # Convert pose in robot frame to pose in map frame
        [sim_xm[0,k+1], sim_xm[1, k+1]] = rbt2map(sim_xr[0, k+1], sim_xr[1, k+1], sim_xr[0, k], sim_xr[1, k], sim_xr[2, k], sim_xm[0, k], sim_xm[1, k])
        sim_xm[2, k+1] = sim_xr[2, k+1]

def updateSensors():
    if k != t_N:
        # Add noise to DVL states to simulate sensor data if DVL update should occur
        if dvl_update == 1:
            dvl_linVel[0, dvl_k] = sim_yr[3, k]+np.random.normal(1)*sqrt(dvl_linVelVar)
            dvl_linVel[1, dvl_k] = sim_yr[4, k]+np.random.normal(1)*sqrt(dvl_linVelVar)
            dvl_angPos[0, dvl_k] = imu_angPos[0, imu_k-1]

        # Add noise to IMU states to simulate sensor data if IMU update should occur
        if imu_update == 1:
            imu_angPos[0, imu_k] = sim_yr[2, k]+np.random.normal(1)*sqrt(imu_angPosVar)
            imu_angVel[0, imu_k] = sim_yr[5, k]+np.random.normal(1)*sqrt(imu_angVelVar)
            imu_linAcc[0, imu_k] = sim_yr[6, k]+np.random.normal(1)*sqrt(imu_linAccVar)
            imu_linAcc[1, imu_k] = sim_yr[7, k]+np.random.normal(1)*sqrt(imu_linAccVar)
            
        # Convert DVL to map frame (this must occur after previous state)
        if dvl_update == 1:

            if dvl_k != 1:

                if (dvl_k == 1):
                    dvl_aprxLinPosRbt[:,dvl_k] = (1/2)*(dvl_linVel[:,dvl_k] + dvl_linVel[:,dvl_k-1])*(1/dvl_rr)
                else:
                    dvl_aprxLinPosRbt[:,dvl_k] = sim_xr[0:2,k-1]+(1/2)*(dvl_linVel[:,dvl_k]+dvl_linVel[:,dvl_k-1])*(1/dvl_rr)    

                [xmf, ymf] = rbt2map(
                    dvl_aprxLinPosRbt[0, dvl_k],
                    dvl_aprxLinPosRbt[1, dvl_k],
                    dvl_aprxLinPosRbt[0, dvl_k-1],
                    dvl_aprxLinPosRbt[1, dvl_k-1],
                    dvl_angPos[0, dvl_k],
                    dvl_aprxLinPosMap[0, dvl_k-1],
                    dvl_aprxLinPosMap[1, dvl_k-1]
                )

                # Save map frame conversion
                dvl_aprxLinPosMap[0, dvl_k] = xmf
                dvl_aprxLinPosMap[1, dvl_k] = ymf
            else:
                dvl_aprxLinPosMap[0, dvl_k] = 0
                dvl_aprxLinPosMap[1, dvl_k] = 0
                                
""" Estimater User Inputs """
# USER INPUT: time step properties
est_dt = 0.025  # estimator time step [s]

# USER INPUT: Number of past sensor updates to store for calculating the incremental moving mean of sensor data
est_linVel_n = 2    # linear velocity mean array size (must be 2 or more)
est_angPos_n = 2    # angular position mean array size (must be 2 or more)
est_angVel_n = 2    # angular velocity mean array size (must be 2 or more)
est_linAcc_n = 2    # linear acceleration mean array size (must be 2 or more)

# USER INPUT: decide which sensors to use for pseudo linear velocity calculation
est_useDvlLinPos = 1   # set to 1 to use DVL trapezoidal method, set to 0 to ignore
est_useImuLinPos = 1   # set to 1 to use IMU trapezoidal method, set to 0 to ignore
est_useDvlLinVel = 1   # set to 1 to use DVL trapezoidal method, set to 0 to ignore
est_useImuLinVel = 1   # set to 1 to use IMU trapezoidal method, set to 0 to ignore
est_linPosBias = 1   # bias (multiplier) on linear velocity output
est_linVelBias = 1   # bias (multiplier) on position output

# USER INPUT: robot frame estimator measurement noise covariance
est_rbt_R_linPos = 2       # linear position [m]
est_rbt_R_angPos = 0.075   # angular position [rad]
est_rbt_R_linVel = 2       # linear velocity [m/s]
est_rbt_R_angVel = 0.075   # angular velocity [rad/s]
est_rbt_R_linAcc = 0.25    # linear acceleration [m/s^s]
est_rbt_R_angAcc = 1       # angular acceleration [rad/s^2] (not measured, so this is actually irrelevant)

# USER INPUT: robot frame estimator process noise covariance
est_rbt_Q_linPos = est_rbt_R_linPos*1e2    # linear position [m]
est_rbt_Q_angPos = est_rbt_R_angPos*1e2    # angular position [rad]
est_rbt_Q_linVel = est_rbt_R_linVel*1e2    # linear velocity [m/s]
est_rbt_Q_angVel = est_rbt_R_angVel*1e-1   # angular velocity [rad/s]
est_rbt_Q_linAcc = est_rbt_R_linAcc*1e-1   # linear acceleration [m/s^s]
est_rbt_Q_angAcc = est_rbt_R_angAcc*1e-1   # angular acceleration [rad/s^2] (not measured, so this is actually irrelevant)

# USER INPUT: robot frame estimator observation/observability
est_H_linPos = 1       # linear position (pseudo-measured)
est_H_angPos = 1       # angular position (measured)
est_H_linVel = 1       # linear velocity (measured)
est_H_angVel = 1       # angular velocity (measured)
est_H_linAcc = 1       # linear acceleration (measured)
est_H_angAcc = 0       # angular acceleration (NOT measured)
            
# USER INPUT: map frame estimator measurement noise covariance
est_map_R_linPos = 2       # linear position [m]
est_map_R_angPos = 0.075   # angular position [rad]
est_map_R_linVel = 1       # linear velocity [m/s]
est_map_R_angVel = 0.075   # angular velocity [rad/s]
            
# USER INPUT: map frame estimator process noise covariance          
est_map_Q_linPos = est_map_R_linPos*1e1    # linear position [m]
est_map_Q_angPos = est_map_R_angPos*1e1    # angular position [rad]
est_map_Q_linVel = est_map_R_linVel*1e1    # linear velocity [m/s]
est_map_Q_angVel = est_map_R_angVel*1e1    # angular velocity [rad/s]
            
# USER INPUT: map frame estimator observation/observability
est_H_linPos = 1       # linear position (pseudo-measured)
est_H_angPos = 1       # angular position (measured)
est_H_linVel = 1       # linear velocity (measured)
est_H_angVel = 1       # angular velocity (measured)      

""" Estimator Setup """
# Instantiate measurement variables and arrays
est_counter = 1
est_k = 1                                      # iteration tracking variable
est_dvlLastUpdateTime = 0                      # last DVL update time [s]
est_imuLastUpdateTime = 0                      # last IMU update time [s]

est_linPosMapArray = matrix(zeros((2,est_linVel_n)))                  # linear velocity directly from DVL in map frame measurement array
est_linVelRbtArray = matrix(zeros((2,est_linVel_n)))                  # linear velocity directly from DVL in robot frame measurement array
est_angPosArray = matrix(zeros((1,est_angPos_n)))          # angular position directly from IMU in map and robot frame measurement array
est_angVelArray = matrix(zeros((1,est_angVel_n)))          # angular velocity directly from IMU in map and robot frame measurement array
est_linAccRbtArray = matrix(zeros((2,est_linAcc_n)))       # linear acceleration directly from IMU in robot frame measurement array

est_linPosDvlAprxMap = matrix(zeros((2,1)))     # approximated linear position in robot frame from trapezoidal method of DVL
est_linPosDvlAprxRbt = matrix(zeros((2,1)))     # approximated linear position in robot frame from trapezoidal method of DVL
est_linPosImuAprxMap = matrix(zeros((2,1)))     # approximated linear position in robot frame from trapezoidal method of IMU
est_linPosImuAprxRbt = matrix(zeros((2,1)))     # approximated linear position in robot frame from trapezoidal method of IMU
est_linVelImuAprxRbt = matrix(zeros((2,1)))              # approximated linear velocity in robot frame from trapezoidal method of IMU


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
est_map_s = array(zeros((6,6,int(endTime/est_dt))))

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
    [1, 0, 0, cos(x[2])*est_dt, -sin(x[2])*est_dt, 0],
    [0, 1, 0, sin(x[2])*est_dt, cos(x[2])*est_dt, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]])                   # state matrix
est_map_B = matrix(zeros((6,3)))           # input matrix


def estMeasArrays():
    # If DVL updated, save data to measurement arrays
    if dvl_update == 1:
               
        # Set estimator DVL last known sensor update to the sensor last known update
        est_dvlLastUpdateTime = dvl_lastUpdateTime    

        est_linPosMapArray[:, 1:] = est_linPosMapArray[:, :-1]
        est_linPosMapArray[:, 0] = dvl_aprxLinPosMap[0:2, dvl_k]

        # Update the linear velocity in the robot frame measurement array
        est_linVelRbtArray[:, 1:] = est_linVelRbtArray[:, :-1]
        est_linVelRbtArray[:, 0] = dvl_linVel[0:2, dvl_k]

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

def estRbtMeas():
    # Position measurements
    est_rbt_m[0:2, est_k] = est_rbt_m[0:2, est_k]    # linear position in robot frame        

    # Angular position measurements
    est_rbt_m[2,est_k] = est_angPosArray[0,0]   # angular position in robot frame  

    # Linear velocity measurements              
    est_rbt_m[3:5,est_k] = est_rbt_m[3:5,est_k]   # linear velocity in robot frame    

    # Angular velocity measurements
    est_rbt_m[5,est_k] = est_angVelArray[0,0]       

    # Linear acceleration measurements
    est_rbt_m[6:8,est_k] = est_linAccRbtArray[0:2,0]      

def rbtLinVel():
    if (est_useDvlLinVel == 1 and est_useImuLinVel == 0):
        est_rbt_m[3:5,est_k] = est_linVelRbtArray[:,0]*est_linVelBias

    if (est_useDvlLinVel == 0 and est_useImuLinVel == 1):
        if (est_k == 1):
            est_linVelImuAprxRbt[:,0] = (1/2)*(est_linAccRbtArray[0:2,1]+est_linAccRbtArray[0:2,0])*(1/imu_rr)                         
        else:
            est_linVelImuAprxRbt[:,0] = est_rbt_x[3:5,est_k-1]+(1/2)*(est_linAccRbtArray[0:2,1]+est_linAccRbtArray[0:2,0])*(1/imu_rr)                                     
               
        est_rbt_m[3:5,est_k] = est_linVelImuAprxRbt[:,0]*est_linVelBias

    elif (est_useDvlLinVel == 1 and est_useImuLinVel == 1):
        if (est_k == 1):
            est_linVelImuAprxRbt[:,0] = (1/2)*(est_linAccRbtArray[0:2,1]+est_linAccRbtArray[0:2,0])*(1/imu_rr)                         
        else:
            est_linVelImuAprxRbt[:,0] = est_rbt_x[3:5,est_k-1]+(1/2)*(est_linAccRbtArray[0:2,1]+est_linAccRbtArray[0:2,0])*(1/imu_rr)                                     
               
        est_rbt_m[3,est_k] = (est_linVelRbtArray[0,0]+est_linVelImuAprxRbt[0,0])/2*est_linVelBias
        est_rbt_m[4,est_k] = (est_linVelRbtArray[1,0]+est_linVelImuAprxRbt[1,0])/2*est_linVelBias

    else:
        est_rbt_m[3:5, est_k] = matrix([[0], [0]]) 

def rbtLinPos():
    if (est_useDvlLinPos == 1 and est_useImuLinPos == 1):

        # if est_k != 1:
            # if (est_k == 1):
                # est_linPosDvlAprxRbt[:,0] = (1/2)*(est_linVelRbtArray[0:2,1]+est_linVelRbtArray[0:2,0])*(1/dvl_rr)                         
                # est_linVelImuAprxRbt[:,0] = (1/2)*(est_linVelImuAprxRbt[0:2,-1]+est_linVelImuAprxRbt[0:2,0])*(1/imu_rr)                         
            
            # else:
                # est_linPosDvlAprxRbt[:,0] = est_rbt_x[0:2,est_k-1]+(1/2)*(est_linVelRbtArray[0:2,1]+est_linVelRbtArray[0:2,0])*(1/dvl_rr)                          
                # est_linVelImuAprxRbt[:,0] = (1/2)*(est_linVelImuAprxRbt[0:2,1]+est_linVelImuAprxRbt[0:2,0])*(1/imu_rr)                         

        if (est_k == 1):
            est_rbt_m[0:2,est_k] = (1/2)*(est_rbt_m[3:5,est_k]+est_rbt_m[3:5,est_k-1])*est_dt 
        else:
            est_rbt_m[0:2,est_k] = est_rbt_x[0:2,est_k-1]+(1/2)*(est_rbt_m[3:5,est_k]+est_rbt_m[3:5,est_k-1])*est_dt

            # est_rbt_m[0, est_k] = est_linPosDvlAprxRbt[0, 0]
            # est_rbt_m[1, est_k] = est_linPosDvlAprxRbt[1, 0]

            # print('dvl', est_linPosDvlAprxRbt[0,0])
            # print('est', est_rbt_m[0,est_k])

        # est_rbt_m[0,est_k] = est_linPosDvlAprxRbt[0,0]*est_linPosBias
        # est_rbt_m[1,est_k] = est_linPosDvlAprxRbt[1,0]*est_linVelBias

        est_rbt_m[0,est_k] = (est_rbt_m[0,est_k]+dvl_aprxLinPosRbt[0, dvl_k])/2*est_linPosBias
        est_rbt_m[1,est_k] = (est_rbt_m[1,est_k]+dvl_aprxLinPosRbt[1, dvl_k])/2*est_linPosBias      
             
        # est_rbt_m[0,est_k] = (est_linPosDvlAprxRbt[0,0]+est_linPosDvlAprxRbt[0,0])/2*est_linVelBias
        # est_rbt_m[1,est_k] = (est_linVelImuAprxRbt[1,0]+est_linVelImuAprxRbt[1,0])/2*est_linVelBias

    else:
        est_rbt_m[0:2, est_k] = matrix([[0], [0]]) 

def mapLinPos():
    if (est_useDvlLinPos == 1 and est_useImuLinPos == 1):

        if est_k != 1:
            # if (est_k == 1):
            #     est_rbt_m[0:2,est_k] = (1/2)*(est_rbt_m[3:5,est_k]+est_rbt_m[3:5,est_k-1])*est_dt 
            # else:
            #     est_rbt_m[0:2,est_k] = est_rbt_x[0:2,est_k-1]+(1/2)*(est_rbt_m[3:5,est_k]+est_rbt_m[3:5,est_k-1])*est_dt


            [xmf, ymf] = rbt2map(
                est_rbt_m[0, est_k],
                est_rbt_m[1, est_k],
                est_rbt_m[0, est_k-1],
                est_rbt_m[1, est_k-1],
                est_angPosArray[0, 0],
                est_map_m[0, est_k-1],
                est_map_m[1, est_k-1]            
            )

            # Save map frame conversion
            est_map_m[0, est_k] = xmf
            est_map_m[1, est_k] = ymf
        else:
            est_map_m[0, est_k] = 0
            est_map_m[1, est_k] = 0                

        est_map_m[0,est_k] = est_map_m[0,est_k]*est_linPosBias
        est_map_m[1,est_k] = est_map_m[1,est_k]*est_linVelBias   

    else:
        est_map_m[0:2, est_k] = matrix([[0], [0]]) 

def estMapMeas():
    # Linear position pseudo measurements 
    if k != t_N:
        est_map_m[0:2, est_k] = est_map_m[0:2, est_k] 

        # est_map_m[0:2, est_k] = est_linPosMapArray[:, 0] 

    # Angular position measurements
    if k != t_N:
        est_map_m[2,est_k] = est_rbt_x[2, est_k+1]   # angular position in map frame 

    # Robot frame linear velocity measurements
    if k != t_N:
        est_map_m[3:5, est_k] = est_rbt_x[3:5,est_k+1]                          

    # Angular velocity measurements   
    if k != t_N:
        est_map_m[5,est_k] = est_rbt_x[5,est_k+1]  # angular velocity in map frame

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

"""  RUN LOOP """
for k in range (0, t_N, 1):
    if k < 48000:

        """ Simulation """
        # Controller
        controller()

        # Simulate robot 
        runSim()

        # Check sensor updates
        if k != t_N:
            # Increment GNSS and set update flag to true if GNSS update should occur
            # if mod(k*dt, 1/dvl_rr) == 0:   # update sensor reading at rr
            if k*dt > (1/dvl_rr)*dvl_counter:
                dvl_k = dvl_k+1   # increment sensor
                dvl_update = True     # set update flag to true
                dvl_lastUpdateTime = t_now # set the last time the sensor updated
                dvl_counter = dvl_counter+1
            else:
                dvl_update = False     # set update flag to false
                
            #  Increment IMU and set update flag to true if IMU update should occur
            # if mod(k*dt, 1/imu_rr) == 0:    # update sensor reading at rr    
            if k*dt > (1/imu_rr)*imu_counter:
                imu_k = imu_k+1     # increment sensor
                imu_update = True      # set update flag to true
                imu_lastUpdateTime = t_now  # set the last time the sensor updated
                imu_counter = imu_counter+1
            else:
                imu_update = False  # set update flag to false

        # Update sensors
        updateSensors()


        """ Estimator """
        # if mod(k*dt, est_dt) > -0.01 or mod(k*dt, est_dt) < 0.01:
        # if mod(k*dt, est_dt) == 0:    # don't want est_dt to trip every single time step because real world, estimator executes at some Hz. If current time is really small
        if k*dt > est_dt*est_counter:   

            # Estimator measurement array
            estMeasArrays()       

            # Linear velocity
            rbtLinVel()

            # # Compute for robot position
            rbtLinPos()

            # Estimator robot measurement            
            estRbtMeas()       

            # Run robot kalman filter
            if k != t_N:
                # if k*dt > est_dt*est_counter:

                x = est_rbt_x[:, est_k]    # last state estimate vector from robot frame
                P = est_rbt_P[:,:,est_k]   # last covariance matrix
                Q = est_rbt_Q              # process noise covariance matrix
                R = est_rbt_R              # measurement noise covariance
                H = est_rbt_H              # observation matrix
                z = est_rbt_m[:,est_k]     # measurement vector
                u = matrix([[0], [0], [0]])  # control input vector (don't give kalman filter knowledge about thruster inputs)

                A = est_rbt_A
                B = est_rbt_B
                   
                state = KalmanFilter(x, P, z, u, A, B, Q, R, H)
                x, P = state.predict(x, P, u)
                x, K, P = state.update(x, P, z)

                    # print('x', x)
                    # # print('P', P)
                    # # print('Q', Q)
                    # # print('R', R)
                    # # print('H', H)
                    # # print('z', z)
                    # # print('u', u)
                    # # print('K', K)

                est_rbt_x[:, est_k+1] = x
                est_rbt_L[:,:, est_k+1] = K
                est_rbt_P[:,:, est_k+1] = P         
           
            # Compute for map position
            mapLinPos()
            
            # Estimator map measurement
            estMapMeas()

            # Run map kalman filter
            if k != t_N:
                
                # if k*dt > est_dt*est_counter:
                    
                x = est_map_x[:,est_k]       # last estimate from robot frame
                P = est_map_P[:,:,est_k]     # last covariance matrix
                Q = est_map_Q                # process noise covariance
                R = est_map_R                # measurement noise covariance
                H = est_map_H                # measurement matrix
                z = est_map_m[:,est_k]       # measurement
                u = matrix([[0], [0], [0]])  # control input vector (don't give kalman filter knowledge about thruster inputs)
                    
                A = est_map_A(est_map_m[:,est_k])
                B = est_map_B
                    
                state = KalmanFilter(x, P, z, u, A, B, Q, R, H)
                x, P = state.predict(x, P, u)
                x, K, P = state.update(x, P, z)

                est_map_x[:,est_k+1] = x
                est_map_L[:,:,est_k+1] = K
                est_map_P[:,:,est_k+1] = P
                    
                    # print('x', x)
                    # print('P', P)
                    # print('Q', Q)
                    # print('R', R)
                    # print('H', H)
                    # print('z', z)
                    # print('u', u)
                    # print('K', K)

                # Increment tracking k variable
                est_k = est_k+1
            
            # increment counter, end of loop
            est_counter = est_counter+1
        
        # ### Save data ###
        simX.append(sim_xm[0, k])
        simY.append(sim_xm[1, k])
        linPosX.append(dvl_aprxLinPosMap[0, dvl_k])
        linPosY.append(dvl_aprxLinPosMap[1, dvl_k])
        estMapX.append(est_map_x[0, est_k])
        estMapY.append(est_map_x[1, est_k])
        
        ### Increment time ###
        t_now = t_now + dt

""" Plots """
# Robot x vs y
genPlots.plotXY(simX, simY, linPosX, linPosY, estMapX, estMapY)


# # Plot Robot position vs time (map frame)
# # Set fonts properties
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# titleFontSize = 20
# defaultFontSize = 12
# markerSize = 0.5

# tt = np.linspace(0, endTime, 48001, endpoint= True)
# ttGnss = np.linspace(0, endTime, 48001, endpoint= True)
# ttImu = np.linspace(0, endTime, 1000, endpoint = True)
# ttEst = np.linspace(0, endTime, 48001, endpoint = True)

# xr = plt.subplot(331)
# plt.plot(tt[:-1], simYr, color = 'k', linestyle = '--', linewidth = 1, markersize = markerSize, label = 'simulated ground truth')
# plt.plot(ttGnss[:-1], gnssLinPosRbtPlot, color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize, label = 'sensor measurement')
# plt.plot(ttEst[:-1], estRbtXPlot, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize, label = 'estimated trajectory')
# plt.title(r'Robot Body Position',fontsize = titleFontSize)
# plt.xlabel(r'time [s]',fontsize = defaultFontSize)
# plt.ylabel(r'$x_r$ [m]',fontsize = defaultFontSize)

# yr = plt.subplot(334)
# plt.plot(tt[:-1], simYr2, color = 'k', linestyle = '--', linewidth = 1, markersize = markerSize, label = 'simulated ground truth')
# plt.plot(ttGnss[:-1], gnssLinPosRbtPlot2, color = 'r', linestyle = 'none', linewidth = 1, marker = 'o', markersize = markerSize, label = 'sensor measurement')
# plt.plot(ttEst[:-1], estRbtXPlot2, color = 'b', linestyle = '-', linewidth = 1, marker = 'o', markersize = markerSize, label = 'estimated trajectory')
# plt.title(r'Robot Body Position',fontsize = titleFontSize)
# plt.xlabel(r'time [s]',fontsize = defaultFontSize)
# plt.ylabel(r'$y_r$ [m]',fontsize = defaultFontSize)


# plt.show()
