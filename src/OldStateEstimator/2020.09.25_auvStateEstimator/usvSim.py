#! /usr/bin/env python

"""
TASKS:
1) Revisit rotations from robot to map frame
2) Test in UUV simulator with ros nodes
"""

from numpy.linalg import inv
from numpy import identity
from numpy import matrix, diag, random, array, zeros, block, mod, reshape
from math import sqrt, cos, sin, atan2, log
from scipy.spatial.transform import Rotation
import math
import matplotlib.pyplot as plt
import numpy as np
import genPlots
from kf import KalmanFilter
from ekf import ExtendedKalmanFilter

from inputs import timeInput, simInputs, senInputs, estInputs
t = timeInput()
sim = simInputs()
sen = senInputs()
est = estInputs()

""" Time Setup """
# Time property calculations
t_N = round(t.end/t.dt) # total number of time steps []
t_now = t.start # current time [s]

""" Simulation Setup """
# State, output, and input vectors
sim_xr = matrix(zeros((12, t_N)))   # simulation state vector in robot reference frame
sim_yr = matrix(zeros((18, t_N)))   # simulation output vector in robot reference frame
sim_xm = matrix(zeros((6, t_N)))    # simulation state vector in map reference frame
sim_u = matrix(zeros((6, t_N)))     # input vector (control inputs act on robot frame)

# Create simulation state matrices
sim_A = matrix([
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],   # x
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],   # y
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],   # z
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],   # roll
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],   # pitch
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],   # yaw
    [0, 0, 0, 0, 0, 0, -sim.bxr/sim.m, 0, 0, 0, 0, 0],  # dx
    [0, 0, 0, 0, 0, 0, 0, -sim.byr/sim.m, 0, 0, 0, 0],  # dy
    [0, 0, 0, 0, 0, 0, 0, 0, -sim.bzr/sim.m, 0, 0, 0],  # dz
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -sim.bir/sim.I, 0, 0],  # droll
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sim.btr/sim.I, 0],  # dpitch
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sim.bpr/sim.I]]) # dyaw  # state matrix[12x12]
sim_B = matrix([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1/sim.m, 0, 0, 0, 0, 0],
    [0, 1/sim.m, 0, 0, 0, 0],
    [0, 0, 1/sim.m, 0, 0, 0],
    [0, 0, 0, 1/sim.I, 0, 0],
    [0, 0, 0, 0, 1/sim.I, 0],
    [0, 0, 0, 0, 0, 1/sim.I]])   # input matrix [12x6]
sim_C = matrix(block([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [sim_A[6:12, :], zeros((6,6))]]))   # output matrix [18x18]
sim_D = matrix(block([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [sim_B[6:12, :]]]))   # feedthrough matrix [18x6] 

""" Sensor Setup """
dvl_aprxLinPosRbt = matrix(zeros((3, t.end*sen.dvl_rr)))   # approximated linear position data from DVL using trapezoidal integration method in robot frame
dvl_aprxLinPosMap = matrix(zeros((3, t.end*sen.dvl_rr)))   # approximated linear position data from DVL using trapezoidal integration method in map frame
dvl_linVel = matrix(zeros((3,t.end*sen.dvl_rr)))  # linear velocity data from DVL
dvl_angPos = matrix(zeros((3,t.end*sen.dvl_rr)))  # angular position data from IMU, references to DVL timestamp
imu_angPos = matrix(zeros((3,t.end*sen.imu_rr)))  # angular position data from IMU
imu_angVel = matrix(zeros((3,t.end*sen.imu_rr)))  # angular velocity from IMU
imu_angVel_n = matrix(zeros((3,t.end*sen.imu_rr)))  # angular velocity from IMU without bias
imu_linAcc = matrix(zeros((3,t.end*sen.imu_rr)))  # linear acceleration data from IMU
imu_linAcc_n = matrix(zeros((3,t.end*sen.imu_rr)))  # linear acceleration data from IMU without bias
imu_acBias = matrix(zeros((3,t.end*sen.imu_rr)))  # acceleration bias
imu_gyBias = matrix(zeros((3,t.end*sen.imu_rr)))  # gyroscope bias
bar_linPos = matrix(zeros((1,t.end*sen.bar_rr)))  # linear z position from barometer sensor

""" Reset Sensor Tracker """
dvl_update = 0  # DVL update tracker (boolean)
imu_update = 0  # IMU update tracker (boolean)
bar_update = 0  # barometer update tracker (boolean)
dvl_k = 1   # DVL iteration tracker
imu_k = 1   # IMU iteration tracker
bar_k = 1   # barometer sensor iteration tracker
dvl_lastUpdateTime = 0  # DVL last update time tracker [s]
imu_lastUpdateTime = 0  # IMU last update time tracker [s]
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

def skewSymmetric(s):
    # x[0] = x, x[1] = y, x[2] = z
    return np.array([[    0, -s[2], s[1]],  
                     [ s[2],     0, -s[0]],
                     [-s[1],  s[0],   0]])

def mapToBodyRot(roll, pitch, yaw):
    R_roll = lambda x: matrix([
        [1,         0,         0],
        [0,  cos(x[6]), sin(x[6])],
        [0, -sin(x[6]), cos(x[6])]
    ])
    R_pitch = lambda x: matrix([
        [cos(x[7]), 0, -sin(x[7])],
        [        0, 1,          0],
        [sin(x[7]), 0,  cos(x[7])]
    ])
    R_yaw = lambda x: matrix([
        [cos(x[8]),  sin(x[8]), 0],
        [-sin(x[8]), cos(x[8]), 0],
        [         0,         0, 1]
    ])
    bodyR = R_roll*R_pitch*R_yaw
    return bodyR.T

def bodyToMap(roll, pitch, yaw):
    R_roll = matrix([
        [1,         0,         0],
        [0,  cos(roll), sin(roll)],
        [0, -sin(roll), cos(roll)]
    ])
    R_pitch = matrix([
        [cos(pitch), 0, -sin(pitch)],
        [        0, 1,          0],
        [sin(pitch), 0,  cos(pitch)]
    ])
    R_yaw = matrix([
        [cos(yaw),  sin(yaw), 0],
        [-sin(yaw), cos(yaw), 0],
        [         0,         0, 1]
    ])
    mapR = R_yaw*R_pitch*R_roll
    return mapR.T

""" Controllers """
# Funky controllers
def controller():
    if (k == 0):
        sim_u[0,k] = 0     # force input in surge direction
        sim_u[1,k] = 0     # force input in sway direction
        sim_u[2,k] = 0     # torque input in heave direction
        sim_u[3,k] = 0     # torque input in roll direction
        sim_u[4,k] = 0     # torque input in pitch direction
        sim_u[5,k] = 0     # torque input in yaw direction
    elif (k > 1 and k < 15000):
        sim_u[0,k] = 300    # force input in surge direction
        sim_u[1,k] = 150    # force input in sway direction
        sim_u[2,k] = 100    # force input in heave direction
        sim_u[3,k] = 50     # torque input in roll direction
        sim_u[4,k] = 50     # torque input in pitch direction
        sim_u[5,k] = 50     # torque input in yaw direction
    elif (k > 1 and k < 30000):
        sim_u[0,k] = -300   # force input in surge direction
        sim_u[1,k] = -150   # force input in sway direction
        sim_u[2,k] = -100    # force input in heave direction
        sim_u[3,k] = -50     # torque input in roll direction
        sim_u[4,k] = -50     # torque input in pitch direction
        sim_u[5,k] = -50     # torque input in yaw direction
    elif (k > 1 and k < 45000):
        sim_u[0,k] = 300  # force input in surge direction
        sim_u[1,k] = 150  # force input in sway direction
        sim_u[2,k] = 50  # force input in heave direction
        sim_u[3,k] = 50     # torque input in roll direction
        sim_u[4,k] = 50     # torque input in pitch direction
        sim_u[5,k] = 50     # torque input in yaw direction
    else:
        sim_u[0,k] = -300   # force input in surge direction
        sim_u[1,k] = 150    # force input in sway direction
        sim_u[2,k] = 100    # force input in heave direction
        sim_u[3,k] = -50     # torque input in roll direction
        sim_u[4,k] = -50     # torque input in pitch direction
        sim_u[5,k] = -50     # torque input in yaw direction

""" Run simulation """
def runSim():
    # Create discrete state matrices for Euler integration
    F = identity(12)+sim_A*t.dt    # discrete state matrix
    G = sim_B*t.dt    # discrete input matrix
    H = sim_C   # discrete output matrix
    J = sim_D   # discrete feedthrough matrix 

    if k != t_N:
        # Simulate plant using discrete Euler integration
        sim_xr[:, k+1] = F*sim_xr[:, k]+G*matrix([[sim_u[0, k]], [sim_u[1, k]], [sim_u[2, k]], [sim_u[3, k]], [sim_u[4, k]], [sim_u[5, k]]]) #  state matrix integration solution 
        sim_yr[:, k+1] = H*block([[sim_xr[:, k+1]], [zeros((6,1))]])+J*matrix([[sim_u[0, k]], [sim_u[1, k]], [sim_u[2, k]], [sim_u[3, k]], [sim_u[4, k]], [sim_u[5, k]]]) # state observer matrix integration solution

        # Convert pose in robot frame to pose in map frame
        [sim_xm[0,k+1], sim_xm[1, k+1]] = rbt2map(sim_xr[0, k+1], sim_xr[1, k+1], sim_xr[0, k], sim_xr[1, k], sim_xr[5, k], sim_xm[0, k], sim_xm[1, k])
        sim_xm[2, k+1] = sim_xr[2, k+1] # z position
        # sim_rot = bodyToMap(sim_xr[3, k+1], sim_xr[4, k+1], sim_xr[5, k+1])
        sim_xm[3, k+1] = sim_xr[3, k+1] # roll
        sim_xm[4, k+1] = sim_xr[4, k+1] # pitch
        sim_xm[5, k+1] = sim_xr[5, k+1] # yaw

""" Sensor Update """
def updateSensors():
    if k != t_N:
        # Add noise to DVL states to simulate sensor data if DVL update should occur
        if dvl_update == 1:
            dvl_linVel[0, dvl_k] = sim_yr[6, k]+np.random.normal(1)*sqrt(sen.dvl_linVelVar)
            dvl_linVel[1, dvl_k] = sim_yr[7, k]+np.random.normal(1)*sqrt(sen.dvl_linVelVar)
            dvl_linVel[2, dvl_k] = sim_yr[8, k]+np.random.normal(1)*sqrt(sen.dvl_linVelVar)
            dvl_angPos[0, dvl_k] = imu_angPos[0, imu_k-1]

        # Add noise to IMU states to simulate sensor data if IMU update should occur
        if imu_update == 1:
            imu_angPos[0, imu_k] = sim_yr[3, k]+np.random.normal(1)*sqrt(sen.imu_angPosVar)
            imu_angPos[1, imu_k] = sim_yr[4, k]+np.random.normal(1)*sqrt(sen.imu_angPosVar)
            imu_angPos[2, imu_k] = sim_yr[5, k]+np.random.normal(1)*sqrt(sen.imu_angPosVar)
            imu_angVel[0, imu_k] = sim_yr[9, k]+np.random.normal(1)*sqrt(sen.imu_angVelVar)
            imu_angVel[1, imu_k] = sim_yr[10, k]+np.random.normal(1)*sqrt(sen.imu_angVelVar)
            imu_angVel[2, imu_k] = sim_yr[11, k]+np.random.normal(1)*sqrt(sen.imu_angVelVar)
            imu_linAcc[0, imu_k] = sim_yr[12, k]+np.random.normal(1)*sqrt(sen.imu_linAccVar)
            imu_linAcc[1, imu_k] = sim_yr[13, k]+np.random.normal(1)*sqrt(sen.imu_linAccVar)
            imu_linAcc[2, imu_k] = sim_yr[14, k]+np.random.normal(1)*sqrt(sen.imu_linAccVar)
            
            # Acceleration bias
            imu_acBias[0, imu_k] = np.random.normal(1)*sqrt(sen.imu_acBiasVar)
            imu_acBias[1, imu_k] = np.random.normal(1)*sqrt(sen.imu_acBiasVar)
            imu_acBias[2, imu_k] = np.random.normal(1)*sqrt(sen.imu_acBiasVar)
            # Remove bias from acceleration
            imu_linAcc_n[0, imu_k] = sim_yr[12, k]+np.random.normal(1)*sqrt(sen.imu_linAccVar)-imu_acBias[0, imu_k]
            imu_linAcc_n[1, imu_k] = sim_yr[13, k]+np.random.normal(1)*sqrt(sen.imu_linAccVar)-imu_acBias[1, imu_k]
            imu_linAcc_n[2, imu_k] = sim_yr[14, k]+np.random.normal(1)*sqrt(sen.imu_linAccVar)-imu_acBias[2, imu_k]

            # Gyro bias
            imu_gyBias[0, imu_k] = np.random.normal(1)*sqrt(sen.imu_gyBiasVar)
            imu_gyBias[1, imu_k] = np.random.normal(1)*sqrt(sen.imu_gyBiasVar)
            imu_gyBias[2, imu_k] = np.random.normal(1)*sqrt(sen.imu_gyBiasVar)
            # Remove bias from gyro
            imu_angVel_n[0, imu_k] = sim_yr[12, k]+np.random.normal(1)*sqrt(sen.imu_linAccVar)-imu_gyBias[0, imu_k]
            imu_angVel_n[1, imu_k] = sim_yr[13, k]+np.random.normal(1)*sqrt(sen.imu_linAccVar)-imu_gyBias[1, imu_k]
            imu_angVel_n[2, imu_k] = sim_yr[14, k]+np.random.normal(1)*sqrt(sen.imu_linAccVar)-imu_gyBias[2, imu_k]
            
            # AHRS (using current IMU AHRS for now)
            # Gyro transformation
            # delOmega = skewSymmetric(imu_angVel)
            # Remove gravity from acceleration data to get lienar acceleration in nav frame (map frame)
            # NED
            # accel_n = Cbn# ENU

        # Add noise to barometer states to simulate sensor data if barometer update should occur
        if bar_update == 1:
            bar_linPos[0, bar_k] = sim_xm[2, k]+np.random.normal(1)*sqrt(sen.bar_linPosVar)

        # DVL approximated robot frame position and convert to map frame (this must occur after previous state)
        if dvl_update == 1:

            if dvl_k != 1:

                # Approximate position in robot frame using trapzeoidal integration method
                if (dvl_k == 1):
                    dvl_aprxLinPosRbt[:,dvl_k] = (1/2)*(dvl_linVel[:,dvl_k] + dvl_linVel[:,dvl_k-1])*(1/sen.dvl_rr)
                else:
                    dvl_aprxLinPosRbt[:,dvl_k] = sim_xr[0:3,k-1]+(1/2)*(dvl_linVel[:,dvl_k]+dvl_linVel[:,dvl_k-1])*(1/sen.dvl_rr)    

                # Convert DVL pose in robot frame to map frames
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
# USER INPUT: robot frame estimator process noise covariance 
est_rbt_Q_linPos = est.rbt_R_linPos*1e1    # linear position [m]
est_rbt_Q_angPos = est.rbt_R_angPos*1e1
est_rbt_Q_linVel = est.rbt_R_linVel*1e1
est_rbt_Q_linAcc = est.rbt_R_linAcc*1e-1   # linear acceleration [m/s^s]
est_rbt_Q_acBias = est.rbt_R_acBias*1e-1
est_rbt_Q_gyBias = est.rbt_R_gyBias*1e-1
            
# USER INPUT: map frame estimator process noise covariance          
est_map_Q_linPos = est.map_R_linPos*1e1    # linear position [m]
est_map_Q_angPos = est.map_R_angPos*1e5    # angular position [rad]
est_map_Q_linVel = est.map_R_linVel*1e5    # linear velocity [m/s]

""" Estimator Setup """
# Instantiate measurement variables and arrays
est_counter = 1 # estimator counter
est_k = 1   # iteration tracking variable
est_dvlLastUpdateTime = 0   # last DVL update time [s]
est_imuLastUpdateTime = 0   # last IMU update time [s]
est_barLastUpdateTime = 0   # last barometer update time [s]
est_dvlSlidingGain = 1  # sliding covariance gain for DVL
est_imuSlidingGain = 1  # sliding covariance gain for IMU
est_barSlidingGain = 1  # sliding covariance gain for barometer
est_dvlImuBarSlidingGain = 1   # sliding covariance gain for DVL+IMU+barometer
est_slidingGain = 1 #   final sliding covariance gain for estimator iteration
est_linPosMapArray = matrix(zeros((3,est.linVel_n)))    # linear position approximated from DVL in map frame measurement array
est_angPosArray = matrix(zeros((3,est.angPos_n)))  # angular position directly from IMU in map and robot frame measurement array
est_linVelRbtArray = matrix(zeros((3,est.linVel_n)))    # linear velocity directly from DVL in robot frame measurement array
est_linAccRbtArray = matrix(zeros((3,est.linAcc_n)))    # linear acceleration directly from IMU in robot frame measurement array
est_linPosDvlAprxMap = matrix(zeros((3,1))) # approximated linear position in map frame from trapezoidal method of DVL
est_linVelImuAprxRbt = matrix(zeros((3,1))) # approximated linear velocity in robot frame from trapezoidal method of IMU
est_acBiasArray = matrix(zeros((3,est.acBias_n)))   # angular velocity directly from IMU in map and robot frame measurement array
est_gyBiasArray = matrix(zeros((3,est.gyBias_n)))   # angular velocity directly from IMU in map and robot frame measurement array

# Instantiate estimator matrices
est_rbt_m = matrix(zeros((18,int(t.end/est.dt))))   # robot frame estimator measurement matrix
est_rbt_x = matrix(zeros((18,int(t.end/est.dt))))   # robot frame estimator state estimate
est_rbt_L = array(zeros((18,18,int(t.end/est.dt)))) # robot frame estimator kalman gain matrix
est_rbt_P = array(zeros((18,18,int(t.end/est.dt)))) # robot frame estimator covariance matrix
est_rbt_s = array(zeros((18,18,int(t.end/est.dt)))) # robot frame sliding covariance gain matrix 

est_map_m = matrix(zeros((9,int(t.end/est.dt))))    # map frame estimator measurement matrix
est_map_x = matrix(zeros((9,int(t.end/est.dt))))    # map frame estimator state estimate
est_map_L = array(zeros((9,9,int(t.end/est.dt))))   # map frame estimator kalman gain matrix
est_map_P = array(zeros((9,9,int(t.end/est.dt))))   # map frame estimator covariance matrix
est_map_s = array(zeros((9,9,int(t.end/est.dt))))   # map frame sliding covariance gain matrix

# gravity = np.array([[0], [0], [9.81]])  # Gravity vector

# Robot frame covariance, obvervation, state and input matrices
est_rbt_Q = matrix([
    [est_rbt_Q_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, est_rbt_Q_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, est_rbt_Q_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, est_rbt_Q_angPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, est_rbt_Q_angPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, est_rbt_Q_angPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, est_rbt_Q_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, est_rbt_Q_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, est_rbt_Q_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_Q_linAcc, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_Q_linAcc, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_Q_linAcc, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_Q_acBias, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_Q_acBias, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_Q_acBias, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_Q_gyBias, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_Q_gyBias, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_Q_gyBias]]) # process noise covariance matrix    
est_rbt_R = matrix([
    [est.rbt_R_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, est.rbt_R_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, est.rbt_R_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, est.rbt_R_angPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, est.rbt_R_angPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, est.rbt_R_angPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, est.rbt_R_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, est.rbt_R_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, est.rbt_R_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, est.rbt_R_linAcc, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est.rbt_R_linAcc, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est.rbt_R_linAcc, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est.rbt_R_acBias, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est.rbt_R_acBias, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est.rbt_R_acBias, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est.rbt_R_gyBias, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est.rbt_R_gyBias, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est.rbt_R_gyBias]])      # measurement noise covariance matrix
est_rbt_H = matrix([
    [est.H_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, est.H_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, est.H_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, est.H_angPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, est.H_angPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, est.H_angPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, est.H_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, est.H_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, est.H_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, est.H_linAcc, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est.H_linAcc, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est.H_linAcc, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est.H_acBias, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est.H_acBias, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est.H_acBias, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est.H_gyBias, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est.H_gyBias, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est.H_gyBias]])       # obervation matrix
est_rbt_A = matrix([
    [1, 0, 0, est.dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, est.dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, est.dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 1, 0, 0, est.dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, est.dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, est.dt, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]) # state matrix
est_rbt_B = matrix(zeros((18,3)))                   # input matrix            
            
# Map frame covariance, obvervation, state and input matrices
est_map_Q = matrix([
    [1, 0, 0, est.dt, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, est.dt, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, est.dt, 0, 0, 0], 
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]])             # process noise covariance matrix    
est_map_R = matrix([
    [est.map_R_linPos, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, est.map_R_linPos, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, est.map_R_linPos, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, est.map_R_angPos, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, est.map_R_angPos, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, est.map_R_angPos, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, est.map_R_linVel, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, est.map_R_linVel, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, est.map_R_linVel]])  # measurement noise covariance matrix
est_map_H = matrix([
    [est.H_linPos, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, est.H_linPos, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, est.H_linPos, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, est.H_angPos, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, est.H_angPos, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, est.H_angPos, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, est.H_linVel, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, est.H_linVel, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, est.H_linVel]])      # obervation matrix 
# est_map_A = lambda x: matrix([  # 0-x, 1-y, 2-z, 3-vx, 4-vy, 5-vz, 6-roll, 7-pitch, 8-yaw 
#     [1, 0, 0, cos(x[8])*cos(x[7])*est.dt, -sin(x[8])*cos(x[6])*est.dt+cos(x[8])*sin(x[7])*sin(x[6])*est.dt, sin(x[8])*sin(x[6])*est.dt+cos(x[8])*sin(x[7])*cos(x[6])*est.dt, 0, 0, 0],
#     [0, 1, 0, sin(x[8])*cos(x[7])*est.dt, cos(x[8])*cos(x[6])*est.dt+sin(x[8])*sin(x[7])*sin(x[6])*est.dt, -cos(x[8])*sin(x[6])*est.dt+sin(x[8])*sin(x[7])*cos(x[6])*est.dt, 0, 0, 0],
#     [0, 0, 1, -sin(x[7]), cos(x[7])*sin(x[6]), cos(x[7])*cos(x[6]), 0, 0, 0], 
#     [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1]])  # state matrix
est_map_A = lambda x: matrix([  # 0-x, 1-y, 2-z, 3-vx, 4-vy, 5-vz, 6-roll, 7-pitch, 8-yaw 
    [1, 0, 0, cos(x[8])*cos(x[7])*est.dt, -sin(x[8])*cos(x[6])*est.dt+cos(x[8])*sin(x[7])*sin(x[6])*est.dt, sin(x[8])*sin(x[6])*est.dt+cos(x[8])*sin(x[7])*cos(x[6])*est.dt, 0, 0, 0],
    [0, 1, 0, sin(x[8])*cos(x[7])*est.dt, cos(x[8])*cos(x[6])*est.dt+sin(x[8])*sin(x[7])*sin(x[6])*est.dt, -cos(x[8])*sin(x[6])*est.dt+sin(x[8])*sin(x[7])*cos(x[6])*est.dt, 0, 0, 0],
    [0, 0, 1, -sin(x[7]), cos(x[7])*sin(x[6]), cos(x[7])*cos(x[6]), 0, 0, 0], 
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]])  # state matrix
est_map_B = matrix(zeros((9,3)))    # input matrix

def estRbtMeas():
    # # Position measurements and gains
    # est_rbt_m[0:2,est_k] = est_rbt_m[0:2,est_k]    # linear position in robot frame        

    # # Barometer measurement and gains
    # est_rbt_m[2,est_k] = est_linPosMapArray[2,0]   # linear z position in robot frame

    # Rotation measurements and gains
    est_rbt_m[3:6,est_k] = est_angPosArray[:,0]   # angular position in robot frame    

    # Linear velocity measurements and gains            
    # est_rbt_m[6:9,est_k] = est_rbt_m[6:9,est_k]   # linear velocity in robot frame  
    est_rbt_m[6:9,est_k] = est_linVelRbtArray[:,0]   # linear velocity in robot frame  

    # Linear acceleration measurements and gains
    est_rbt_m[9:12,est_k] = est_linAccRbtArray[:,0]    # linear acceleration in robot frame     

    # IMU acceleration bias measurements and gains
    est_rbt_m[12:15,est_k] = est_acBiasArray[:,0]    # linear acceleration in robot frame     

    # IMU gyroscope bias measurements and gains
    est_rbt_m[15:18,est_k] = est_gyBiasArray[:,0]    # angular velocity in robot frame     

def rbtLinVel():
    # Takes the given velocity data from the DVL
    if (est.useDvlLinVel == 1 and est.useImuLinVel == 0):
        # save robot frame velocity to measurement matrix
        est_rbt_m[6:9,est_k] = est_linVelRbtArray[:,0]  

    # # Takes only the trapezoidal integration method from the IMU
    # if (est.useDvlLinVel == 0 and est.useImuLinVel == 1):
    #     if (est_k == 1):
    #         est_linVelImuAprxRbt[:,0] = (1/2)*(est_linAccRbtArray[:,1]+est_linAccRbtArray[:,0])*(1/sen.imu_rr)                         
    #     else:
    #         est_linVelImuAprxRbt[:,0] = est_rbt_x[6:9,est_k-1]+(1/2)*(est_linAccRbtArray[:,1]+est_linAccRbtArray[:,0])*(1/sen.imu_rr)                                     
               
    #     # save robot frame velocity to measurement matrix
    #     est_rbt_m[6:9,est_k] = est_linVelImuAprxRbt[:,0]*est.linVelBias 

    # # Takes both the velocity data from DVL and integrated IMU data
    # elif (est.useDvlLinVel == 1 and est.useImuLinVel == 1):
    #     if (est_k == 1):
    #         est_linVelImuAprxRbt[:,0] = (1/2)*(est_linAccRbtArray[:,1]+est_linAccRbtArray[:,0])*(1/sen.imu_rr)                         
    #     else:
    #         est_linVelImuAprxRbt[:,0] = est_rbt_x[6:9,est_k-1]+(1/2)*(est_linAccRbtArray[:,1]+est_linAccRbtArray[:,0])*(1/sen.imu_rr)                                     
               
    #     # save robot frame velocity to measurement matrix
    #     est_rbt_m[6,est_k] = (est_linVelRbtArray[0,0]+est_linVelImuAprxRbt[0,0])/2*est.linVelBias 
    #     est_rbt_m[7,est_k] = (est_linVelRbtArray[1,0]+est_linVelImuAprxRbt[1,0])/2*est.linVelBias 
    #     est_rbt_m[8,est_k] = (est_linVelRbtArray[2,0]+est_linVelImuAprxRbt[2,0])/2*est.linVelBias 

    # If condition throws an error
    else:
        est_rbt_m[6:9, est_k] = matrix([[0], [0], [0]]) 
    
def mapLinPos():
    # Use dvl to get position

    if (est.useDvlLinVel == 1 and est.useImuLinVel == 1):
        est_map_m[2, est_k] = est_linPosMapArray[2, 0]*est.linPosBias
    #     if (est_k != 1):
    #         if (est_k == 1):
    #             est_rbt_m[0:2,est_k] = (1/2)*(est_rbt_m[3:5,est_k]+est_rbt_m[3:5,est_k-1])*est_dt                        
    #         else:
    #             est_rbt_m[0:2,est_k] = est_rbt_x[0:2,est_k-1]+(1/2)*(est_rbt_m[3:5,est_k]+est_rbt_m[3:5,est_k-1])*est_dt 
                    
    #         # Convert DVL pose in robot frame to map frames
    #         [xmf, ymf] = rbt2map(
    #             est_rbt_m[0, est_k],
    #             est_rbt_m[1, est_k],
    #             est_rbt_m[0, est_k-1],
    #             est_rbt_m[1, est_k-1],
    #             est_rbt_m[6, est_k],
    #             est_map_m[0, est_k-1],
    #             est_map_m[1, est_k-1]
    #         )

    #         # Save map frame conversion
    #         est_map_m[0, est_k] = xmf
    #         est_map_m[1, est_k] = ymf
    #     else:
    #         est_map_m[0, est_k] = 0
    #         est_map_m[1, est_k] = 0        
    else:
        est_map_m[0:2, est_k] = matrix([[0], [0], [0]]) 

def estMapMeas():
    # Linear position pseudo measurements 
    # est_map_m[0:2, est_k] = est_map_m[0:2, est_k] 
    est_map_m[0:2,est_k] = est_linPosMapArray[0:2,0]    # linear x and y position in map frame

    est_map_m[2,est_k] = est_linPosMapArray[2,0]    # linear z position in map frame

   # Rotation measurements
    if k != t_N:
        est_map_m[3:6,est_k] = est_rbt_x[3:6,est_k+1]   # rotation in map frame 
    
    # Robot frame linear velocity measurements
    if k != t_N:
        est_map_m[6:9,est_k] = est_rbt_x[6:9,est_k+1]   # linear velocity in map frame  

""" Store data """
# Simulation paramters
simXMap = []
simYMap = []
simZMap = []
simRollMap = []
simPitchMap = []
simYawMap = []

simRollRbt = []
simPitchRbt = []
simYawRbt = []
simVelXRbt = []
simVelYRbt = []
simVelZRbt = []
simAcBiasX = []
simAcBiasY = []
simAcBiasZ = []
simGyBiasX = []
simGyBiasY = []
simGyBiasZ = []

# Sensor measurements
linPosX = []
linPosY = []
linPosZ = []
rollMap = []
pitchMap = []
yawMap = []
roll = []
pitch = []
yaw = []
linVelX = []
linVelY = []
linVelZ = []
acBiasX = []
acBiasY = []
acBiasZ = []
gyBiasX = []
gyBiasY = []
gyBiasZ = []

# Estimator parameters
estXMap = []
estYMap = []
estZMap = []
estRollMap = []
estPitchMap = []
estYawMap = []
estVelXRbt = []
estVelYRbt = []
estVelZRbt = []
estAcBiasX = []
estAcBiasY = []
estAcBiasZ = []
estGyBiasX = []
estGyBiasY = []
estGyBiasZ = []

# counters
imu_counter = 1
dvl_counter = 1
bar_counter = 1

# time = np.linspace(0, end, t_N)
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
            # Increment DVL and set update flag to true if DVL update should occur
            if k*t.dt > (1/sen.dvl_rr)*dvl_counter:
                dvl_k = dvl_k+1 # increment sensor
                dvl_update = 1  # set update flag to true
                dvl_lastUpdateTime = t_now # set the last time the sensor updated
                dvl_counter = dvl_counter+1 # increment counter
            else:
                dvl_update = 0  # set update flag to false
                    
            #  Increment IMU and set update flag to true if IMU update should occur
            if k*t.dt > (1/sen.imu_rr)*imu_counter:
                imu_k = imu_k+1 # increment sensor
                imu_update = 1  # set update flag to true
                imu_lastUpdateTime = t_now  # set the last time the sensor updated
                imu_counter = imu_counter+1 # increment counter
            else:
                imu_update = 0  # set update flag to false
            
            #  Increment barometer and set update flag to true if barometer update should occur
            if k*t.dt > (1/sen.bar_rr)*bar_counter:
                bar_k = bar_k+1 # increment sensor
                bar_update = 1  # set update flag to true
                bar_lastUpdateTime = t_now  # set the last time the sensor updated
                bar_counter = bar_counter+1 # increment counter
            else:
                bar_update = 0  # set update flag to false

        # Update sensors
        updateSensors()

        """ Estimator """
        if k*t.dt > est.dt*est_counter:   

            # Estimator measurement array
            # estMeasArrays()       
            # If DVL updated, save data to measurement arrays
            if dvl_update == 1:
                    
                # Set estimator DVL last known sensor update to 
                # the sensor last known update
                est_dvlLastUpdateTime = dvl_lastUpdateTime

                # Update the linear position in the map frame measurement array
                est_linPosMapArray[0:2,1:] = est_linPosMapArray[0:2,:-1]
                est_linPosMapArray[0:2,0] = dvl_aprxLinPosMap[0:2,dvl_k]

                # Update the linear velocity in the robot frame measurement array
                est_linVelRbtArray[:,1:] = est_linVelRbtArray[:,:-1]
                est_linVelRbtArray[:,0] = dvl_linVel[:,dvl_k]

            # If IMU updated, save data to measurement arrays
            if imu_update == 1:

                # Set estimator IMU last known sensor update to 
                # the sensor last known update
                est_imuLastUpdateTime = imu_lastUpdateTime

                # Update the angular position (in map and robot frame) measurement array
                est_angPosArray[:,1:] = est_angPosArray[:,:-1]
                est_angPosArray[:,0] = imu_angPos[:,imu_k]
                        
                # Update the linear acceleration in robot frame measurement array
                est_linAccRbtArray[:,1:] = est_linAccRbtArray[:, :-1]
                est_linAccRbtArray[:,0] = imu_linAcc[:,imu_k]   

                # Update the acceleration bias in robot frame measurement array
                est_acBiasArray[:,1:] = est_acBiasArray[:, :-1]
                est_acBiasArray[:,0] = imu_acBias[:,imu_k]   

                # Update the gyroscope bias in est_gyBiasArray frame measurement array
                est_gyBiasArray[:,1:] = est_gyBiasArray[:, :-1]
                est_gyBiasArray[:,0] = imu_gyBias[:,imu_k]
            
            # If barometer updated, save data to measurement arrays
            if bar_update == 1:

                # Set estimator barometer last known sensor update to 
                # the sensor last known update
                est_barLastUpdateTime = bar_lastUpdateTime
                
                # Update the linear position in the map frame measurement array
                est_linPosMapArray[2,1:] = est_linPosMapArray[2,:-1]
                est_linPosMapArray[2,0] = bar_linPos[0,bar_k]

            # Estimator robot measurement            
            estRbtMeas()  

            # Linear velocity
            # rbtLinVel()   

            # Map linear position
            # mapLinPos()

            # Run robot kalman filter
            if k != t_N:
                # if k*dt > est_dt*est_counter:
                x = est_rbt_x[:,est_k]  # last state estimate vector from robot frame
                P = est_rbt_P[:,:,est_k] # last covariance matrix
                Q = est_rbt_Q # process noise covariance matrix
                R = est_rbt_R # measurement noise covariance
                H = est_rbt_H # observation matrix
                z = est_rbt_m[:,est_k]  # measurement vector
                u = matrix([[0], [0], [0]]) # control input vector (don't give kalman filter knowledge about thruster inputs)

                # Discrete EKF
                A = est_rbt_A
                B = est_rbt_B
                    
                state = ExtendedKalmanFilter(x, P, z, u, A, B, Q, R, H)
                x, P = state.predict(x, P, u)
                x, K, P = state.update(x, P, z)

                # Store state estimate
                est_rbt_x[:, est_k+1] = x
                est_rbt_L[:,:, est_k+1] = K
                est_rbt_P[:,:, est_k+1] = P         
                
            # Estimator map measurement
            estMapMeas()  

            # Map linear position
            # mapLinPos()
                
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
                        
                state = ExtendedKalmanFilter(x, P, z, u, A, B, Q, R, H)
                x, P = state.predict(x, P, u)
                x, K, P = state.update(x, P, z)

                est_map_x[:,est_k+1] = x
                est_map_L[:,:,est_k+1] = K
                est_map_P[:,:,est_k+1] = P

                # Increment tracking k variable
                est_k = est_k+1
                
            # increment counter, end of loop
            est_counter = est_counter+1
            
        ### Save data ###
        # Simulation parameters
        simXMap.append(sim_xm[0,k])
        simYMap.append(sim_xm[1,k])
        simZMap.append(sim_xm[2,k])
        simRollMap.append(sim_xm[3,k])
        simPitchMap.append(sim_xm[4,k])
        simYawMap.append(sim_xm[5,k])

        simRollRbt.append(sim_yr[3,k])
        simPitchRbt.append(sim_yr[4,k])
        simYawRbt.append(sim_yr[5,k])
        simVelXRbt.append(sim_yr[6,k])
        simVelYRbt.append(sim_yr[7,k])
        simVelZRbt.append(sim_yr[8,k])

        # Sensor measurements
        linPosX.append(dvl_aprxLinPosMap[0,dvl_k])
        linPosY.append(dvl_aprxLinPosMap[1,dvl_k])
        linPosZ.append(bar_linPos[0,bar_k])
        roll.append(imu_angPos[0, imu_k])
        pitch.append(imu_angPos[1, imu_k])
        yaw.append(imu_angPos[2, imu_k])
        linVelX.append(dvl_linVel[0,dvl_k])
        linVelY.append(dvl_linVel[1,dvl_k])
        linVelZ.append(dvl_linVel[2,dvl_k])
        acBiasX.append(imu_acBias[0, imu_k])
        acBiasY.append(imu_acBias[1, imu_k])
        acBiasZ.append(imu_acBias[2, imu_k])
        gyBiasX.append(imu_gyBias[0, imu_k])
        gyBiasY.append(imu_gyBias[1, imu_k])
        gyBiasZ.append(imu_gyBias[2, imu_k])

        # Estimator parameters
        estXMap.append(est_map_x[0,est_k])
        estYMap.append(est_map_x[1,est_k])
        estZMap.append(est_map_x[2,est_k])
        estRollMap.append(est_map_x[3, est_k])
        estPitchMap.append(est_map_x[4, est_k])
        estYawMap.append(est_map_x[5, est_k])
        estVelXRbt.append(est_rbt_x[6,est_k])
        estVelYRbt.append(est_rbt_x[7,est_k])
        estVelZRbt.append(est_rbt_x[8,est_k])
        estAcBiasX.append(est_rbt_x[12,est_k])
        estAcBiasY.append(est_rbt_x[13,est_k])
        estAcBiasZ.append(est_rbt_x[14,est_k])
        estGyBiasX.append(est_rbt_x[15,est_k])
        estGyBiasY.append(est_rbt_x[16,est_k])
        estGyBiasZ.append(est_rbt_x[17,est_k])

        # Increment time 
        t_now = t_now + t.dt

""" Plots """
genPlots.plotXY(simXMap, simYMap, linPosX, linPosY, estXMap, estYMap)
genPlots.plotMap(t_N, t.end, simXMap, simYMap, simZMap, simRollMap, simPitchMap, simYawMap, linPosX, linPosY, linPosZ, rollMap, pitchMap, yawMap, estXMap, estYMap, estZMap, estRollMap, estPitchMap, estYawMap)
genPlots.plotRbt(t_N, t.end, simRollRbt, simPitchRbt, simYawRbt, simVelXRbt, simVelYRbt, simVelZRbt, roll, pitch, yaw, linVelX, linVelY, linVelZ, estVelXRbt, estVelYRbt, estVelZRbt)
genPlots.plotBias(t_N, t.end, acBiasX, acBiasY, acBiasZ, gyBiasX, gyBiasY, gyBiasZ, estAcBiasX, estAcBiasY, estAcBiasZ, estGyBiasX, estGyBiasY, estGyBiasZ)

plt.show()

# if __name__ == "__main__":
#     # execute only if run as a script
#     main()