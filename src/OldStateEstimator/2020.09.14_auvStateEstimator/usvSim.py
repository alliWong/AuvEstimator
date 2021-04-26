#! /usr/bin/env python

"""
TASKS:
1) Implement EKF 
2) Change simulation state space into 3D
2) Add pressure sensor
3) Rotation treament (gyro bias)
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

from inputs import get_sim

sim = get_sim()

""" Simulation User Inputs """
# USER INPUT: time properties
sim.dt = 0.001      # simulation time step [s]
startTime = 0   # simulation start time [s]
endTime = 50    # simulation end time [s]

# USER INPUT: underwater vehicle lumped parameters
sim_m = 225     # mass/inertia [kg]
sim_I = 100     # rotational moment of inertia [kg*m^2]
sim_bxr = 40    # drag in the surge (robot frame along x) direction [N*s/m]
sim_byr = 400   # drag in the sway (robot frame along y) direction [N*s/m]
sim_bzr = 10   # drag in heave (robot frame along z) direction [N*s/m]
sim_bir = 300   # rotational drag in the roll (robot frame about x) direction [N*s/m]
sim_btr = 300   # rotational drag in the pitch (robot frame about y) direction [N*s/m]
sim_bpr = 300   # rotational drag in the yaw (robot frame about z) direction [N*s/m]

# USER INPUT: sensor setup
dvl_rr = 20             # refresh rate of DVL sensor [Hz]
dvl_linVelVar = 1      # variance of the DVL sensor data [m/s]
imu_rr = 50            # refresh rate of IMU sensor [Hz]
imu_angRotVar = 0.05   # variance of the IMU angular position sensor data [rad]
imu_linAccVar = 0.1   # variance of the IMU linear acceleration sensor data [m/s^2]
# imu_angVelVar = 0.05   # variance of the IMU angular velocity sensor data [rad/s]
imu_acBiasVar = 0.1

# Barometer
bar_rr = 1
bar_linPosVar = 1

""" Time Properties """
t_N = round(endTime/sim.dt) # total number of time steps []
t_now = startTime       # current time [s]

""" Simulation Setup """
# State, output, and input vectors
sim_xr = matrix(zeros((12, t_N)))    # simulation state vector in robot reference frame
sim_yr = matrix(zeros((18, t_N)))   # simulation output vector in robot reference frame
sim_xm = matrix(zeros((6, t_N)))    # simulation state vector in map reference frame
sim_u = matrix(zeros((6, t_N)))     # input vector (control inputs act on robot frame)

# Create simulation state matrices
sim_A = matrix([
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],   
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],   
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],   
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],   
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],   
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],   
    [0, 0, 0, 0, 0, 0, -sim_bxr/sim_m, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -sim_byr/sim_m, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -sim_bzr/sim_m, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -sim_bir/sim_I, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sim_btr/sim_I, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -sim_bpr/sim_I]])   # state matrix [12x12]
sim_B = matrix([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1/sim_m, 0, 0, 0, 0, 0],
    [0, 1/sim_m, 0, 0, 0, 0],
    [0, 0, 1/sim_m, 0, 0, 0],
    [0, 0, 0, 1/sim_I, 0, 0],
    [0, 0, 0, 0, 1/sim_I, 0],
    [0, 0, 0, 0, 0, 1/sim_I]])   # input matrix [12x6]
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
    F = identity(12)+sim_A*sim.dt    # discrete state matrix
    G = sim_B*sim.dt    # discrete input matrix
    H = sim_C   # discrete output matrix
    J = sim_D   # discrete feedthrough matrix 

    if k != t_N:
        # Simulate plant using discrete Euler integration
        sim_xr[:, k+1] = F*sim_xr[:, k]+G*matrix([[sim_u[0, k]], [sim_u[1, k]], [sim_u[2, k]], [sim_u[3, k]], [sim_u[4, k]], [sim_u[5, k]]]) #  state matrix integration solution 
        sim_yr[:, k+1] = H*block([[sim_xr[:, k+1]], [zeros((6,1))]])+J*matrix([[sim_u[0, k]], [sim_u[1, k]], [sim_u[2, k]], [sim_u[3, k]], [sim_u[4, k]], [sim_u[5, k]]])    # state observer matrix integration solution

        # Convert pose in robot frame to pose in map frame
        [sim_xm[0,k+1], sim_xm[1, k+1]] = rbt2map(sim_xr[0, k+1], sim_xr[1, k+1], sim_xr[0, k], sim_xr[1, k], sim_xr[5, k], sim_xm[0, k], sim_xm[1, k])
        sim_xm[2, k+1] = sim_xr[2, k+1]
        sim_xm[3, k+1] = sim_xr[3, k+1] # roll
        sim_xm[4, k+1] = sim_xr[4, k+1] # pitch
        sim_xm[5, k+1] = sim_xr[5, k+1] # yaw

def updateSensors():
    if k != t_N:
        # Add noise to DVL states to simulate sensor data if DVL update should occur
        if dvl_update == 1:
            dvl_linVel[0, dvl_k] = sim_yr[6, k]+np.random.normal(1)*sqrt(dvl_linVelVar)
            dvl_linVel[1, dvl_k] = sim_yr[7, k]+np.random.normal(1)*sqrt(dvl_linVelVar)
            dvl_linVel[2, dvl_k] = sim_yr[8, k]+np.random.normal(1)*sqrt(dvl_linVelVar)
            dvl_angRot[0, dvl_k] = imu_angRot[0, imu_k-1]

        # Add noise to IMU states to simulate sensor data if IMU update should occur
        if imu_update == 1:
            imu_angRot[0, imu_k] = sim_yr[3, k]+np.random.normal(1)*sqrt(imu_angRotVar)
            imu_angRot[1, imu_k] = sim_yr[4, k]+np.random.normal(1)*sqrt(imu_angRotVar)
            imu_angRot[2, imu_k] = sim_yr[5, k]+np.random.normal(1)*sqrt(imu_angRotVar)
            imu_linAcc[0, imu_k] = sim_yr[12, k]+np.random.normal(1)*sqrt(imu_linAccVar)
            imu_linAcc[1, imu_k] = sim_yr[13, k]+np.random.normal(1)*sqrt(imu_linAccVar)
            imu_linAcc[2, imu_k] = sim_yr[14, k]+np.random.normal(1)*sqrt(imu_linAccVar)

            # # Remove bias from inputs (F=ma -> a = F/m)
            imu_acBias[0, imu_k] = (sim_u[0, k]/sim_m)+np.random.normal(1)*sqrt(imu_acBiasVar)
            imu_acBias[1, imu_k] = (sim_u[1, k]/sim_m)+np.random.normal(1)*sqrt(imu_acBiasVar)
            imu_acBias[2, imu_k] = (sim_u[2, k]/sim_m)+np.random.normal(1)*sqrt(imu_acBiasVar)
            # imu_angVel[0, imu_k] = sim_yr[7, k]+np.random.normal(1)*sqrt(imu_angVelVar)
            
        # Add noise to barometer states to simulate sensor data if barometer update should occur
        if bar_update == 1:
            bar_linPos[0, bar_k] = sim_xm[2, k]+np.random.normal(1)*sqrt(bar_linPosVar)

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

""" Estimater User Inputs """
# USER INPUT: time step properties
est_dt = 0.025  # estimator time step [s]

# USER INPUT: Number of past sensor updates to store for calculating the incremental moving mean of sensor data
est_angRot_n = 2    # angular position mean array size (must be 2 or more)
est_linVel_n = 2    # linear velocity mean array size (must be 2 or more)
est_linAcc_n = 2    # linear acceleration mean array size (must be 2 or more)
est_acBias_n = 2    
est_gyBias_n = 2

# USER INPUT: decide which sensors to use for pseudo linear velocity calculation
est_useDvlLinPos = 1   # set to 1 to use DVL trapezoidal method, set to 0 to ignore
est_useImuLinPos = 1   # set to 1 to use IMU trapezoidal method, set to 0 to ignore
est_useDvlLinVel = 1   # set to 1 to use DVL trapezoidal method, set to 0 to ignore
est_useImuLinVel = 1   # set to 1 to use IMU trapezoidal method, set to 0 to ignore
est_linPosBias = 1.25   # bias (multiplier) on linear velocity output
est_linVelBias = 1.25   # bias (multiplier) on position output

# USER INPUT: robot frame estimator measurement noise covariance
est_rbt_R_linPos = 2       # linear position [m]
est_rbt_R_linVel = 2       # linear velocity [m/s]
est_rbt_R_angRot = 2       # rotation [rad]
est_rbt_R_linAcc = 0.25    # linear acceleration [m/s^s]
est_rbt_R_acBias = 2       # acceleration bias [m^2/s]
est_rbt_R_gyBias = 2       # gyro bias

# USER INPUT: robot frame estimator process noise covariance
est_rbt_Q_linPos = est_rbt_R_linPos*1e2    # linear position [m]
est_rbt_Q_linVel = est_rbt_R_linVel*1e2
est_rbt_Q_angRot = est_rbt_R_angRot*1e-1
est_rbt_Q_linAcc = est_rbt_R_linAcc*1e-1   # linear acceleration [m/s^s]
est_rbt_Q_acBias = est_rbt_R_acBias*1e-1
est_rbt_Q_gyBias = est_rbt_R_gyBias*1e-1

# USER INPUT: H jacobian robot frame estimator observation/observability
est_H_linPos = 1       # linear position (pseudo-measured)
est_H_linVel = 1       # linear velocity (measured)
est_H_angRot = 1       # angular velocity (measured)
est_H_linAcc = 1       # linear acceleration (measured)
est_H_acBias = 1       # acceleration bias (measured)
est_H_gyBias = 1       # gyro bias (measured)
            
# USER INPUT: map frame estimator measurement noise covariance
est_map_R_linPos = 2       # linear position [m]
est_map_R_linVel = 1       # linear velocity [m/s]
est_map_R_angRot = 0.075   # angular position [rad]
            
# USER INPUT: map frame estimator process noise covariance          
est_map_Q_linPos = est_map_R_linPos*1e1    # linear position [m]
est_map_Q_linVel = est_map_R_linVel*1e1    # linear velocity [m/s]
est_map_Q_angRot = est_map_R_angRot*1e1    # angular position [rad]
            
# USER INPUT: H jacobian map frame estimator observation/observability
est_H_linPos = 1       # linear position (pseudo-measured)
est_H_linVel = 1       # linear velocity (measured)
est_H_angRot = 1       # angular position (measured)

""" Estimator Setup """
# Instantiate measurement variables and arrays
est_counter = 1
est_k = 1                                      # iteration tracking variable
est_dvlLastUpdateTime = 0                      # last DVL update time [s]
est_imuLastUpdateTime = 0                      # last IMU update time [s]
est_dvlSlidingGain = 1
est_imuSlidingGain = 1
est_barSlidingGain = 1
est_dvlImuSlidingGain = 1
est_slidingGain = 1

est_linPosMapArray = matrix(zeros((3,est_linVel_n)))    # linear position approximated from DVL in map frame measurement array
est_linVelRbtArray = matrix(zeros((3,est_linVel_n)))    # linear velocity directly from DVL in robot frame measurement array
est_linAccRbtArray = matrix(zeros((3,est_linAcc_n)))    # linear acceleration directly from IMU in robot frame measurement array
est_linPosDvlAprxMap = matrix(zeros((3,1)))     # approximated linear position in map frame from trapezoidal method of DVL
est_linVelImuAprxRbt = matrix(zeros((3,1)))     # approximated linear velocity in robot frame from trapezoidal method of IMU

# est_angRotMapArray = matrix(zeros((3,est_angRot_n)))       # angular position directly from IMU in map and robot frame measurement array
est_angRotArray = matrix(zeros((3,est_angRot_n)))       # angular position directly from IMU in map and robot frame measurement array
est_acBiasArray = matrix(zeros((3,est_acBias_n)))       # angular velocity directly from IMU in map and robot frame measurement array
est_gyBiasArray = matrix(zeros((3,est_gyBias_n)))       # angular velocity directly from IMU in map and robot frame measurement array

# Instantiate estimator matrices
est_rbt_m = matrix(zeros((18,int(endTime/est_dt))))              # robot frame estimator measurement matrix
est_rbt_x = matrix(zeros((18,int(endTime/est_dt))))              # robot frame estimator state estimate
est_rbt_L = array(zeros((18,18,int(endTime/est_dt))))            # robot frame estimator kalman gain matrix
est_rbt_P = array(zeros((18,18,int(endTime/est_dt))))            # robot frame estimator covariance matrix
est_rbt_s = array(zeros((18,18,int(endTime/est_dt))))

est_map_m = matrix(zeros((9,int(endTime/est_dt))))              # map frame estimator measurement matrix
est_map_x = matrix(zeros((9,int(endTime/est_dt))))              # map frame estimator state estimate
est_map_L = array(zeros((9,9,int(endTime/est_dt))))            # map frame estimator kalman gain matrix
est_map_P = array(zeros((9,9,int(endTime/est_dt))))            # map frame estimator covariance matrix
est_map_s = array(zeros((9,9,int(endTime/est_dt))))

# Gravity vector
gravity = np.array([[0], [0], [9.81]])

# Robot frame covariance, obvervation, state and input matrices
est_rbt_Q = matrix([
    [est_rbt_Q_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, est_rbt_Q_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, est_rbt_Q_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, est_rbt_Q_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, est_rbt_Q_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, est_rbt_Q_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, est_rbt_Q_angRot, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, est_rbt_Q_angRot, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, est_rbt_Q_angRot, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
    [est_rbt_R_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, est_rbt_R_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, est_rbt_R_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, est_rbt_R_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, est_rbt_R_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, est_rbt_R_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, est_rbt_R_angRot, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, est_rbt_R_angRot, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, est_rbt_R_angRot, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_R_linAcc, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_R_linAcc, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_R_linAcc, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_R_acBias, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_R_acBias, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_R_acBias, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_R_gyBias, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_R_gyBias, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_rbt_R_gyBias]])      # measurement noise covariance matrix
est_rbt_H = matrix([
    [est_H_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, est_H_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, est_H_linPos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, est_H_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, est_H_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, est_H_linVel, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, est_H_angRot, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, est_H_angRot, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, est_H_angRot, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, est_H_linAcc, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_H_linAcc, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_H_linAcc, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_H_acBias, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_H_acBias, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_H_acBias, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_H_gyBias, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_H_gyBias, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, est_H_gyBias]])       # obervation matrix
est_rbt_A = matrix([
    [1, 0, 0, est_dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, est_dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, est_dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 1, 0, 0, est_dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, est_dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, est_dt, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])                # state matrix
est_rbt_B = matrix(zeros((18,3)))                   # input matrix            
            
# Map frame covariance, obvervation, state and input matrices
est_map_Q = matrix([
    [1, 0, 0, est_dt, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, est_dt, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, est_dt, 0, 0, 0], 
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]])             # process noise covariance matrix    
est_map_R = matrix([
    [est_map_R_linPos, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, est_map_R_linPos, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, est_map_R_linPos, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, est_map_R_linVel, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, est_map_R_linVel, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, est_map_R_linVel, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, est_map_R_angRot, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, est_map_R_angRot, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, est_map_R_angRot]])  # measurement noise covariance matrix
est_map_H = matrix([
    [est_H_linPos, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, est_H_linPos, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, est_H_linPos, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, est_H_linVel, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, est_H_linVel, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, est_H_linVel, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, est_H_angRot, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, est_H_angRot, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, est_H_angRot]])      # obervation matrix 
# est_map_A = lambda x: matrix([
#     [1, 0, 0, cos(x[2])*est_dt, -sin(x[2])*est_dt, 0, 0, 0, 0],
#     [0, 1, 0, sin(x[2])*est_dt,  cos(x[2])*est_dt, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0], 
#     [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1]])  # state matrix
est_map_A = lambda x: matrix([  # 0-x, 1-y, 2-z, 3-vx, 4-vy, 5-vz, 6-roll, 7-pitch, 8-yaw 
    [1, 0, 0, cos(x[8])*cos(x[7])*est_dt, -cos(x[6])*sin(x[8])*est_dt+cos(x[8])*sin(x[7])*sin(x[6])*est_dt, sin(x[8])*sin(x[6])*est_dt + cos(x[6])*sin(x[7])*cos(x[8])*est_dt, 0, 0, 0],
    [0, 1, 0, sin(x[8])*cos(x[7])*est_dt, cos(x[8])*cos(x[6])*est_dt + sin(x[8])*sin(x[7])*sin(x[6])*est_dt, -cos(x[8])*sin(x[6])*est_dt + sin(x[8])*sin(x[7])*cos(x[6])*est_dt, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]])  # state matrix
est_map_B = matrix(zeros((9,3)))    # input matrix

def estMeasArrays():
    # If DVL updated, save data to measurement arrays
    if dvl_update == 1:
               
        # Update the linear position in the map frame measurement array
        est_linPosMapArray[0:2,1:] = est_linPosMapArray[0:2,:-1]
        est_linPosMapArray[0:2,0] = dvl_aprxLinPosMap[0:2,dvl_k]

        # Update the linear velocity in the robot frame measurement array
        est_linVelRbtArray[:, 1:] = est_linVelRbtArray[:, :-1]
        est_linVelRbtArray[:, 0] = dvl_linVel[:, dvl_k]

    # If IMU updated, save data to measurement arrays
    if imu_update == 1:

        # Update the angular position (in map and robot frame) measurement array
        est_angRotArray[:, 1:] = est_angRotArray[:, :-1]
        est_angRotArray[:, 0] = imu_angRot[:, imu_k]
                   
        # Update the linear acceleration in robot frame measurement array
        est_linAccRbtArray[:,1:] = est_linAccRbtArray[:, :-1]
        est_linAccRbtArray[:,0] = imu_linAcc[:, imu_k]   

        # Update the acceleration bias in robot frame measurement array
        est_acBiasArray[:,1:] = est_acBiasArray[:, :-1]
        est_acBiasArray[:,0] = imu_acBias[:, imu_k]   
    
    # If barometer updated, save data to measurement arrays
    if bar_update == 1:
        
        # Update the linear position in the map frame measurement array
        est_linPosMapArray[2,1:] = est_linPosMapArray[2,:-1]
        est_linPosMapArray[2,0] = bar_linPos[0,bar_k]

# def estSlidingCovGain():
#     # time calculations




def estRbtMeas():
    # Position measurements
    est_rbt_m[0:2, est_k] = est_rbt_m[0:2, est_k]   # linear position in robot frame        

    # Barometer measurement
    est_rbt_m[2, est_k] = est_linPosMapArray[2,0]   # linear z position in robot frame

    # Linear velocity measurements              
    est_rbt_m[3:6,est_k] = est_rbt_m[3:6,est_k]   # linear velocity in robot frame  

    # Rotation measurements
    est_rbt_m[6:9,est_k] = est_angRotArray[:,0]   # angular position in robot frame    

    # Linear acceleration measurements
    est_rbt_m[9:12,est_k] = est_linAccRbtArray[:,0]    # linear acceleration in robot frame     

    # IMU acceleration bias measurements
    est_rbt_m[12:15,est_k] = est_acBiasArray[:,0]    # linear acceleration in robot frame     

def rbtLinVel():

    # Takes the given velocity data from the DVL
    if (est_useDvlLinVel == 1 and est_useImuLinVel == 0):
        # save robot frame velocity to measurement matrix
        est_rbt_m[3:6,est_k] = est_linVelRbtArray[:,0]*est_linVelBias  

    # Takes only the trapezoidal integration method
    if (est_useDvlLinVel == 0 and est_useImuLinVel == 1):
        if (est_k == 1):
            est_linVelImuAprxRbt[:,0] = (1/2)*(est_linAccRbtArray[:,1]+est_linAccRbtArray[:,0])*(1/imu_rr)                         
        else:
            est_linVelImuAprxRbt[:,0] = est_rbt_x[3:6,est_k-1]+(1/2)*(est_linAccRbtArray[:,1]+est_linAccRbtArray[:,0])*(1/imu_rr)                                     
               
        # save robot frame velocity to measurement matrix
        est_rbt_m[3:6,est_k] = est_linVelImuAprxRbt[:,0]*est_linVelBias 

    # Takes both the velocity data from DVL and integrated IMU data
    elif (est_useDvlLinVel == 1 and est_useImuLinVel == 1):
        if (est_k == 1):
            est_linVelImuAprxRbt[:,0] = (1/2)*(est_linAccRbtArray[:,1]+est_linAccRbtArray[:,0])*(1/imu_rr)                         
        else:
            est_linVelImuAprxRbt[:,0] = est_rbt_x[3:6,est_k-1]+(1/2)*(est_linAccRbtArray[:,1]+est_linAccRbtArray[:,0])*(1/imu_rr)                                     
               
        # save robot frame velocity to measurement matrix
        est_rbt_m[3,est_k] = (est_linVelRbtArray[0,0]+est_linVelImuAprxRbt[0,0])/2*est_linVelBias 
        est_rbt_m[4,est_k] = (est_linVelRbtArray[1,0]+est_linVelImuAprxRbt[1,0])/2*est_linVelBias 
        est_rbt_m[5,est_k] = (est_linVelRbtArray[2,0]+est_linVelImuAprxRbt[2,0])/2*est_linVelBias 

    # If condition throws an error
    else:
        est_rbt_m[3:6, est_k] = matrix([[0], [0], [0]]) 
    
def mapLinPos():
    if (est_useDvlLinVel == 1 and est_useImuLinVel == 1):
        est_map_m[2, est_k] = est_linPosMapArray[2, 0]*est_linPosBias
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
    if k != t_N:
        est_map_m[0:2, est_k] = est_linPosMapArray[0:2,0]
        # est_map_m[0:2, est_k] = est_map_m[0:2, est_k] 
        
        est_map_m[2, est_k] = est_linPosMapArray[2,0]

    # Robot frame linear velocity measurements
    if k != t_N:
        est_map_m[3:6, est_k] = est_rbt_x[3:6,est_k+1]          

   # Angular position measurements
    if k != t_N:
        est_map_m[6:9,est_k] = est_rbt_x[6:9, est_k+1]   # angular position in map frame 

   # Acceleration bias    
    if k != t_N:
        # IMU acceleration bias measurements
        est_rbt_m[12:15,est_k] = est_acBiasArray[:,0]    

""" Store data """
# Store data for plotting
simX = []
simY = []
simZ = []
linPosX = []
linPosY = []
estMapX = []
estMapY = []
estMapZ = []
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

        """ Estimator """
        if k*sim.dt > est_dt*est_counter:   

            # Estimator measurement array
            estMeasArrays()       

            # Estimator robot measurement            
            estRbtMeas()  

            # Linear velocity
            rbtLinVel()   

            # Map linear position
            mapLinPos()

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
                    
                state = ExtendedKalmanFilter(x, P, z, u, A, B, Q, R, H)
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
            
        ### Save data ###
        simX.append(sim_xm[0, k])
        simY.append(sim_xm[1, k])
        simZ.append(sim_xm[2, k])
        linPosX.append(dvl_aprxLinPosMap[0, dvl_k])
        linPosY.append(dvl_aprxLinPosMap[1, dvl_k])
        estMapX.append(est_map_x[0, est_k])
        estMapY.append(est_map_x[1, est_k])
        estMapZ.append(est_map_x[2, est_k])
            
        ### Increment time ###
        t_now = t_now + sim.dt

""" Plots """
# Robot x vs y
genPlots.plotXY(simX, simY, linPosX, linPosY, estMapX, estMapY)
genPlots.plotMap(t_N, endTime, simX, simY, simZ, estMapX, estMapY, estMapZ, linPosX, linPosY)
plt.show()
