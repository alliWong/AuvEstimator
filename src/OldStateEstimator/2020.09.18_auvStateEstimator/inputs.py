'''
USER INPUTS
Hyperparameters wrapped in argparse
This file contains tuanable parameters   

You can change the values by changing their default fields or by command-line
arguments. For example, "python main.py --dt 0.5"
'''

import argparse

# Time parameters
def timeInput():
    parser = argparse.ArgumentParser(description='Time properties')

    parser.add_argument('--dt', type = int, default = 0.001,
                        help='simulation time step [s]')
    parser.add_argument('--start', type = int, default = 0,
                        help='simulation start time [s]')
    parser.add_argument('--end', type = int, default = 50,
                        help='simulation end time [s]')   
    
    t = parser.parse_args()
    return t

# Simulation parameters 
def simInputs():
    parser = argparse.ArgumentParser(description='underwater vehicle lumped parameters')

    parser.add_argument('--m', type = int, default = 225,
                        help='mass/inertia [kg]')
    parser.add_argument('--I', type = int, default = 100,
                        help='rotational moment of inertia [kg*m^2]')   
    parser.add_argument('--bxr', type = int, default = 40,
                        help='drag in the surge (robot frame along x) direction [N*s/m]')
    parser.add_argument('--byr', type = int, default = 400,
                        help='drag in the sway (robot frame along y) direction [N*s/m]')   
    parser.add_argument('--bzr', type = int, default = 10,
                        help='drag in heave (robot frame along z) direction [N*s/m]')
    parser.add_argument('--bir', type = int, default = 300,
                        help='rotational drag in the roll (robot frame about x) direction [N*s/m]')   
    parser.add_argument('--btr', type = int, default = 300,
                        help='rotational drag in the pitch (robot frame about y) direction [N*s/m]')
    parser.add_argument('--bpr', type = int, default = 300,
                        help='rotational drag in the yaw (robot frame about z) direction [N*s/m]')   
    
    sim = parser.parse_args()
    return sim

# Sensor parameters
def senInputs():
    parser = argparse.ArgumentParser(description='Sensor setup')
    
    parser.add_argument('--dvl_rr', type = int, default = 20,
                        help='refresh rate of DVL sensor [Hz]')
    parser.add_argument('--dvl_linVelVar', type = int, default = 0.1,
                        help='variance of the DVL sensor data [m/s]')
    parser.add_argument('--imu_rr', type = int, default = 50,
                        help='refresh rate of IMU sensor [Hz]')
    parser.add_argument('--imu_angPosVar', type = int, default = 0.05,
                        help='variance of the IMU angular position sensor data [rad]')   
    parser.add_argument('--imu_angVelVar', type = int, default = 0.05,
                        help='variance of the IMU angular velocity sensor data [rad/s]')   
    parser.add_argument('--imu_linAccVar', type = int, default = 0.1,
                        help='variance of the IMU linear acceleration sensor data [m/s^2]')
    parser.add_argument('--imu_acBiasVar', type = int, default = 0.1,
                        help='variance of the IMU linear acceleration bias sensor data [m/s^2]')   
    parser.add_argument('--bar_rr', type = int, default = 10,
                        help='refresh rate of barometer sensor [Hz]')
    parser.add_argument('--bar_linPosVar', type = int, default = 0.1,
                        help='variance of the barometer sensor data [m]')   
    
    sen = parser.parse_args()
    return sen

# Estimator parameters
def estInputs():
    parser = argparse.ArgumentParser(description='Estimator setup')
    
    # Time step property
    parser.add_argument('--dt', type = int, default = 0.025,    # 40 Hz
                        help='estimator time step [s]')

    # Number of past sensor updates to store for calculating the moving mean of sensor data
    parser.add_argument('--angPos_n', type = int, default = 2,
                        help='angular position mean array size (must be 2 or more)')
    parser.add_argument('--linVel_n', type = int, default = 2,
                        help='linear velocity mean array size (must be 2 or more)')
    parser.add_argument('--linAcc_n', type = int, default = 2,
                        help='linear acceleration mean array size (must be 2 or more)')   
    parser.add_argument('--acBias_n', type = int, default = 2,
                        help='acceleration bias mean array size (must be 2 or more)')
    parser.add_argument('--gyBias_n', type = int, default = 2,
                        help='gyro bias mean array size (must be 2 or more)')   

    # Sliding adaptive covariance gain
    parser.add_argument('--dvlSlidingCovariance', type = int, default = 0,
                        help='1 for DVL sliding covariance gain, 0 for binary covariance gain calculation')  
    parser.add_argument('--imuSlidingCovariance', type = int, default = 0,
                        help='1 for IMU sliding covariance gain, 0 for binary covariance gain calculation')  
    parser.add_argument('--barSlidingCovariance', type = int, default = 0,
                        help='1 for barometer sliding covariance gain, 0 for binary covariance gain calculation')  

    # Decide which sensors to use for pseudo linear velocit and position calculation    
    parser.add_argument('--useDvlLinPos', type = int, default = 1,
                        help='set to 1 to use DVL trapezoidal method, set to 0 to ignore')
    parser.add_argument('--useImuLinPos', type = int, default = 0,
                        help='set to 1 to use IMU trapezoidal method, set to 0 to ignore')   
    parser.add_argument('--useDvlLinVel', type = int, default = 1,
                        help='set to 1 to use DVL trapezoidal method, set to 0 to ignore')  
    parser.add_argument('--useImuLinVel', type = int, default = 0,
                        help='set to 1 to use IMU trapezoidal method, set to 0 to ignore')
    parser.add_argument('--linPosBias', type = int, default = 1.15,
                        help='bias (multiplier) on linear position output')  
    parser.add_argument('--linVelBias', type = int, default = 1.15,    # directly measured, not needed
                        help='bias (multiplier) on linear velocity output')

    # Robot frame estimator measurement noise covariance
    parser.add_argument('--rbt_R_linPos', type = int, default = 2,
                        help='linear position [m]')
    parser.add_argument('--rbt_R_angPos', type = int, default = 0.075,
                        help='rotation [rad]')
    parser.add_argument('--rbt_R_linVel', type = int, default = 2,
                        help='linear velocity [m/s]')
    parser.add_argument('--rbt_R_linAcc', type = int, default = 0.25,
                        help='linear acceleration [m/s^s]')   
    parser.add_argument('--rbt_R_acBias', type = int, default = 2,
                        help='acceleration bias [m^2/s]')
    parser.add_argument('--rbt_R_gyBias', type = int, default = 2,
                        help='gyro bias [rad/s]') 

    # # Robot frame estimator process noise covariance
    # parser.add_argument('--rbt_Q_linPos', type = float, default = '--rbt_R_linPos'*1e2,
    #                     help='linear position [m]')
    # parser.add_argument('--rbt_Q_linVel', type = float, default = '--rbt_R_linVel'*1e2,
    #                     help='linear velocity [m/s]')
    # parser.add_argument('--rbt_Q_angPos', type = float, default = '--rbt_R_angPos'*1e-1,
    #                     help='rotation [rad]')
    # parser.add_argument('--rbt_Q_linAcc', type = float, default = '--rbt_R_linAcc'*1e-1,
    #                     help='linear acceleration [m/s^s]')   
    # parser.add_argument('--rbt_Q_acBias', type = float, default = '--rbt_R_acBias'*1e-1,
    #                     help='acceleration bias [m^2/s]')
    # parser.add_argument('--rbt_Q_gyBias', type = float, default = '--rbt_R_gyBias'*1e-1,
    #                     help='gyro bias [rad/s]') 

    # H jacobian robot frame estimator observation/observability
    parser.add_argument('--H_linPos', type = int, default = 1,
                        help='linear position (pseudo-measured)')
    parser.add_argument('--H_angPos', type = int, default = 1,
                        help='angular velocity (measured)')
    parser.add_argument('--H_linVel', type = int, default = 1,
                        help='linear velocity (measured)')
    parser.add_argument('--H_linAcc', type = int, default = 1,
                        help='linear acceleration (measured)')   
    parser.add_argument('--H_acBias', type = int, default = 1,
                        help='acceleration bias (not measured)')
    parser.add_argument('--H_gyBias', type = int, default = 1,
                        help='gyro bias (not measured)')   

    # Map frame estimator measurement noise covariance
    parser.add_argument('--map_R_linPos', type = int, default = 2,
                        help='linear position [m]')
    parser.add_argument('--map_R_linVel', type = int, default = 1,
                        help='linear velocity [m/s]')
    parser.add_argument('--map_R_angPos', type = int, default = 0.075,
                        help='angular position [rad]') 

    # # Map frame estimator process noise covariance
    # parser.add_argument('--map_R_linPos', type = float, default = '--map_R_linPos'*1e1,
    #                     help='linear position [m]')
    # parser.add_argument('--map_R_linVel', type = float, default = '--map_R_linVel'*1e1,
    #                     help='linear velocity [m/s]')
    # parser.add_argument('--map_R_angPos', type = float, default = '--map_R_angPos'*1e1,
    #                     help='angular position [rad]')     

    est = parser.parse_args()
    return est