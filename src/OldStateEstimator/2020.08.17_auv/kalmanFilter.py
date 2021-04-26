#! /usr/bin/env python
from numpy.linalg import inv
from numpy import identity
from numpy import matrix, diag, random, array, zeros, block
from helper_utils import MovingAverage
from helper_utils import Logger
from helper_utils import KalmanPlotter
import math
import matplotlib.pyplot as plt

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


    def predict(self, x, P):
        # Prediction for state vector and covariance
        x = self.A * x + self.B*self.u
        P = self.A * P * (self.A.T) + self.Q
        return(x, P)

    def update(self, x, P, z):
        # Compute Kalman Gain
        K = P * (self.H.T) * inv(self.H * P * (self.H.T) + self.R)

        # Correction based on observation
        x += K * (z - self.H * x)
        P = P - K * self.H * P
        return (x, P)


#################################################################################3
"""
Simulation user input
"""
# Time properties
dt = 0.001      # simulation time step [s]
endTime = 50    # simulation end time [s]
t_N = round(endTime/dt) # total number of time steps [s]

# USV lumped parameters
m = 225
I = 100
bxr = 25
byr = 300
bpr = 300

# Robot FRAME initial conditions
ic_x0 = 0
ic_y0 = 0
ic_tz = 0
ic_dx0 = 0
ic_dy0 = 0
ic_dtz0 = 0

# Sensor setup
gps_rr = 2      # refresh rate of GPS sensor [Hz]
gps_snr = 2     # signal to noise ratio of GPS sensor
imu_rr = 20     # refresh rate of IMU sensor [Hz]
imu_snr = 40    # signal to noise ratio of IMU sensor

#######################################
# State, output, and input vectors
sim_xr = zeros((9, t_N))
sim_yr = zeros((9, t_N))
sim_xm = zeros((9, t_N))
sim_ym = zeros((9, t_N))
sim_u = zeros((3, t_N))

# Sensor data vectors
gps_xr = zeros((9, t_N))
gps_xm = zeros((9, t_N))
imu_xr = zeros((9, t_N))
imu_xm = zeros((9, t_N))

# Sensor update tracker
gps_dt = 0
imu_dt = 0

#####################
p_A = array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, -bxr/m, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -byr/m, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -bpr/I, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]           
])

p_B = array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1/m, 0, 0],
    [0, 1/m, 0],
    [0, 0, 1/I],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])

p_C = block([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [p_B[3:6, :]]
])

p_D = block([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  
    [p_B[3:6, :]] 
])

####################################
def tf_r2m(vecR, tz):
    vecM = array([
        [math.cos(tz), -math.sin(tz), 0, 0, 0, 0, 0, 0, 0],   
        [math.sin(tz), math.cos(tz), 0, 0, 0, 0, 0, 0, 0],    
        [0, 0, 1, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, math.cos(tz), -math.sin(tz), 0, 0, 0, 0],  
        [0, 0, 0, math.sin(tz), math.cos(tz), 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 1, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, math.cos(tz), -math.sin(tz), 0], 
        [0, 0, 0, 0, 0, 0, math.sin(tz), math.cos(tz), 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 1]*vecR
    ])
    return vecM

def tf_m2r(vecM, tz):
    vecR = array([
        [math.cos(tz), -math.sin(tz), 0, 0, 0, 0, 0, 0, 0],   
        [math.sin(tz), math.cos(tz), 0, 0, 0, 0, 0, 0, 0],    
        [0, 0, 1, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, math.cos(tz), -math.sin(tz), 0, 0, 0, 0],  
        [0, 0, 0, math.sin(tz), math.cos(tz), 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 1, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, math.cos(tz), -math.sin(tz), 0], 
        [0, 0, 0, 0, 0, 0, math.sin(tz), math.cos(tz), 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 1]*vecM
    ])
    return vecR    

#####################################
# Insert intial conditions
sim_xm[0, 1] = ic_x0
sim_xm[1, 0] = ic_y0
sim_xm[2, 0] = ic_tz
sim_xm[3, 0] = ic_dx0
sim_xm[4, 0] = ic_dy0
sim_xm[5, 0] = ic_dtz0
sim_ym[:, 0] = sim_xm[:, 0]

######################
F = identity(9)+p_A*dt
G = p_B*dt
H = p_C
J = p_D

################# PLOT ###############
simPlot = True
senPlot = False
caEstPlot = False
ddEstPlot = False

for k in range (0, t_N):
    if (k > 1):
        sim_u[0, k] = sim_u[0, k-1]+0.0025
        sim_u[1, k] = sim_u[1, k-1]+0.0025
        sim_u[2, k] = sim_u[2, k-1]+0.0025

    if (k != t_N):
        sim_xr[:, k+1] = F*sim_yr[:, k]+G*sim_u
        sim_yr[:, k+1] = H*sim_yr[:, k+1]+J*sim_u

        sim_xm[:,k+1] = tf_r2m(sim_xr[:, k+1], sim_xr[2, k+1])
        sim_ym[:, k+1] = tf_r2m(sim_yr[:, k+1], sim_yr[2, k+1])
    
# Robot x vs y
plt.figure()
if simPlot:
    plt.plot(sim_ym[0, :], sim_ym[2, :], color = 'b')
    plt.xlabel('x-position [m]')
    plt.ylabel('y-position [m]')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# """
# Simulation user input
# """
# # Time properties
# dt = 0.001      # simulation time step [s]
# endTime = 50    # simulation end time [s]
# t_N = round(endTime/dt) # total number of time steps [s]

# # USV lumped parameters
# m = 225
# I = 100
# bxr = 25
# byr = 300
# bpr = 300

# # Robot FRAME initial conditions
# ic_x0 = 0
# ic_y0 = 0
# ic_tz = 0
# ic_dx0 = 0
# ic_dy0 = 0
# ic_dtz0 = 0

# # Sensor setup
# gps_rr = 2      # refresh rate of GPS sensor [Hz]
# gps_snr = 2     # signal to noise ratio of GPS sensor
# imu_rr = 20     # refresh rate of IMU sensor [Hz]
# imu_snr = 40    # signal to noise ratio of IMU sensor

# ################
# caest_dt = 0.005
# caest_Q = array([
#     [0.1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0.1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0.1, 0, 0, 0, 0, 0, 0], 
#     [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],       
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0],   
#     [0, 0, 0, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1]
# ])
# caest_R = array([
#     [0.1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0.1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0.1, 0, 0, 0, 0, 0, 0], 
#     [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],       
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0],   
#     [0, 0, 0, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1]    
# ])
# caest_H = array([
#     [1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0], 
#     [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],       
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0],   
#     [0, 0, 0, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1]    
# ])

# ################
# ddest_dt = 0.005
# caest_Q = array([
#     [0.1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0.1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0.1, 0, 0, 0, 0, 0, 0], 
#     [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],       
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0],   
#     [0, 0, 0, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1]
# ])
# caest_R = array([
#     [0.1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0.1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0.1, 0, 0, 0, 0, 0, 0], 
#     [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],       
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0],   
#     [0, 0, 0, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1]    
# ])
# caest_H = array([
#     [1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0], 
#     [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],       
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0],   
#     [0, 0, 0, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1]    
# ])

# ############
# # State, output, and input vectors
# sim_xr = zeros((9, t_N))
# sim_yr = zeros((9, t_N))
# sim_xm = zeros((9, t_N))
# sim_ym = zeros((9, t_N))
# sim_u = zeros((3, t_N))

# # Sensor data vectors
# gps_xr = zeros((9, t_N))
# gps_xm = zeros((9, t_N))
# imu_xr = zeros((9, t_N))
# imu_xm = zeros((9, t_N))

# # Sensor update tracker
# gps_dt = 0
# imu_dt = 0

# #########################
# caest_xm = zeros((9, t_N))
# caest_xr = zeros((9, t_N))
# caest_Lx = zeros((9, 9, t_N))
# caest_Px = zeros((9, 9, t_N))

# ############################
# ddest_xm = zeros((9, t_N))
# ddest_xr = zeros((9, t_N))
# ddest_Lx = zeros((9, 9, t_N))
# ddest_Px = zeros((9, 9, t_N))
# ddest_ym = zeros((9, t_N))
# ddest_yr = zeros((9, t_N))
# ddest_Ly = zeros((9, 9, t_N))
# ddest_Py = zeros((9, 9, t_N))

# #############################
# p_A = array([
#     [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 0, 0, -bxr/m, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, -byr/m, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, -bpr/I, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0]           
# ])

# p_B = array([
#     [0, 0, 0],
#     [0, 0, 0],
#     [0, 0, 0],
#     [1/m, 0, 0],
#     [0, 1/m, 0],
#     [0, 0, 1/I],
#     [0, 0, 0],
#     [0, 0, 0],
#     [0, 0, 0]
# ])

# p_C = block([
#     [1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [p_B[3:6, :]]
# ])

# ##############
# vecM = array([
#     [math.cos(x), -math.sin(x), 0, 0, 0, 0, 0, 0, 0],   
#     [math.sin(x), math.cos(x), 0, 0, 0, 0, 0, 0, 0],    
#     [0, 0, 1, 0, 0, 0, 0, 0, 0],    
 
# ])

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
#     (x, P) = kf.predict(x, P)
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
