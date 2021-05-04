#! /usr/bin/env python
# """
# The purpose of this file is to estimate the state of the vehicle with EKF
# using DVL, IMU, and barometer
# """

""" Import libraries """
import sys
import numpy as np
from numpy import array, zeros, reshape, matmul, eye, sqrt, cos, sin
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
from transformations import quaternion_from_euler, euler_from_quaternion

from ekf import ExtendedKalmanFilter
from commons import EuclideanDistance, SkewSymmetric, Rot, TrapIntegrate, MapAngVelTrans, PressureToDepth

class EkfEstimator(object):
  def __init__(self, dt, enuFrame, pseudoLinPos):
    """ Initial Parameters """
    self.est_dt = dt
    self.enuFrame = enuFrame # frame flag (1 to set ENU, 0 to set NED)
    # Decide which sensor to calculate pseudo position estimate
    if pseudoLinPos == 0:
      self.est_useDvlLinPos = 1
    if pseudoLinPos == 1:
      self.est_useImuLinPos = 1
    if pseudoLinPos == 2:
      self.est_useDvlLinPos = 1
      self.est_useImuLinPos = 1

    """ Sensor setup """
    # Instantiate DVL variables and arrays
    self.sen_dvl_update = 0 # dvl update flag (0 to set 0, 1 to set 1)
    self.sen_dvl_time = 0 # current dvl sensor time
    self.sen_dvl_previousTime = 0 # last time since dvl updated
    self.sen_dvl_rbtLinVel = zeros(shape=(3,1)) # robot frame linear velocity from dvl
    self.sen_dvl_mapLinVel = zeros(shape=(3,1)) # NED map frame linear velocity from dvl
    self.sen_dvl_aprxMapLinPos = zeros(shape=(3,1)) # map frame approximated linear position from dvl
    self.sen_dvl_offset = zeros(shape=(3,1)) # dvl physical offset from vehicle COM

    # Instantiate IMU variables and arrays
    self.sen_imu_update = 0 # imu update flag (0 to set 0, 1 to set 1)
    self.sen_imu_time = 0 # current imu sensor time
    self.sen_imu_previousTime = 0 # last time since imu updated
    self.sen_imu_rbtAngVel = zeros(shape=(3,1)) # robot frame angular velocity from IMU
    self.sen_imu_rbtLinAcc = zeros(shape=(3,1)) # robot frame linear acceleration from IMU
    self.sen_imu_mapLinAcc = zeros(shape=(3,1)) # map frame linear acceleration from IMU
    self.sen_imu_mapLinAccNoGrav = zeros(shape=(3,1)) # map frame linear acceleration without gravity
    self.sen_imu_mapAngPos = zeros(shape=(3,3)) # orientation rotation matrix
    self.sen_imu_mapEulAngPos = zeros(shape=(3,1)) # orientation euler angle matrix
    self.sen_imu_mapAngVel = zeros(shape=(3,1)) # map frame angular velocity from IMU
    self.sen_imu_aprxMapLinPos = zeros(shape=(3,1)) # map frame approximated linear position from integrated linear acceleration IMU
    self.sen_imu_aprxMapLinVel = zeros(shape=(3,1)) # map frame approximated linear velocity from integrated linear acceleration IMU
    self.sen_imu_lastMapAprxLinVel = zeros(shape=(3,1)) # map frame approximated last linear velocity from integrated linear acceleration IMU

    # Instantiate Barometer variables and arrays
    self.sen_bar_update = 0 # barometer update flag (0 to set 0, 1 to set 1)
    self.sen_bar_previousTime = 0 # last time since bar updated
    self.sen_bar_mapLinPos = 0 # map frame barometer linear position

    # Sensor frame setup
    # DVL rigid frame transformation wrt IMU
    # DVL frame in ENU configuration(x-forward, z-upwards)
    self.sen_dvl_enuFrameRoll = np.deg2rad(0)
    self.sen_dvl_enuFramePitch = np.deg2rad(0) # -90
    self.sen_dvl_enuFrameYaw = np.deg2rad(0)
    self.frameTrans = Rot(self.sen_dvl_enuFrameRoll, self.sen_dvl_enuFramePitch, self.sen_dvl_enuFrameYaw) # compute for the corresponding rotation matrix
    self.dvl_offsetTransRbtLinVel = np.zeros(shape=(3,3)) # dvl frame transformation considering dvl linear position offset wrt IMU

    """ Estimator setup """
    # Instantiate estimator measurement variables and arrays
    self.meas_update = 0
    self.est_mapLinPos = zeros(shape=(3,1)) # map frame estimator linear position array
    self.est_mapLinVel = zeros(shape=(3,1)) # map frame estimator linear velocity array
    self.est_mapPrevLinVel = zeros(shape=(3,1)) # previous map frame estimator linear velocity array
    self.est_mapAngPos = zeros(shape=(3,3)) # map frame angular position array
    self.est_mapEulAngPos = zeros(shape=(3,1)) # orientation euler angle matrix
    self.est_mapAngVel = zeros(shape=(3,1)) # map frame angular velocity array
    self.est_mapLinAcc = zeros(shape=(3,1)) # map frame linear acceleration
    self.est_mapPrevLinAcc = zeros(shape=(3,1)) # previous map frame linear acceleration
    self.est_rbtLinVel = zeros(shape=(3,1)) # robot frame estimator linear velocity array
    self.est_rbtAngVel = zeros(shape=(3,1)) # robot frame angular velocity array
    self.est_rbtLinAcc = zeros(shape=(3,1)) # robot frame linear acceleration
    self.est_rbtAccBias = zeros(shape=(3,1)) # robot frame acceleration bias
    self.est_rbtGyrBias = zeros(shape=(3,1)) # robot frame gyroscope bias

    # Instantiate estimator matrices
    est_inputDimState = 6 # number of estimator inputs
    est_measDimState = 13 # number of measurements being taken
    self.est_dimState = 15 # number of estimate states
    self.gravity = array([[0], [0], [9.80665]]) # gravity vector [m/s^2]
    self.est_A = eye(self.est_dimState) # jacobian state matrix
    self.est_H = zeros(shape=(est_measDimState,self.est_dimState)) # jacobian observation matrix
    self.est_u = zeros(shape=(est_inputDimState,1)) # input matrix
    self.est_m = zeros(shape=(self.est_dimState,1)) # frame estimator measurement matrix
    self.est_x = zeros(shape=(self.est_dimState,1)) # frame estimator state estimate
    self.est_L = zeros(shape=(self.est_dimState,self.est_dimState)) # frame estimator kalman gain matrix
    self.est_P = zeros(shape=(self.est_dimState,self.est_dimState)) # frame estimator covariance matrix
    self.est_prev_x = zeros(shape=(self.est_dimState,1)) # frame estimator state estimate

  def ComputeQ(self, est_Q_linPos, est_Q_angPos, est_Q_linVel, est_Q_acBias, est_Q_gyBias):
    self.est_Q = array([
      [np.power(est_Q_linPos,2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, np.power(est_Q_linPos,2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, np.power(est_Q_linPos,2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, np.power(est_Q_angPos,2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, np.power(est_Q_angPos,2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, np.power(est_Q_angPos,2), 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, np.power(est_Q_linVel,2), 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, np.power(est_Q_linVel,2), 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, np.power(est_Q_linVel,2), 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, np.power(est_Q_acBias,2), 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.power(est_Q_acBias,2), 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.power(est_Q_acBias,2), 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.power(est_Q_gyBias,2), 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.power(est_Q_gyBias,2), 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.power(est_Q_gyBias,2)]]) # process noise covariance matrix

  def ComputeR(self, est_R_linPosZ, est_R_angPos, est_R_linVel, est_R_acBias, est_R_gyBias):
    self.est_R = array([
      [np.power(est_R_linPosZ,2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, np.power(est_R_angPos,2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, np.power(est_R_angPos,2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, np.power(est_R_angPos,2), 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, np.power(est_R_linVel,2), 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, np.power(est_R_linVel,2), 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, np.power(est_R_linVel,2), 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, np.power(est_R_acBias,2), 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, np.power(est_R_acBias,2), 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, np.power(est_R_acBias,2), 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.power(est_R_gyBias,2), 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.power(est_R_gyBias,2), 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.power(est_R_gyBias,2)]]) # measurement noise covariance matrix

  """ Raw sensor measurements """
  def DvlCallback(self, rbtLinVel, offset, t):
    # Instantiate dvl variables
    self.sen_dvl_update = 1
    self.sen_dvl_time = t

    # Raw measurements
    self.sen_dvl_rbtLinVel = rbtLinVel # robot linear velocity array
    self.sen_dvl_offset = offset # offset

    # Perform measurement update
    self.meas_update = 1
    self.RunEst()
    self.meas_update = 0

    # Update sensor flag
    self.sen_dvl_update = 0

  def ImuCallback(self, rbtLinAcc, rbtAngVel, mapAngPos, mapEulAngPos, rbtAccBias, rbtGyrBias, t):
    # Instantiate imu variables
    self.sen_imu_update = 1
    self.sen_imu_time = t

    # Raw measurements
    self.sen_imu_rbtLinAcc = rbtLinAcc # robot linear acceleration array
    self.sen_imu_rbtAngVel = rbtAngVel # robot angular velocity array
    self.sen_imu_mapAngPos = mapAngPos # map angular position rotation matrix array
    self.sen_imu_mapEulAngPos = mapEulAngPos # map angular position euler angle matrix array
    self.sen_imu_accBias = rbtAccBias # robot accel bias array
    self.sen_imu_gyrBias = rbtGyrBias # robot gyro bias array

    # Perform time update
    self.meas_update = 0
    self.RunEst()

    # Update sensor flag
    self.sen_imu_update = 0

  def DepthCallback(self, rbtLinPos, t):
    # Instantiate barometer variables
    self.sen_bar_update = 1
    self.sen_bar_time = t

    # Raw measurements
    self.sen_bar_mapLinPos = rbtLinPos # map linear position in z-direction

    # Perform time update
    self.meas_update = 0
    self.RunEst()

    # Update sensor flag
    self.sen_bar_update = 0

  """ Convert sensor measurements from robot body frame to map frame """
  def RbtToMap(self):
    if self.enuFrame == 1: # ENU (x-forward, z-upwards)
      # IMU
      # Convert linear acceleration from robot frame into map frame
      self.sen_imu_mapLinAcc = np.matmul(self.sen_imu_mapAngPos, self.sen_imu_rbtLinAcc)
      self.sen_imu_mapLinAccNoGrav = self.est_u[0:3] - self.gravity
      # Convert angular velocity from robot frame into map frame
      self.sen_imu_mapAngVel = np.matmul(MapAngVelTrans(self.sen_imu_mapEulAngPos[0], self.sen_imu_mapEulAngPos[1], self.sen_imu_mapEulAngPos[2]), self.sen_imu_rbtAngVel)

      # DVL
      # Correct DVL coordinate frame wrt to ENU
      dvl_enuTransRbtLinVel = np.matmul(self.frameTrans, self.sen_dvl_rbtLinVel)
      dvl_enuTransRbtLinVel -= np.cross(self.sen_imu_rbtAngVel.T, self.sen_dvl_offset.T).T
      # Convert velocity from robot frame into map frame
      self.sen_dvl_mapLinVel = np.matmul(self.sen_imu_mapAngPos, dvl_enuTransRbtLinVel)

  """ Estimator measurement array """
  def SenMeasArrays(self):
    # Update DVL sensor array if a sensor update occurs
    if self.sen_dvl_update == 1:
      # Set estimator last known sensor update to last known sensor update
      self.sen_dvl_previousTime = self.sen_dvl_time

      # Update the linear velocity in map and robot frame measurement array
      self.est_mapLinVel = self.sen_dvl_mapLinVel
      self.est_rbtLinVel = self.sen_dvl_rbtLinVel

    # Update barometer sensor array if a sensor update occurs
    if self.sen_bar_update == 1:
      # Set estimator last known sensor update to last known sensor update
      self.sen_bar_previousTime = self.sen_bar_time

      # Update the linear position (map frame) measurement array
      self.est_mapLinPos[2] = self.sen_bar_mapLinPos

    # Update IMU sensor array if a sensor update occurs
    if self.sen_imu_update == 1:
      # Set estimator last known sensor update to last known sensor update
      self.sen_imu_previousTime = self.sen_imu_time

      # Update the linear acceleration in map and robot frame measurement array
      self.est_mapLinAcc = self.sen_imu_mapLinAcc
      self.est_rbtLinAcc = self.sen_imu_rbtLinAcc

      # Update the angular position in map frame measurement array
      self.est_mapAngPos = self.sen_imu_mapAngPos # rotation matrix
      self.est_mapEulAngPos = self.sen_imu_mapEulAngPos # euler angle

      # Update the angular velocity in map and robot frame measurement array
      self.est_mapAngVel = self.sen_imu_mapAngVel
      self.est_rbtAngVel = self.sen_imu_rbtAngVel

      # Update the acceleration and gyro bias in robot frame measurement array
      self.est_rbtAccBias = self.sen_imu_accBias
      self.est_rbtGyrBias = self.sen_imu_gyrBias

  """ Compute for map frame linear position """
  def MapLinPos(self):
    # Use only DVL
    if self.est_useDvlLinPos == 1 and self.est_useImuLinPos == 0:
      # Integrate velocity using the trapezoidal method to compute for position
      self.est_mapLinPos[0:2] = TrapIntegrate(self.sen_dvl_time, self.est_mapLinVel[0:2], self.est_x[0:2], self.sen_dvl_previousTime, self.est_mapPrevLinVel[0:2])

    # Use only IMU
    if self.est_useDvlLinPos == 0 and self.est_useImuLinPos == 1:
      # Integrate acceleration using the trapezoidal method to compute for velocity
      self.sen_imu_aprxMapLinVel[0:2] = TrapIntegrate(self.sen_imu_time, self.est_mapLinAcc[0:2], self.est_x[6:8], self.sen_imu_previousTime, self.est_mapPrevLinAcc[0:2])
      self.est_mapLinPos[0:2] = TrapIntegrate(self.sen_imu_time, self.sen_imu_aprxMapLinVel[0:2], self.est_x[0:2], self.sen_imu_previousTime, self.sen_imu_lastMapAprxLinVel[0:2])

    # Use both DVL and IMU
    if self.est_useDvlLinPos == 1 and self.est_useImuLinPos == 1:
      # Integrate velocity using the trapezoidal method to compute for position
      self.sen_dvl_aprxMapLinPos[0:2] = TrapIntegrate(self.sen_dvl_time, self.est_mapLinVel[0:2], self.est_x[0:2], self.sen_dvl_previousTime, self.est_mapPrevLinVel[0:2])

      # Integrate acceleration using the trapezoidal method to compute for velocity then position
      self.sen_imu_aprxMapLinVel[0:2] = TrapIntegrate(self.sen_imu_time, self.est_mapLinAcc[0:2], self.est_x[6:8], self.sen_imu_previousTime, self.est_mapPrevLinAcc[0:2])
      self.sen_imu_aprxMapLinPos[0:2] = TrapIntegrate(self.sen_imu_time, self.sen_imu_aprxMapLinVel[0:2], self.est_x[0:2], self.sen_imu_previousTime, self.sen_imu_lastMapAprxLinVel[0:2])

      # Combine linear position calculated from DVL and IMU
      self.est_mapLinPos[0:2] = (self.sen_dvl_aprxMapLinPos[0:2] + self.sen_imu_aprxMapLinPos[0:2])/2

    # Update parameters
    self.est_mapPrevLinVel = self.est_mapLinVel
    self.est_mapPrevLinAcc = self.est_mapLinAcc
    self.sen_imu_lastMapAprxLinVel = self.sen_imu_aprxMapLinVel

  """ Prediction step """
  def TimePropagation(self):
    # Position update
    self.est_x[0:3] = self.est_x[0:3] + self.est_x[6:9]*self.est_dt

    # Rotation update (constant angular velocity)
    self.est_x[3:6] = self.est_x[3:6] + self.est_u[3:6]*self.est_dt

    # Velocity update
    self.est_x[6:9] = self.est_x[6:9] + self.sen_imu_mapLinAccNoGrav*self.est_dt

    # Compute state transition jacobian matrix
    self.ComputeJacobianA()

  def ComputeJacobianA(self):
    # Rotation matrix transform
    Rzyx = Rot(self.est_x[3], self.est_x[4], self.est_x[5])

    # Compute jacobian of state transition matrix
    cr = cos(self.est_x[3])
    sr = sin(self.est_x[3])
    tp = np.tan(self.est_x[4])
    secp = 1/cos(self.est_x[4])
    cp = cos(self.est_x[4])
    sp = sin(self.est_x[4])
    cy = cos(self.est_x[5])
    sy = sin(self.est_x[5])
    mat1 = np.array([
      [0, cr*sp*cy+sr*sy, -sr*sp*cy+cr*sy],
      [-sp*cy, sr*cp*cy-cr*sy, cr*cp*cy+sr*sy],
      [0, cr*sp*cy+sr*sy, -sr*sp*cy+cr*sy]
    ])
    J1 = matmul(mat1, self.est_x[6:9]) * self.est_dt
    mat2 = np.array([
      [0, cr*sp*sy-sr*cy, sr*sp*sy-cr*cy],
      [-sp*sy, sr*cp*sy+cr*cy, cr*cp*sy-sr*cy],
      [cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy]
    ])
    J2 = matmul(mat2, self.est_x[6:9]) * self.est_dt
    mat3 = np.array([
      [0, cr*cp, -sr*cp],
      [-cp, -sr*sp, -cr*sp],
      [0, 0, 0]
    ])
    J3 = matmul(mat3, self.est_x[6:9]) * self.est_dt
    p = np.array([[1], [0], [0]])
    mat4 = np.array([
      [0, cr*tp, -sr*tp],
      [0, sr*(tp**2+1), -cr*cp],
      [0, 0, 0]
    ])
    J4 = p+matmul(mat4, self.est_u[3:6]) * self.est_dt
    q = np.array([[0], [1], [0]])
    mat5 = np.array([
      [0, -sr, cr],
      [0, 0, 0],
      [0, 0, 0]
    ])
    J5 = q+matmul(mat5, self.est_u[3:6]) * self.est_dt
    r = np.array([[0], [0], [1]])
    mat6 = np.array([
      [0, cr*secp, -sr*secp],
      [0, sr*secp*tp, cr*secp*tp],
      [0, 0, 0]
    ])
    J6 = r+matmul(mat6, self.est_u[3:6]) * self.est_dt

    self.est_A = array([
        [1, 0, 0, J1[0], J1[1], J1[2], Rzyx[0,0]*self.est_dt, Rzyx[0,1]*self.est_dt, Rzyx[0,2]*self.est_dt, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, J2[0], J2[1], J2[2], Rzyx[1,0]*self.est_dt, Rzyx[1,1]*self.est_dt, Rzyx[1,2]*self.est_dt, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, J3[0], J3[1], J3[2], Rzyx[2,0]*self.est_dt, Rzyx[2,1]*self.est_dt, Rzyx[2,2]*self.est_dt, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, J4[0], J4[1], J4[2], 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, J5[0], J5[1], J5[2], 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, J6[0], J6[1], J6[2], 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # dx
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # dy
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # dz
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype = float) # jacobian state matrix

  """ Observation measurement array """
  def EstObsArray(self):
    if self.sen_bar_update == 1:
      self.est_H[0,2] = 1

    if self.sen_dvl_update == 1:
      self.est_H[4,6] = 1
      self.est_H[5,7] = 1
      self.est_H[6,8] = 1
      self.est_H[7,9] = 1
      self.est_H[8,10] = 1
      self.est_H[9,11] = 1
      self.est_H[10,12] = 1
      self.est_H[11,13] = 1
      self.est_H[12,14] = 1

    if self.sen_imu_update == 1:
      self.est_H[1,3] = 1
      self.est_H[2,4] = 1
      self.est_H[3,5] = 1

  """ Estimator measurement array """
  def EstMeasArray(self):
    # Position measurements
    self.est_m[0:3] = self.est_mapLinPos # linear position in map frame

    # Rotation measurements
    self.est_m[3:6] = self.est_mapEulAngPos # angular position in map frame

    # Linear velocity measurements
    self.est_m[6:9] = self.est_mapLinVel # linear velocity in map frame

    # Linear acceleration bias measurements
    self.est_m[9:12] = self.est_rbtAccBias # linear accel bias in robot frame

    # Gyroscope bias measurements
    self.est_m[12:15] = self.est_rbtGyrBias # linear gyro bias in robot frame

  """ Run Estimator """
  def RunEst(self):
    # Assign inputs
    self.est_u[0:3] = self.sen_imu_mapLinAcc # linear acceleration input
    self.est_u[3:6] = self.sen_imu_mapAngVel # gyroscope input

    self.RbtToMap() # convert sensor measurements from robot frame into map frame
    self.SenMeasArrays() # collect sensor measurements
    self.MapLinPos() # compute map frame linear position
    self.EstObsArray() # update observation matrix

    # Instantiate EKF
    xDim = self.est_dimState # number of states
    x = self.est_x # last estimate from robot frame
    P = self.est_P # last covariance matrix
    Q = self.est_Q # process noise covariance
    R = self.est_R # measurement noise covariance
    H = self.est_H # measurement matrix
    z = self.est_m[2:15] # measurement
    u = self.est_u # control input vector
    A = self.est_A # jacobian state matrix
    state = ExtendedKalmanFilter(xDim, x, P, z, u, A, Q, R, H)
    # Perform time update if condition is 0
    if self.meas_update == 0:
      self.TimePropagation()
      self.est_P = state.predict(P)
    # Perform measurement update if condition is 1
    else:
      self.EstMeasArray() # estimator measurement array
      self.est_x, self.est_L, self.est_P = state.update(x, P, z)

    # Output state, covariance, and control input
    self.OutputEst()

  def OutputEst(self):
    return self.est_x, self.est_P, self.est_u