#! /usr/bin/env python
# """
# The purpose of this file is to estimate the state of the vehicle with DVL dead reckoning
# """

""" Import libraries """
import sys
import numpy as np
from numpy import array, zeros, reshape, matmul, eye, sqrt, cos, sin
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
from transformations import quaternion_from_euler, euler_from_quaternion

from ekf import ExtendedKalmanFilter
from commons import EuclideanDistance, SkewSymmetric, Rot, TrapIntegrate, MapAngVelTrans, PressureToDepth

class DeadReckon(object):
    def __init__(self, enuFrame):
        """ Initial Parameters """
        self.enuFrame = enuFrame
        """ Sensor setup """
        # Instantiate DVL variables and arrays 
        self.sen_dvl_update = 0 # dvl update flag (0 to set 0, 1 to set 1)
        self.sen_dvl_time = 0 # current dvl sensor time
        self.sen_dvl_previousTime = 0 # last time since dvl updated
        self.sen_dvl_rbtLinVel = zeros(shape=(3,1)) # robot frame linear velocity from dvl 
        self.sen_dvl_mapLinVel = zeros(shape=(3,1)) # NED map frame linear velocity from dvl
        self.sen_dvl_offset = zeros(shape=(3,1)) # dvl physical offset from vehicle COM

        # Instantiate IMU variables and arrays
        self.sen_imu_update = 0 # imu update flag (0 to set 0, 1 to set 1)
        self.sen_imu_time = 0 # current imu sensor time
        self.sen_imu_previousTime = 0 # last time since imu updated
        self.sen_imu_rbtAngVel = zeros(shape=(3,1)) # robot frame angular velocity from IMU
        self.sen_imu_mapAngPos = zeros(shape=(3,3)) # orientation rotation matrix
        self.sen_imu_mapEulAngPos = zeros(shape=(3,1)) # orientation euler angle matrix 
        self.sen_imu_mapAngVel = zeros(shape=(3,1)) # map frame angular velocity from IMU

        # Sensor frame setup
        # Configure DVL frame to ENU (x-forward, z-upwards)
        self.sen_dvl_enuFrameRoll = np.deg2rad(0)
        self.sen_dvl_enuFramePitch = np.deg2rad(90) # -90
        self.sen_dvl_enuFrameYaw = np.deg2rad(0)
        # Configure DVL frame to NED (x-forward, z-downwards)
        self.sen_dvl_nedFrameRoll = np.deg2rad(0)
        self.sen_dvl_nedFramePitch = np.deg2rad(90)
        self.sen_dvl_nedFrameYaw = np.deg2rad(180)
        # Configure IMU frame to NED (x-forward, z-downwards)
        self.sen_imu_nedFrameRoll = np.deg2rad(180)
        self.sen_imu_nedFramePitch = np.deg2rad(0)
        self.sen_imu_nedFrameYaw = np.deg2rad(0)
        self.frameTrans = Rot(self.sen_dvl_enuFrameRoll, self.sen_dvl_enuFramePitch, self.sen_dvl_enuFrameYaw)
        
        """ DR setup """
        # Instantiate DR measurement variables and arrays
        self.meas_update = 0
        self.dr_mapLinVel = zeros(shape=(3,1)) # map frame estimator linear velocity array
        self.dr_mapAngPos = zeros(shape=(3,3)) # map frame angular position array
        self.dr_mapEulAngPos = zeros(shape=(3,1)) # orientation euler angle matrix 
        self.dr_mapAngVel = zeros(shape=(3,1)) # map frame angular velocity array
        self.dr_rbtLinVel = zeros(shape=(3,1)) # robot frame estimator linear velocity array
        self.dr_rbtAngVel = zeros(shape=(3,1)) # robot frame angular velocity array
        self.position_curr = zeros(shape=(3,1)) # map frame estimator linear position array
        self.position_previous = zeros(shape=(3,1)) # map frame estimator linear position array

    """ Raw sensor measurements """
    def DvlCallback(self, rbtLinVel, offset, dt):
        # Instantiate dvl variables
        self.sen_dvl_update = 1
        self.dt = dt

        # Raw measurements
        self.sen_dvl_rbtLinVel = rbtLinVel # robot linear velocity array
        self.sen_dvl_offset = offset # offset

        # Perform measurement update
        self.meas_update = 1
        self.RunDr()
        self.meas_update = 0

        # Update sensor flag
        self.sen_dvl_update = 0

    def ImuCallback(self, rbtAngVel, mapAngPos, mapEulAngPos):
        # Instantiate imu variables
        self.sen_imu_update = 1

        # Raw measurements
        self.sen_imu_rbtAngVel = rbtAngVel # robot angular velocity array
        self.sen_imu_mapAngPos = mapAngPos # map angular position rotation matrix array
        self.sen_imu_mapEulAngPos = mapEulAngPos # map angular position euler angle matrix array

        # Perform time update
        self.meas_update = 0
        self.RunDr()

        # Update sensor flag
        self.sen_imu_update = 0

    """ Convert sensor measurements from robot body frame to map frame """
    def RbtToMap(self):
        if self.enuFrame == 1: # ENU (x-forward, z-upwards)
            # IMU
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
            # Update the linear velocity in map and robot frame measurement array
            self.dr_mapLinVel = self.sen_dvl_mapLinVel
            self.dr_rbtLinVel = self.sen_dvl_rbtLinVel

        # Update IMU sensor array if a sensor update occurs
        if self.sen_imu_update == 1:
            # Update the angular position in map frame measurement array
            self.dr_mapAngPos = self.sen_imu_mapAngPos # rotation matrix
            self.dr_mapEulAngPos = self.sen_imu_mapEulAngPos # euler angle

            # Update the angular velocity in map and robot frame measurement array
            self.dr_mapAngVel = self.sen_imu_mapAngVel
            self.dr_rbtAngVel = self.sen_imu_rbtAngVel

    """ Run Estimator """ 
    def RunDr(self):    
        self.RbtToMap() # convert sensor measurements from robot frame into map frame
        self.SenMeasArrays() # collect sensor measurements

        if self.meas_update == 1:
            self.position_curr = self.position_previous + self.dr_mapLinVel*self.dt
        self.position_previous = self.position_curr 

        # Output state
        self.OutputDr()

    def OutputDr(self):
        return self.position_curr, self.dr_mapLinVel, self.dr_mapAngVel
        