#!/usr/bin/env python 
"""
This file computes the x-error, y-error, z-error, and distance between 
the estimator and dead reckoning against ground truth.
"""
import sys
import numpy as np
from numpy import array, zeros
from commons import Rmse, Distance, EuclideanDistance
from sklearn.metrics import mean_squared_error

class ErrorAnalysis(object):
	def __init__(self):
		""" Instantiate variables"""
		# Create array containers
		self.gt_x_list = array([], dtype='int32')
		self.gt_y_list = array([], dtype='int32')
		self.dr_x_list = array([], dtype='int32')
		self.dr_y_list = array([], dtype='int32')
		self.est_x_list = array([], dtype='int32')
		self.est_y_list = array([], dtype='int32')
		self.gt_est_x_list = array([], dtype='int32')
		self.gt_est_y_list = array([], dtype='int32')
		# GT variables
		self.gt_pose_prev = zeros(shape=(15,1)) # ground truth pose array
		self.gt_est_pose_prev = zeros(shape=(15,1)) # ground truth pose array
		self.est_gt_pose = zeros(shape=(15,1))
		# DR variables
		self.dr_rmse = zeros(shape=(15,1))
		self.dr_dist_error = 0 # dead reckoning distance error
		self.dr_error_pose = zeros(shape=(15,1)) # dead reckoning error pose array
		self.dr_dist_error_rmse = 0
		# Est variables
		self.est_rmse = zeros(shape=(15,1))
		self.est_dist_error = 0 # estimator distance error
		self.est_error_pose = zeros(shape=(15,1)) # estimator error pose array
		self.est_dist_error_rmse = 0
		# Update flags
		self.est_update = 0
		self.dr_update = 0
		# Distance variables
		self.dr_dist_traveled = 0 # distance traveled by vehicle
		self.est_dist_traveled = 0 # distance traveled by vehicle

	""" Run Analysis """ 
	def DrAnalysis(self, gt_pose, dr_pose):    
		# Append variables
		self.gt_x_list = np.append(self.gt_x_list, gt_pose[0], axis=0)
		self.gt_y_list = np.append(self.gt_y_list, gt_pose[1], axis=0)
		self.dr_x_list = np.append(self.dr_x_list, dr_pose[0], axis=0)
		self.dr_y_list = np.append(self.dr_y_list, dr_pose[1], axis=0)

		# Calculate RMSE
		x_rmse = mean_squared_error(self.gt_x_list, self.dr_x_list, squared=False) # calculate y rmse
		y_rmse = mean_squared_error(self.gt_y_list, self.dr_y_list, squared=False) # calculate x rmse
		self.dr_dist_error_rmse = Distance(x_rmse, y_rmse) # calculate distance rmse

		# Compute distance traveled by the vehicle
		dx = EuclideanDistance(gt_pose[0], gt_pose[1], self.gt_pose_prev[0], self.gt_pose_prev[1])
		self.dr_dist_traveled += dx
		
		# Update gt pose
		self.gt_pose_prev[0] = gt_pose[0]
		self.gt_pose_prev[1] = gt_pose[1]
		self.gt_pose_prev[2] = gt_pose[2]

		return self.dr_dist_traveled, self.dr_dist_error_rmse

	def EstAnalysis(self, gt_pose, est_pose):    
		# Append variables
		self.gt_est_x_list = np.append(self.gt_est_x_list, gt_pose[0], axis=0)
		self.gt_est_y_list = np.append(self.gt_est_y_list, gt_pose[1], axis=0)
		self.est_x_list = np.append(self.est_x_list, est_pose[0], axis=0)
		self.est_y_list = np.append(self.est_y_list, est_pose[1], axis=0)

		# Calculate RMSE
		x_rmse = mean_squared_error(self.gt_est_x_list, self.est_x_list, squared=False) # calculate y rmse
		y_rmse = mean_squared_error(self.gt_est_y_list, self.est_y_list, squared=False) # calculate x rmse
		self.est_dist_error_rmse = Distance(x_rmse, y_rmse) # calculate distance

		# Compute distance traveled by the vehicle
		dx = EuclideanDistance(gt_pose[0], gt_pose[1], self.gt_est_pose_prev[0], self.gt_est_pose_prev[1])
		self.est_dist_traveled += dx
		
		# Update gt pose
		self.gt_est_pose_prev[0] = gt_pose[0]
		self.gt_est_pose_prev[1] = gt_pose[1]
		self.gt_est_pose_prev[2] = gt_pose[2]

		return self.est_dist_traveled, self.est_dist_error_rmse
