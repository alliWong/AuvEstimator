#!/usr/bin/env python 
"""
This file computes the x-error, y-error, z-error, and distance between 
the estimator and dead reckoning against ground truth.
"""
import sys
import numpy as np
from numpy import array, zeros
from commons import EuclideanDistance, Rmse, Distance
from sklearn.metrics import mean_squared_error
from threading import Timer

class ErrorAnalysis(object):
	def __init__(self):
		# GT variables
		self.gt_pose_prev = zeros(shape=(15,1)) # ground truth pose array
		self.gt_pose = zeros(shape=(15,1))
		# DR variables
		self.dr_pose_prev = zeros(shape=(15,1)) # ground truth pose array
		self.dr_pose = zeros(shape=(15,1))
		self.dr_rmse = zeros(shape=(15,1))
		self.dr_dist_error = 0 # dead reckoning distance error
		self.dr_error_pose = zeros(shape=(15,1)) # dead reckoning error pose array
		self.dr_dist_error_rmse = 0
		# Est variables
		self.est_pose = zeros(shape=(15,1))
		self.est_rmse = zeros(shape=(15,1))
		self.est_dist_error = 0 # estimator distance error
		self.est_error_pose = zeros(shape=(15,1)) # estimator error pose array
		self.est_dist_error_rmse = 0

		# Update flags
		self.est_update = 0
		self.dr_update = 0

		self.dist_traveled = 0 # distance traveled by vehicle
		self.gt_x_list = array([],dtype='int32')
		self.gt_y_list = array([],dtype='int32')
		self.dr_x_list = array([],dtype='int32')
		self.dr_y_list = array([],dtype='int32')

		self.gt_x_init_pose = 0
		self.dr_x_init_pose = 0
		self.gt_y_init_pose = 0
		self.dr_y_init_pose = 0

	""" Raw sensor measurements """
	def GtCallback(self, pose):
		self.gt_pose = pose 

		self.RunAnalysis()

		# Update gt pose
		self.gt_pose_prev[0] = self.gt_pose[0]
		self.gt_pose_prev[1] = self.gt_pose[1]
		self.gt_pose_prev[2] = self.gt_pose[2]

	def DrCallback(self, pose):
		self.dr_update = 1
		self.dr_pose = pose

		self.RunAnalysis()

		# Update dt pose
		self.dr_pose_prev[0] = self.dr_pose[0]
		self.dr_pose_prev[1] = self.dr_pose[1]
		self.dr_pose_prev[2] = self.dr_pose[2]

		# Update flag
		self.dr_update = 0

	def EstCallback(self, pose):
		self.est_update = 1
		self.est_pose = pose
		
		self.RunAnalysis()

		# Update flag
		self.est_update = 0

	""" Run Analysis """ 
	def RunAnalysis(self):    
		if self.dr_update == 1:
			# # EVALUATE: Compute the difference between the ground truth and DR coordinates
			print('EEEEEEEEEEEEEEEEEEEEEEEEEENDDDDDD')
			print('gt_x_list2', np.shape(self.gt_x_list))
			print('dr_x_list2', np.shape(self.dr_x_list))
			print('gt_y_list2', np.shape(self.gt_y_list))
			print('dr_y_list2', np.shape(self.dr_y_list))
			# print('EEEEEEEEEEEEEEEEEEEEEEEEEENDDDDDD')

			self.gt_x_list = np.append(self.gt_x_list, self.gt_pose[0], axis=0)
			self.gt_y_list = np.append(self.gt_y_list, self.gt_pose[1], axis=0)
			self.dr_x_list = np.append(self.dr_x_list, self.dr_pose[0], axis=0)
			self.dr_y_list = np.append(self.dr_y_list, self.dr_pose[1], axis=0)

			gt_x_len = len(self.gt_x_list)
			gt_y_len = len(self.gt_y_list)
			dr_x_len = len(self.dr_x_list)
			dr_y_len = len(self.dr_y_list)

			# # Absolute mean error
			x_rmse = Rmse(self.gt_x_list, self.dr_x_list)
			y_rmse = Rmse(self.gt_y_list, self.dr_y_list)
			self.dr_dist_error_rmse = Distance(x_rmse, y_rmse)
			self.OutputDr()
		
	def OutputDr(self):
		return self.dr_error_pose, self.dr_rmse, self.dr_dist_error, self.dr_dist_error_rmse

	def OutputEst(self):
		return self.est_error_pose, self.est_rmse, self.est_dist_error, self.est_dist_error_rmse
