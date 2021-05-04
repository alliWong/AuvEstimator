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
import plots

def compute_results_error(data, use_fgo, use_gt, use_dr, use_ekf, use_depth):
	""" Input is a dict containing the input and output """
	# convert list to np array for easier operation
	data_np = {}
	for meas_type, samples in data.items():
		if samples:
			data_np[meas_type] = np.asarray(samples)
	if not data_np:
		print("Nothing to compute..")
		return
	# find min time
	min_time = None
	for meas_type, sample_array in data_np.items():
		if not min_time or sample_array[0, 0] < min_time:
			min_time = sample_array[0, 0]

	""" Map Linear Position """
	### X ###
	for meas_type, sample_array in data_np.items():
		if (meas_type == "dr" and use_dr):
			dr_x = sample_array[:, 1]
		elif (meas_type == "dr_gt" and use_dr):
			dr_gt_x = sample_array[:, 1]
		elif (meas_type == "ekfEst" and use_ekf):
			ekf_x = sample_array[:, 1]
		elif (meas_type == "ekf_gt" and use_ekf):
			ekf_gt_x = sample_array[:, 1]
		elif (meas_type == "fgo" and use_fgo):
			fgo_x = sample_array[:, 1]
		elif (meas_type == "fgo_gt" and use_fgo):
			fgo_gt_x = sample_array[:, 1]
	### Y ###
	for meas_type, sample_array in data_np.items():
		if (meas_type == "dr" and use_dr):
			dr_y = sample_array[:, 2]
		elif (meas_type == "dr_gt" and use_dr):
			dr_gt_y = sample_array[:, 2]
		elif (meas_type == "ekfEst" and use_ekf):
			ekf_y = sample_array[:, 2]
		elif (meas_type == "ekf_gt" and use_ekf):
			ekf_gt_y = sample_array[:, 2]
		elif (meas_type == "fgo" and use_fgo):
			fgo_y = sample_array[:, 2]
		elif (meas_type == "fgo_gt" and use_fgo):
			fgo_gt_y = sample_array[:, 2]
	### Z ###
	for meas_type, sample_array in data_np.items():
		if (meas_type == "dr" and use_dr):
			dr_z = (sample_array[:, 0] - min_time, sample_array[:, 3])
		elif (meas_type == "dr_gt" and use_dr):
			dr_gt_z = (sample_array[:, 0] - min_time, sample_array[:, 3])
		elif (meas_type == "ekfEst" and use_ekf):
			ekf_z = (sample_array[:, 0] - min_time, sample_array[:, 3])
		elif (meas_type == "ekf_gt" and use_ekf):
			est_gt_z = (sample_array[:, 0] - min_time, sample_array[:, 3])
		elif (meas_type == "fgo" and use_fgo):
			fgo_z = sample_array[:, 3]
		elif (meas_type == "fgo_gt" and use_fgo):
			fgo_gt_z = sample_array[:, 3]

	""" Angular position """
	### Roll ###
	for meas_type, sample_array in data_np.items():
		if (meas_type == "dr" and use_dr):
			dr_roll = sample_array[:, 4]
		elif (meas_type == "dr_gt" and use_dr):
			dr_gt_roll = sample_array[:, 4]
		elif (meas_type == "ekfEst" and use_ekf):
			ekf_roll = sample_array[:, 4]
		elif (meas_type == "ekf_gt" and use_ekf):
			ekf_gt_roll = sample_array[:, 4]
		elif (meas_type == "fgo" and use_fgo):
			fgo_roll = sample_array[:, 4]
		elif (meas_type == "fgo_gt" and use_fgo):
			fgo_gt_roll = sample_array[:, 4]
	### Pitch ###
	for meas_type, sample_array in data_np.items():
		if (meas_type == "dr" and use_dr):
			dr_pitch = sample_array[:, 5]
		elif (meas_type == "dr_gt" and use_dr):
			dr_gt_pitch = sample_array[:, 5]
		elif (meas_type == "ekfEst" and use_ekf):
			ekf_pitch = sample_array[:, 5]
		elif (meas_type == "ekf_gt" and use_ekf):
			ekf_gt_pitch = sample_array[:, 5]
		elif (meas_type == "fgo" and use_fgo):
			fgo_pitch = sample_array[:, 5]
		elif (meas_type == "fgo_gt" and use_fgo):
			fgo_gt_pitch = sample_array[:, 5]
	### Yaw ###
	for meas_type, sample_array in data_np.items():
		if (meas_type == "dr" and use_dr):
			dr_yaw = sample_array[:, 6]
		elif (meas_type == "dr_gt" and use_dr):
			dr_gt_yaw = sample_array[:, 6]
		elif (meas_type == "ekfEst" and use_ekf):
			ekf_yaw = sample_array[:, 6]
		elif (meas_type == "ekf_gt" and use_ekf):
			ekf_gt_yaw = sample_array[:, 6]
		elif (meas_type == "fgo" and use_fgo):
			fgo_yaw = sample_array[:, 6]
		elif (meas_type == "fgo_gt" and use_fgo):
			fgo_gt_yaw = sample_array[:, 6]

	""" Linear Velocity """
	### X Linear velocity ###
	for meas_type, sample_array in data_np.items():
		if (meas_type == "dr" and use_dr):
			dr_vx = (sample_array[:, 0] - min_time, sample_array[:, 7])
		elif (meas_type == "dr_gt" and use_dr):
			dr_gt_vx = (sample_array[:, 0] - min_time, sample_array[:, 7])
		elif (meas_type == "ekfEst" and use_ekf):
			ekf_vx = (sample_array[:, 0] - min_time, sample_array[:, 7])
		elif (meas_type == "ekf_gt" and use_ekf):
			est_gt_vx = (sample_array[:, 0] - min_time, sample_array[:, 7])
	### Y Linear velocity  ###
	for meas_type, sample_array in data_np.items():
		if (meas_type == "dr" and use_dr):
			dr_vy = (sample_array[:, 0] - min_time, sample_array[:, 8])
		elif (meas_type == "dr_gt" and use_dr):
			dr_gt_vy = (sample_array[:, 0] - min_time, sample_array[:, 8])
		elif (meas_type == "ekfEst" and use_ekf):
			ekf_vy = (sample_array[:, 0] - min_time, sample_array[:, 8])
		elif (meas_type == "ekf_gt" and use_ekf):
			est_gt_vy = (sample_array[:, 0] - min_time, sample_array[:, 8])
	### Z Linear velocity  ###
	for meas_type, sample_array in data_np.items():
		if (meas_type == "dr" and use_dr):
			dr_vz = (sample_array[:, 0] - min_time, sample_array[:, 9])
		elif (meas_type == "dr_gt" and use_dr):
			dr_gt_vz = (sample_array[:, 0] - min_time, sample_array[:, 9])
		elif (meas_type == "ekfEst" and use_ekf):
			ekf_vz = (sample_array[:, 0] - min_time, sample_array[:, 9])
		elif (meas_type == "ekf_gt" and use_ekf):
			est_gt_vz = (sample_array[:, 0] - min_time, sample_array[:, 9])

	""" Error Analysis """
<<<<<<< HEAD:src/StateEstimator/errorAnalysis.py
	if use_dr:	
		### DR ###
		# 1) Calculate DR distance RMSE
		errorVecDrMapX = np.abs(dr_x - dr_gt_x)
		errorVecDrMapY = np.abs(dr_y - dr_gt_y)
		absMeanErrorDrMapX = np.mean(errorVecDrMapX)
		absMeanErrorDrMapY = np.mean(errorVecDrMapY)
		absErrorMeanDrMapXY = np.sqrt(absMeanErrorDrMapX**2 + absMeanErrorDrMapY**2)
		# 2) Calculate DR std
		absMeanErrorStdDrMapX = np.std(errorVecDrMapX)
		absMeanErrorStdDrMapY = np.std(errorVecDrMapY)
		absErrorMeanStdDrMapXY = np.sqrt(absMeanErrorStdDrMapX**2 + absMeanErrorStdDrMapY**2)
		# 3) Calculate DR rotations RMSE
		errorVecDrRoll = np.abs((dr_roll - dr_gt_roll)**2)
		errorVecDrMapPitch = np.abs((dr_pitch - dr_gt_pitch)**2)
		errorVecDrMapYaw = np.abs((dr_yaw - dr_gt_yaw)**2)
		absMeanErrorDrRoll = np.mean(errorVecDrRoll)
		absMeanErrorDrPitch = np.mean(errorVecDrMapPitch)
		absMeanErrorDrYaw = np.mean(errorVecDrMapYaw)
		# 4) Calculate DR std
		absMeanErrorStdDrMapX = np.std(errorVecDrMapX)
		absMeanErrorStdDrMapY = np.std(errorVecDrMapY)
		absErrorMeanStdDrMapXY = np.sqrt(absMeanErrorStdDrMapX**2 + absMeanErrorStdDrMapY**2)
		# 5) Print results
		print('absErrorMeanDrMapXY', absErrorMeanDrMapXY)
		print('absErrorMeanStdDrMapXY', absErrorMeanStdDrMapXY)
		print('absMeanErrorDrRoll', absMeanErrorDrRoll)
		print('absMeanErrorDrPitch', absMeanErrorDrPitch)
		print('absMeanErrorDrYaw', absMeanErrorDrYaw)

	if use_ekf:
		### EKF ###
		# 1) Calculate EKF distance RMSE
		errorVecEkfMapX = np.abs(ekf_x - ekf_gt_x)
		errorVecEkfMapY = np.abs(ekf_y - ekf_gt_y)
		absMeanErrorEkfMapX = np.mean(errorVecEkfMapX)
		absMeanErrorEkfMapY = np.mean(errorVecEkfMapY)
		absErrorMeanEkfMapXY = np.sqrt(absMeanErrorEkfMapX**2 + absMeanErrorEkfMapY**2)
		# 2) Calculate EKF std
		absMeanErrorStdEkfMapX = np.std(errorVecEkfMapX)
		absMeanErrorStdEkfMapY = np.std(errorVecEkfMapY)
		absErrorMeanStdEkfMapXY = np.sqrt(absMeanErrorStdEkfMapX**2 + absMeanErrorStdEkfMapY**2)
		# 3) Calculate EKF rotations RMSE
		errorVecEkfRoll = np.abs((ekf_roll - ekf_gt_roll)**2)
		errorVecEkfMapPitch = np.abs((ekf_pitch - ekf_gt_pitch)**2)
		errorVecEkfMapYaw = np.abs((ekf_yaw - ekf_gt_yaw)**2)
		absMeanErrorEkfRoll = np.mean(errorVecEkfRoll)
		absMeanErrorEkfPitch = np.mean(errorVecEkfMapPitch)
		absMeanErrorEkfYaw = np.mean(errorVecEkfMapYaw)
		# Print results
		print('absErrorMeanEkfMapXY', absErrorMeanEkfMapXY)
		print('absErrorMeanStdEkfMapXY', absErrorMeanStdEkfMapXY)
		print('absMeanErrorEkfRoll', absMeanErrorEkfRoll)
		print('absMeanErrorEkfPitch', absMeanErrorEkfPitch)
		print('absMeanErrorEkfYaw', absMeanErrorEkfYaw)
		# print Error change
		errorChange =((absErrorMeanDrMapXY-absErrorMeanEkfMapXY)/absErrorMeanDrMapXY)*100
		print('DrToEstMeanErrorChange', errorChange)

	if use_fgo:
		### FGO ###
		# 1) Calculate FGO distance RMSE
		errorVecFgoMapX = np.abs(fgo_x - fgo_gt_x)
		errorVecFgoMapY = np.abs(fgo_y - fgo_gt_y)
		absMeanErrorFgoMapX = np.mean(errorVecFgoMapX)
		absMeanErrorFgoMapY = np.mean(errorVecFgoMapY)
		absErrorMeanFgoMapXY = np.sqrt(absMeanErrorFgoMapX**2 + absMeanErrorFgoMapY**2)
		# 2) Calculate FGO std
		absMeanErrorStdFgoMapX = np.std(errorVecFgoMapX)
		absMeanErrorStdFgoMapY = np.std(errorVecFgoMapY)
		absErrorMeanStdFgoMapXY = np.sqrt(absMeanErrorStdFgoMapX**2 + absMeanErrorStdFgoMapY**2)
		# 3) Calculate FGO rotations RMSE
		errorVecFgoRoll = np.abs((fgo_roll - fgo_gt_roll)**2)
		errorVecFgoMapPitch = np.abs((fgo_pitch - fgo_gt_pitch)**2)
		errorVecFgoMapYaw = np.abs((fgo_yaw - fgo_gt_yaw)**2)
		absMeanErrorFgoRoll = np.mean(errorVecFgoRoll)
		absMeanErrorFgoPitch = np.mean(errorVecFgoMapPitch)
		absMeanErrorFgoYaw = np.mean(errorVecFgoMapYaw)
		# Print results
		print('absErrorMeanFgoMapXY', absErrorMeanFgoMapXY)
		print('absErrorMeanStdEkfMapXY', absErrorMeanStdFgoMapXY)
		print('absMeanErrorFgoRoll', absMeanErrorFgoRoll)
		print('absMeanErrorFgoPitch', absMeanErrorFgoPitch)
		print('absMeanErrorFgoYaw', absMeanErrorFgoYaw)
		# # print Error change
		# errorChange = ((absErrorMeanDrMapXY-absErrorMeanFgoMapXY)/absErrorMeanDrMapXY)*100
		# print('DrToFgoMeanErrorChange', errorChange)
=======
	# ### DR ###
	# # 1) Calculate DR distance RMSE
	# errorVecDrMapX = np.abs(dr_x - dr_gt_x)
	# errorVecDrMapY = np.abs(dr_y - dr_gt_y)
	# absMeanErrorDrMapX = np.mean(errorVecDrMapX)
	# absMeanErrorDrMapY = np.mean(errorVecDrMapY)
	# absErrorMeanDrMapXY = np.sqrt(absMeanErrorDrMapX**2 + absMeanErrorDrMapY**2)
	# # 2) Calculate DR std
	# absMeanErrorStdDrMapX = np.std(errorVecDrMapX)
	# absMeanErrorStdDrMapY = np.std(errorVecDrMapY)
	# absErrorMeanStdDrMapXY = np.sqrt(absMeanErrorStdDrMapX**2 + absMeanErrorStdDrMapY**2)
	# # 3) Calculate DR rotations RMSE
	# errorVecDrRoll = np.abs((dr_roll - dr_gt_roll)**2)
	# errorVecDrMapPitch = np.abs((dr_pitch - dr_gt_pitch)**2)
	# errorVecDrMapYaw = np.abs((dr_yaw - dr_gt_yaw)**2)
	# absMeanErrorDrRoll = np.mean(errorVecDrRoll)
	# absMeanErrorDrPitch = np.mean(errorVecDrMapPitch)
	# absMeanErrorDrYaw = np.mean(errorVecDrMapYaw)
	# # 4) Calculate DR std
	# absMeanErrorStdDrMapX = np.std(errorVecDrMapX)
	# absMeanErrorStdDrMapY = np.std(errorVecDrMapY)
	# absErrorMeanStdDrMapXY = np.sqrt(absMeanErrorStdDrMapX**2 + absMeanErrorStdDrMapY**2)
	# # 5) Print results
	# print('absErrorMeanDrMapXY', absErrorMeanDrMapXY)
	# print('absErrorMeanStdDrMapXY', absErrorMeanStdDrMapXY)
	# print('absMeanErrorDrRoll', absMeanErrorDrRoll)
	# print('absMeanErrorDrPitch', absMeanErrorDrPitch)
	# print('absMeanErrorDrYaw', absMeanErrorDrYaw)

	# ### EKF ###
	# # 1) Calculate EKF distance RMSE
	# errorVecEkfMapX = np.abs(ekf_x - ekf_gt_x)
	# errorVecEkfMapY = np.abs(ekf_y - ekf_gt_y)
	# absMeanErrorEkfMapX = np.mean(errorVecEkfMapX)
	# absMeanErrorEkfMapY = np.mean(errorVecEkfMapY)
	# absErrorMeanEkfMapXY = np.sqrt(absMeanErrorEkfMapX**2 + absMeanErrorEkfMapY**2)
	# # 2) Calculate EKF std
	# absMeanErrorStdEkfMapX = np.std(errorVecEkfMapX)
	# absMeanErrorStdEkfMapY = np.std(errorVecEkfMapY)
	# absErrorMeanStdEkfMapXY = np.sqrt(absMeanErrorStdEkfMapX**2 + absMeanErrorStdEkfMapY**2)
	# # 3) Calculate EKF rotations RMSE
	# errorVecEkfRoll = np.abs((ekf_roll - ekf_gt_roll)**2)
	# errorVecEkfMapPitch = np.abs((ekf_pitch - ekf_gt_pitch)**2)
	# errorVecEkfMapYaw = np.abs((ekf_yaw - ekf_gt_yaw)**2)
	# absMeanErrorEkfRoll = np.mean(errorVecEkfRoll)
	# absMeanErrorEkfPitch = np.mean(errorVecEkfMapPitch)
	# absMeanErrorEkfYaw = np.mean(errorVecEkfMapYaw)
	# # Print results
	# print('absErrorMeanEkfMapXY', absErrorMeanEkfMapXY)
	# print('absErrorMeanStdEkfMapXY', absErrorMeanStdEkfMapXY)
	# print('absMeanErrorEkfRoll', absMeanErrorEkfRoll)
	# print('absMeanErrorEkfPitch', absMeanErrorEkfPitch)
	# print('absMeanErrorEkfYaw', absMeanErrorEkfYaw)
	# # print Error change
	# errorChange =((absErrorMeanDrMapXY-absErrorMeanEkfMapXY)/absErrorMeanDrMapXY)*100
	# print('DrToEstMeanErrorChange', errorChange)

	### FGO ###
	# 1) Calculate FGO distance RMSE
	errorVecFgoMapX = np.abs(fgo_x - fgo_gt_x)
	errorVecFgoMapY = np.abs(fgo_y - fgo_gt_y)
	absMeanErrorFgoMapX = np.mean(errorVecFgoMapX)
	absMeanErrorFgoMapY = np.mean(errorVecFgoMapY)
	absErrorMeanFgoMapXY = np.sqrt(absMeanErrorFgoMapX**2 + absMeanErrorFgoMapY**2)
	# 2) Calculate FGO std
	absMeanErrorStdFgoMapX = np.std(errorVecFgoMapX)
	absMeanErrorStdFgoMapY = np.std(errorVecFgoMapY)
	absErrorMeanStdFgoMapXY = np.sqrt(absMeanErrorStdFgoMapX**2 + absMeanErrorStdFgoMapY**2)
	# 3) Calculate FGO rotations RMSE
	errorVecFgoRoll = np.abs((fgo_roll - fgo_gt_roll)**2)
	errorVecFgoMapPitch = np.abs((fgo_pitch - fgo_gt_pitch)**2)
	errorVecFgoMapYaw = np.abs((fgo_yaw - fgo_gt_yaw)**2)
	absMeanErrorFgoRoll = np.mean(errorVecFgoRoll)
	absMeanErrorFgoPitch = np.mean(errorVecFgoMapPitch)
	absMeanErrorFgoYaw = np.mean(errorVecFgoMapYaw)
	# Print results
	print('absErrorMeanFgoMapXY', absErrorMeanFgoMapXY)
	print('absErrorMeanStdEkfMapXY', absErrorMeanStdFgoMapXY)
	print('absMeanErrorFgoRoll', absMeanErrorFgoRoll)
	print('absMeanErrorFgoPitch', absMeanErrorFgoPitch)
	print('absMeanErrorFgoYaw', absMeanErrorFgoYaw)
	# Print Error change
	# errorChange = ((absErrorMeanDrMapXY-absErrorMeanFgoMapXY)/absErrorMeanDrMapXY)*100
	# print('DrToFgoMeanErrorChange', errorChange)
>>>>>>> origin:src/2020.03.01_auvStateEstimator/errorAnalysis.py


	# ### DR ###
	# absErrorMeanDrMapXYList = []
	# for i in range(len(dr_gt_time)):
	# 	# 1) Calculate DR distance RMSE
	# 	dr_x_rmse = Rmse(dr_gt_x[i], dr_x[i])
	# 	dr_y_rmse = Rmse(dr_gt_y[i], dr_y[i])
	# 	absErrorMeanDrMapXY = Distance(dr_x_rmse, dr_y_rmse)

	# 	# errorVecDrMapX = np.abs(dr_x[i] - dr_gt_x[i])
	# 	# errorVecDrMapY = np.abs(dr_y[i] - dr_gt_y[i])
	# 	# absMeanErrorDrMapX = np.mean(errorVecDrMapX)
	# 	# absMeanErrorDrMapY = np.mean(errorVecDrMapY)
	# 	# absErrorMeanDrMapXY = np.sqrt(np.mean(errorVecDrMapX**2 + errorVecDrMapY**2))
	# 	# 2) Calculate DR std
	# 	# absMeanErrorStdDrMapX = np.std(errorVecDrMapX)
	# 	# absMeanErrorStdDrMapY = np.std(errorVecDrMapY)
	# 	# absErrorMeanStdDrMapXY = np.sqrt(absMeanErrorStdDrMapX**2 + absMeanErrorStdDrMapY**2)

	# 	absErrorMeanDrMapXYList.append(absErrorMeanDrMapXY)

	# ### EKF ###
	# absErrorMeanEkfMapXYList = []
	# for i in range(len(ekf_gt_time)):
	# 	# 1) Calculate DR distance RMSE
	# 	errorVecEkfMapX = np.abs(ekf_x[i] - ekf_gt_x[i])
	# 	errorVecEkfMapY = np.abs(ekf_y[i] - ekf_gt_y[i])
	# 	absMeanErrorEkfMapX = np.mean(errorVecEkfMapX)
	# 	absMeanErrorEkfMapY = np.mean(errorVecEkfMapY)
	# 	absErrorMeanEkfMapXY = np.sqrt(absMeanErrorEkfMapX**2 + absMeanErrorEkfMapY**2)
	# 	# 2) Calculate DR std
	# 	absMeanErrorStdEkfMapX = np.std(errorVecEkfMapX)
	# 	absMeanErrorStdEkfMapY = np.std(errorVecEkfMapY)
	# 	absErrorMeanStdEkfMapXY = np.sqrt(absMeanErrorStdEkfMapX**2 + absMeanErrorStdEkfMapY**2)
	# 	absErrorMeanEkfMapXYList.append(absErrorMeanEkfMapXY)


	# """ Plot """
	# plots.plot_error(dr_gt_time, ekf_gt_time, absErrorMeanDrMapXYList, absErrorMeanEkfMapXYList)



