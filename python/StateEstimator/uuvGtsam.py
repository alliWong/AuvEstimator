#!/usr/bin/env python
# pylint: disable=invalid-name, E1101

# """
# GTSAM iSAM2 implementation
# """
import sys
sys.path.append("/usr/local/")

import gtsam
import heapq
import time
import numpy as np
from collections import deque
from gtsam.symbol_shorthand import B, V, X
from gtsam_auv import PriorFactorPose3Z, PriorFactorVel

def gtsam_pose_to_numpy(gtsam_pose):
	"""Convert GTSAM pose to numpy arrays (position, orientation)"""
	position = np.array([gtsam_pose.x(),
						 gtsam_pose.y(),
						 gtsam_pose.z()])
	q = gtsam_pose.rotation().quaternion()
	orientation = np.array([q[1], q[2], q[3], q[0]]) # xyzw
	return position, orientation

def numpy_pose_to_gtsam(position, orientation):
	"""Convert numpy arrays (position, orientation) to GTSAM pose"""
	return gtsam.Pose3(gtsam.Rot3.Quaternion(orientation[3],
											 orientation[0],
											 orientation[1],
											 orientation[2]),
								gtsam.Point3(position[0],
											 position[1],
											 position[2]))

class GtsamEstimator():
	""" ISAM2 Fusion"""
	def __init__(self, params):
		""" Initialize variables """
		self.imu_meas_predict = [] # IMU measurements for pose prediction
		self.imu_opt_meas = [] # IMU measurements for pose prediction between measurement updates
		self.imu_samples = [] # imu samples
		self.opt_meas = [] # optimizes measurements
		self.__opt_meas_buffer_time = params['opt_meas_buffer_time']

		""" IMU Preintegration """
		# IMU preintegration parameters
		self.imu_preint_params = gtsam.PreintegrationParams(np.asarray(params['g']))
		self.imu_preint_params.setAccelerometerCovariance(np.eye(3) * np.power(params['acc_nd_sigma'], 2))
		self.imu_preint_params.setGyroscopeCovariance(np.eye(3) * np.power(params['acc_nd_sigma'], 2))
		self.imu_preint_params.setIntegrationCovariance(np.eye(3) * params['int_cov_sigma']**2)
		self.imu_preint_params.setUse2ndOrderCoriolis(params['setUse2ndOrderCoriolis'])
		self.imu_preint_params.setOmegaCoriolis(np.array(params['omega_coriolis']))

		""" Initialize Parameters """
		# ISAM2 initialization
		isam2_params = gtsam.ISAM2Params()
		isam2_params.setRelinearizeThreshold(params['relinearize_th'])
		isam2_params.setRelinearizeSkip(params['relinearize_skip'])
		isam2_params.setFactorization(params['factorization'])
		# self.isam2 = gtsam.ISAM2(isam2_params)
		self.isam2 = gtsam.ISAM2()
		self.new_factors = gtsam.NonlinearFactorGraph()
		self.new_initial_ests = gtsam.Values()
		self.min_imu_sample = 2 # minimum imu sample count needed for integration
		# ISAM2 keys
		self.poseKey = X(0)
		self.velKey = V(0)
		self.biasKey = B(0)

		""" Set initial variables """
		# Initial state
		self.last_opt_time = self.current_time = 0
		self.current_global_pose = numpy_pose_to_gtsam(
			params['init_pos'], 
			params['init_ori'])
		self.current_global_vel = np.asarray(
			params['init_vel'])
		self.current_bias = gtsam.imuBias.ConstantBias(
			np.asarray(params['init_acc_bias']),
			np.asarray(params['init_gyr_bias']))
		self.imu_accum = gtsam.PreintegratedImuMeasurements(self.imu_preint_params)
		# uncertainty of the initial state
		self.init_pos_cov = gtsam.noiseModel.Isotropic.Sigmas(np.array([
			params['init_ori_cov'], params['init_ori_cov'], params['init_ori_cov'],
			params['init_pos_cov'], params['init_pos_cov'], params['init_pos_cov']]))
		self.init_vel_cov = gtsam.noiseModel.Isotropic.Sigmas(np.array([
			params['init_vel_cov'], params['init_vel_cov'], params['init_vel_cov']]))
		self.init_bias_cov = gtsam.noiseModel.Isotropic.Sigmas(np.array([
			params['init_acc_bias_cov'], params['init_acc_bias_cov'], params['init_acc_bias_cov'],
			params['init_gyr_bias_cov'], params['init_gyr_bias_cov'], params['init_gyr_bias_cov']]))
		# measurement noise
		self.dvl_cov = gtsam.noiseModel.Isotropic.Sigmas(np.asarray(
			params['dvl_cov']))
		self.bar_cov = gtsam.noiseModel.Isotropic.Sigmas(np.asarray(
			params['bar_cov']))
		self.bias_cov = gtsam.noiseModel.Isotropic.Sigmas(np.concatenate((
			params['sigma_acc_bias_evol'],
			params['sigma_gyr_bias_evol'])))

		""" Set initial state """
		# Initial factors
		prior_pose_factor = gtsam.PriorFactorPose3(
			self.poseKey,
			self.current_global_pose,
			self.init_pos_cov)
		self.new_factors.add(prior_pose_factor)
		prior_vel_factor = gtsam.PriorFactorVector(
			self.velKey,
			self.current_global_vel,
			self.init_vel_cov)
		self.new_factors.add(prior_vel_factor)
		prior_bias_factor = gtsam.PriorFactorConstantBias(
			self.biasKey,
			self.current_bias,
			self.init_bias_cov)
		self.new_factors.add(prior_bias_factor)
		# Initial estimates
		self.new_initial_ests.insert(self.poseKey, self.current_global_pose)
		self.new_initial_ests.insert(self.velKey, self.current_global_vel)
		self.new_initial_ests.insert(self.biasKey, self.current_bias)

	def AddBarMeasurement(self, time, position):
		""" Add barometer measurements 
		Inputs
			linear position: gloabl z axis direction linear position
		"""
		heapq.heappush(self.opt_meas, (time, 'bar', position))
		buffer_time = heapq.nlargest(1, self.opt_meas)[0][0] - heapq.nsmallest(1, self.opt_meas)[0][0]
		if buffer_time >= self.__opt_meas_buffer_time:
			self.measurement_update(heapq.heappop(self.opt_meas))

	def AddDvlMeasurement(self, time, b_velocity):
		""" Add DVL measurements 
		Inputs
			velocity: global frame velocity
			b_velocity: body frame velocity
		"""
		heapq.heappush(self.opt_meas, (time, 'dvl', b_velocity))
		buffer_time = heapq.nlargest(1, self.opt_meas)[0][0] - heapq.nsmallest(1, self.opt_meas)[0][0]
		if buffer_time >= self.__opt_meas_buffer_time:
			self.measurement_update(heapq.heappop(self.opt_meas))

	def AddImuMeasurement(self, time, linear_acceleration, angular_velocity, dt):
		""" Add IMU measurement
		Inputs
			linear_acceleration: np.array([x, y, z])
			angular_velocity: np.array([x, y, z])
		Outputs
			position: np.array([x, y, z]) 
			orientation: np.array([qx, qy, qz, qw])
			velocity: np.array([vx, vy, vz])
			acc_bias: np.array([bax, bay, baz])
			gyr_bias: np.array([bgx, bgy, bgz])
		"""
		heapq.heappush(self.imu_meas_predict, (time, linear_acceleration, angular_velocity, dt))
		heapq.heappush(self.imu_opt_meas, (time, linear_acceleration, angular_velocity, dt))
		# Process sample
		return self.ImuPredict()

	def measurement_update(self, measurement):
		""" Update factors based on measurement factor type """
		meas_time = measurement[0]
		meas_type = measurement[1]
		imu_samples = []
		while True:
			if not self.imu_opt_meas:
				break
			sample = heapq.heappop(self.imu_opt_meas)
			if sample[0] < meas_time:
				imu_samples.append(sample)
			else:
				break
		if len(imu_samples) < self.min_imu_sample:
			# Must have (minimum 2) new IMU measurements since last measurement update
			# If not, put samples back and ignore this measurement
			for sample in imu_samples:
				heapq.heappush(self.imu_opt_meas, sample)
			print("Ignoring measurement at: {}".format(meas_time))
			return
		# Update keys
		self.poseKey += 1
		self.velKey += 1
		self.biasKey += 1
		# Add barometer position z factor
		if meas_type == 'bar':
			bar_posZ = measurement[2]
			bar_factor = PriorFactorPose3Z(
				self.poseKey,
				bar_posZ,
				self.bar_cov)
			self.new_factors.add(bar_factor)
		# Add dvl velocity factor
		elif meas_type == 'dvl':
			b_dvl_vel = measurement[2]
			dvl_factor = PriorFactorVel(
				self.poseKey,
				self.velKey,
				b_dvl_vel,
				self.dvl_cov)
			self.new_factors.add(dvl_factor)
		# Optimize measurements
		self.Optimize(meas_time, imu_samples)

	def ImuPredict(self):
		""" Predict NavState with IMU """
		if self.current_time > self.last_opt_time: # when new optimized pose is available
			# store state at measurement time
			self.last_opt_time = self.current_time
			# reset integration
			self.imu_accum.resetIntegration()
			#print("Old IMU, New IMU, opt: {}, {}, {}".format(imu_samples[0][0], imu_samples[-1][0], last_opt_time))
			new_imu_samples = []
			for sample in self.imu_samples:
				if sample[0] > self.last_opt_time:
					self.imu_accum.integrateMeasurement(sample[1], sample[2], sample[3])
					new_imu_samples.append(sample)
			self.imu_samples = new_imu_samples
		# Extract new samples from queue
		(time, linear_acceleration, angular_velocity, dt) = heapq.heappop(self.imu_meas_predict)
		# Store sample for integration when new measurement is available
		self.imu_samples.append((time, linear_acceleration, angular_velocity, dt))
		# Integrate
		self.imu_accum.integrateMeasurement(linear_acceleration, angular_velocity, dt)
		# Compute NavState
		predicted_nav_state = self.imu_accum.predict(gtsam.NavState(self.current_global_pose, self.current_global_vel), self.current_bias)
		# extract position and orientation from NavState
		pos, ori = gtsam_pose_to_numpy(predicted_nav_state.pose()) 
		return (pos, ori, self.current_global_vel, self.current_bias.accelerometer(), self.current_bias.gyroscope())

	def ImuUpdate(self, imu_samples):
		""" Update NavState using IMU factor """
		# Reset integration
		imu_accum = gtsam.PreintegratedImuMeasurements(self.imu_preint_params)
		# Integrate IMU measurements at measurement time
		for sample in imu_samples:
			imu_accum.integrateMeasurement(sample[1], sample[2], sample[3])
		# Compute NavState
		predicted_nav_state = imu_accum.predict(gtsam.NavState(self.current_global_pose, self.current_global_vel), self.current_bias)
		# Add IMU factor
		imu_factor = gtsam.ImuFactor(
			self.poseKey-1, self.velKey-1,
			self.poseKey, self.velKey,
			self.biasKey, imu_accum)
		self.new_factors.add(imu_factor)
		return predicted_nav_state

	def Optimize(self, meas_time, imu_samples):
		"""Perform optimization"""
		# IMU preintegration until measurement time
		predicted_nav_state = self.ImuUpdate(imu_samples)
		# Add estimates
		self.new_initial_ests.insert(self.poseKey, predicted_nav_state.pose())
		self.new_initial_ests.insert(self.velKey, predicted_nav_state.velocity())
		self.new_initial_ests.insert(self.biasKey, self.current_bias)
		# Add IMU bias factor
		bias_factor = gtsam.BetweenFactorConstantBias(
			self.biasKey-1, self.biasKey, gtsam.imuBias.ConstantBias(), self.bias_cov)
		self.new_factors.add(bias_factor)
		# Compute result
		result = self.Isam2Update()
		if result:
			self.current_time = meas_time
			self.current_global_pose = result.atPose3(self.poseKey)
			self.current_global_vel = result.atVector(self.velKey)
			self.current_bias = result.atConstantBias(self.biasKey)

	def Isam2Update(self):
		"""ISAM2 update and pose estimation"""
		result = None
		try:
			# ISAM2 update
			self.isam2.update(self.new_factors, self.new_initial_ests)
			result = self.isam2.calculateEstimate()
		except IndexError as e:
			print("Index error in optimization: {}".format(e))
		except TypeError as e:
			print("Type error in optimization: {}".format(e))
		# Reset
		self.new_factors.resize(0)
		self.new_initial_ests.clear()
		return result