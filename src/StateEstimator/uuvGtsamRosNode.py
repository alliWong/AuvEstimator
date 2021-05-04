#!/usr/bin/env python
# pylint: disable=invalid-name, E1101

# """
# ROS Wrapper: GTSAM state estimation
# """

import sys
import os
import rospy
import rosbag
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import PoseWithCovarianceStamped

import numpy as np
from numpy import array, zeros, reshape, matmul, eye, sqrt, cos, sin
import uuvGtsam
import plots
from sensor_msgs.msg import FluidPressure
from uuv_sensor_ros_plugins_msgs.msg import DVL
from transformations import quaternion_from_euler, euler_from_quaternion
from commons import PressureToDepth, Rot
from auv_estimator.msg import State, Covariance, Inputs, Estimator
from nav_msgs.msg import Odometry
import errorAnalysis

class GtsamEstRosNode():
	""" GTSAM ROS Interface """
	def __init__(self):
		rospy.init_node('GtsamEst', anonymous=True) # initialize GTSAM ROS node
		""" ROS parameters """
		# Grab topics
		self.gt_topic = rospy.get_param('~gt_topic', "/rexrov2/pose_gt") # GT messages
		self.dr_topic = rospy.get_param('~dr_topic', "/dr/pose") # DR messages
		self.ekf_topic = rospy.get_param('~ekf_topic', "/est/state") # EKF estimator messages
		self.imu_topic = rospy.get_param('~imu_topic', "/rexrov2/imu") # IMU messages
		self.depth_topic = rospy.get_param('~depth_topic', "/bar/depth") # processed barometer to depth messages
		self.dvl_topic = rospy.get_param('~dvl_topic', "/rexrov2/dvl") # DVL messages
		# DVL parameters
		self.sen_dvl_offsetX = rospy.get_param("~dvl_offsetX") # offset relative from the sensor to vehicle center of mass in x-direction
		self.sen_dvl_offsetY = rospy.get_param("~dvl_offsetY") # offset relative from the sensor to vehicle center of mass in y-direction
		self.sen_dvl_offsetZ = rospy.get_param("~dvl_offsetZ") # offset relative from the sensor to vehicle center of mass in z-direction
		# Grab bag file information
		self.bag_file_path = rospy.get_param('~bag_file_path', "") # if this path is set, data is read from the bag file via rosbag API as fast as possible
		self.bar_interval = rospy.get_param('~bar_interval', 0) # set zero to use all available barometer messages
		self.dvl_interval = rospy.get_param('~dvl_interval', 0) # set zero to use all available DVL messages
		self.dr_interval = rospy.get_param('~dr_interval', 0) # set zero to use all available dead reckoning messages
		self.ekf_interval = rospy.get_param('~ekf_interval', 0) # set zero to use all available EKF messages
		self.bag_secs_to_skip = rospy.get_param('~bag_secs_to_skip', 0.0) # skip "x" seconds in the beginning of bag file [s]
		self.fixed_dt = rospy.get_param('~fixed_dt', None) # fixed timestep (dt) for IMU samples (set None to use real time IMU timesteps)
		# Decide which sensors will be used for sensor fusion
		self.use_dvl = rospy.get_param('~use_dvl', False) # enable/disable the use of DVL messages
		self.use_bar = rospy.get_param('~use_bar', False) # enable/disable the use of barometer messages
		self.use_gt = rospy.get_param('~use_gt', False) # enable/disable the use of GT messages
		self.use_dr = rospy.get_param('~use_dr', False) # enable/disable the use of DR messages
		self.use_ekf = rospy.get_param('~use_ekf', False) # enable/disable the use of EKF estimator messages
		self.use_fgo = rospy.get_param('~use_fgo', False) # enable/disable the use of FGO estimator messages
		# Plot information
		self.plot_results = rospy.get_param('~plot_results', False) # plot data after processing the bag file
		self.save_dir = rospy.get_param('~save_dir', '/tmp') # save resulting images in a directory
		# Error analysis
		self.compute_error = rospy.get_param('~compute_error', False) # perform error analysis after processing the bag file
		# Publishers
		self.gtsam_pub = rospy.Publisher("gtsam/pose", PoseWithCovarianceStamped, queue_size=10)

		""" IMU Preintegration Parameters """
		params = {}
		# Minimum IMU sample count to accept as measurements for optimization
		params['opt_meas_buffer_time'] = rospy.get_param('~opt_meas_buffer_time', 0.3) # measurement optimization buffer size [s]
		# Optimization parameters
		params['relinearize_th'] = rospy.get_param('~relinearize_th', 0.01)
		params['relinearize_skip'] = rospy.get_param('~relinearize_skip', 10)
		params['factorization'] = rospy.get_param('~relinearize_skip', 'CHOLESKY')
		# IMU preintegration parameters
		params['g'] = rospy.get_param('~g', [0, 0, -9.81]) # gravity vector
		params['acc_nd_sigma'] = rospy.get_param('~acc_nd_sigma', [20, 20, 20]) # accelerometer noise density [m/s^2]
		params['gyro_nd_sigma'] = rospy.get_param('~gyro_nd_sigma', [300, 300, 300]) # gyroscope noise density [degrees/s]
		params['int_cov_sigma'] = rospy.get_param('~int_cov_sigma', 0) # associated error when integrating velocities for position
		params['setUse2ndOrderCoriolis'] = rospy.get_param('~setUse2ndOrderCoriolis', False) # set 2nd order Coriolis
		params['omega_coriolis'] = rospy.get_param('~omega_coriolis', [0, 0, 0]) # set omega Coriolis
		# initial vehicle state
		params['init_pos'] = rospy.get_param('~init_pos', [0, 0, -25]) # vehicle initial linear position [m]
		params['init_ori'] = rospy.get_param('~init_ori', [0, 0, 0, 1]) # vehicle initial angular position [quaternion]
		params['init_vel'] = rospy.get_param('~init_vel', [0, 0, 0]) # vehicle initial velocity [m/s]
		params['init_acc_bias'] = rospy.get_param('~init_acc_bias', [0, 0, 0]) # vehicle initial acceleration bias [m/s^2]
		params['init_gyr_bias'] = rospy.get_param('~init_gyr_bias', [0, 0, 0]) # vehicle initial gyroscope bias [degrees/s]
		# uncertainty of the initial vehicle state
		params['init_pos_cov'] = rospy.get_param('~init_pos_cov', 10) # initial vehicle initial linear position covariance [m]
		params['init_ori_cov'] = rospy.get_param('~init_ori_cov', 10) # initial vehicle initial angular position covariance [quaternion]
		params['init_vel_cov'] = rospy.get_param('~init_vel_cov', 10) # initial vehicle initial linear velocity covariance [m/s]
		params['init_acc_bias_cov'] = rospy.get_param('~init_acc_bias_cov', 1.0) # initial vehicle initial acceleration bias covariance [m/s^2]
		params['init_gyr_bias_cov'] = rospy.get_param('~init_gyr_bias_cov', 0.1) # initial vehicle initial gyroscope bias covariance [degrees/s]
		# measurement noise
		params['dvl_cov'] = rospy.get_param('~dvl_cov', [0.001, 0.001, 0.001]) # dvl sensor measurement covariance [m/s]
		params['bar_cov'] = rospy.get_param('~bar_cov', [0.2]) # barometer sensor measurement covariance [m]
		params['sigma_acc_bias_evol'] = rospy.get_param('~sigma_acc_bias_evol', [0.0004, 0.0004, 0.0004]) # linear acceleration sensor measurement covariance [m/s^2]
		params['sigma_gyr_bias_evol'] = rospy.get_param('~sigma_gyr_bias_evol', [0.0025, 0.0025, 0.0025]) # angular velocity sensor measurement covariance [degrees/s]

		""" Variables """
		# time variables
		self.is_initialized = False # enable/disable GTSAM fusion
		self.imu_last_update_time = None # enable/disable GTSAM fusion
		self.bar_last_update_time = None # IMU time since last update
		self.dvl_last_update_time = None # DVL time since last update
		self.dr_last_update_time = None # DR time since last update
		self.ekf_last_update_time = None # EKF time since last update
		# pose variables
		self.last_rbt_pose = None # last robot pose in map frame
		self.gt_pose = None # GT pose message
		self.ekf_pose = None # EKF pose message
		self.dr_pose = None # DR pose message
		self.fgo_pose = None # FGO pose message

		""" Analysis/Processing variables """
		# Plotting variables
		if self.plot_results:
			if self.use_bar:
				self.fusion_items = "BAR + IMU"
			elif self.use_dvl:
				self.fusion_items = "DVL + IMU"
			elif self.use_dvl and self.use_bar:
				self.fusion_items = "DVL + BAR + IMU"
			self.plot_data = {'IMU': [], 'GT': [], 'BAR': [], 'DVL': [], 'DR': [], 'EKF': [], self.fusion_items: []}

		# Error analysis variables
		if self.compute_error:
			self.error_results = {'dr': [], 'dr_gt': [], 'dr_bar': [], 'ekfEst': [], 'ekf_gt': [], 'ekf_bar': [], 'fgo': [], 'fgo_gt': []}

		""" Sensor variables """
		# DVL rigid frame transformation wrt IMU
		# DVL frame in ENU configuration(x-forward, z-upwards)
		self.sen_dvl_enuFrameRoll = np.deg2rad(0)
		self.sen_dvl_enuFramePitch = np.deg2rad(0) # -90
		self.sen_dvl_enuFrameYaw = np.deg2rad(0)
		self.frameTrans = Rot(self.sen_dvl_enuFrameRoll, self.sen_dvl_enuFramePitch, self.sen_dvl_enuFrameYaw) # compute for the corresponding rotation matrix
		self.dvl_offsetTransRbtLinVel = np.zeros(shape=(3,3)) # dvl frame transformation considering dvl linear position offset wrt IMU

		""" GTSAM ROS Wrapper """
		self.gtsam_fusion = uuvGtsam.GtsamEstimator(params) # initialize GTSAM fusion
		self.is_initialized = True # set GTSAM intialization to true

	def RunGtsam(self):
		""" Run GTSAM """
		# Run GTSAM using rosbag
		if self.bag_file_path: # initialize rosbag path if there is user input value
			rospy.loginfo("Processing file using rosbag: {}. Please wait..".format(self.bag_file_path))
			if self.bag_secs_to_skip > 0: # skips "x" seconds in the beginning of bag file if there is user input value
				rospy.loginfo("Skipping {} seconds from the start of rosbag.".format(self.bag_secs_to_skip))
			bag = rosbag.Bag(self.bag_file_path) # set rosbag variable
			total_time_secs = int(bag.get_end_time() - bag.get_start_time()) # compute total time of rosbag [s]
			init_t = None # set intial time
			last_info_time_secs = int(bag.get_start_time()) # update last time
			for topic, msg, t in bag.read_messages(topics=[self.gt_topic, self.dr_topic, self.ekf_topic, self.imu_topic, self.depth_topic, self.dvl_topic]):
				if not init_t:
					init_t = t
					continue
				if (t - init_t).to_sec() < self.bag_secs_to_skip:
					continue
				if rospy.is_shutdown():
					break
				elapsed_time_secs = int(t.to_sec() - bag.get_start_time())
				if elapsed_time_secs % 100 == 0 and elapsed_time_secs != last_info_time_secs:
					last_info_time_secs = elapsed_time_secs
					rospy.loginfo("Elapsed time: {}/{} [s]".format(elapsed_time_secs, total_time_secs))
				if topic == self.imu_topic:
					self.ImuCallback(msg)
				elif topic == self.gt_topic:
					self.GtCallback(msg)
				elif topic == self.dr_topic:
					self.DrCallback(msg)
				elif topic == self.ekf_topic:
					self.EkfCallback(msg)
				elif topic == self.depth_topic:
					self.BarCallback(msg)
				elif topic == self.dvl_topic:
					self.DvlCallback(msg)
			bag.close() # close bag file
			rospy.loginfo("Bag processed.")
			if self.compute_error: # compute error if initialized
				rospy.loginfo("Computing error. Please wait..")
				errorAnalysis.compute_results_error(self.error_results, self.use_fgo, self.use_gt, self.use_dr, self.use_ekf, self.use_bar)
			if self.plot_results: # plot data if initialized
				rospy.loginfo("Preparing plots. Please wait..")
				if not os.path.exists(self.save_dir):
					os.makedirs(self.save_dir)
				plots.plot_all(self.plot_data, self.fusion_items, self.save_dir, self.use_gt, self.use_dr, self.use_ekf, self.use_bar, self.use_dvl, self.use_fgo)
		# Run GTSAM using ROS subscribers (real time)
		else:
			# Subscribers
			self.sub_dvl = rospy.Subscriber(self.dvl_topic, DVL, self.DvlCallback) # DVL
			self.sub_imu = rospy.Subscriber(self.imu_topic, Imu, self.ImuCallback) # IMU
			self.sub_depth = rospy.Subscriber(self.depth_topic, Odometry, self.BarCallback) # depth
			rospy.spin()

	def ImuCallback(self, msg):
		""" IMU Callback messages """
		if self.imu_last_update_time:

			# Note: Raw imu angular position measurements is computed solely for the purpose of graphing
			# 		raw dvl sensor measurements in navigation frame. Angular position is not being
			# 		used in the factor graph.
			#		Input IMU data is in rotation matrix form.
			euler = euler_from_quaternion([msg.orientation.x,
										msg.orientation.y,
										msg.orientation.z,
										msg.orientation.w]) # Convert IMU data from quarternion to euler
			unwrapEuler = np.unwrap(euler) # unwrap euler angles
			imu_mapEulAng = np.array([[unwrapEuler[0]],
									[unwrapEuler[1]],
									[unwrapEuler[2]]]) # imu angular position array
			self.imu_mapAngPos = Rot(imu_mapEulAng[0], imu_mapEulAng[1], imu_mapEulAng[2])

			# Sensor measurements
			self.lin_acc = np.array([
				msg.linear_acceleration.x,
				msg.linear_acceleration.y,
				msg.linear_acceleration.z
			])
			self.ang_vel = np.array([
				msg.angular_velocity.x,
				msg.angular_velocity.y,
				msg.angular_velocity.z
			])
			dt = msg.header.stamp.to_sec() - self.imu_last_update_time
			if self.fixed_dt:
				dt = self.fixed_dt
			# add measurements to factor graph
			imu_pos, imu_ori, vel, acc_bias, gyr_bias = self.gtsam_fusion.AddImuMeasurement(
				msg.header.stamp.to_sec(), self.lin_acc, self.ang_vel, dt)
			# convert pose from IMU frame to robot frame
			rbt_pos = imu_pos
			rbt_ori = imu_ori
			# store internally
			euler_imu_ori = np.asarray(euler_from_quaternion(imu_ori)) / np.pi * 180.
			self.last_rbt_pose = (rbt_pos, rbt_ori)
			self.fgo_pose = np.concatenate((imu_pos, euler_imu_ori, vel), axis=0)
			# publish pose
			self.PublishPose(msg.header.stamp, rbt_pos, rbt_ori)

			# data for plots
			if self.plot_results:
				# store input
				self.plot_data['IMU'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec(), dt]), self.lin_acc, self.ang_vel), axis=0))
				# store output
				self.plot_data[self.fusion_items].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), imu_pos, euler_imu_ori, vel, acc_bias, gyr_bias), axis=0))

			# data for error analysis
			if self.compute_error:
				# grab estimator pose
				self.error_results['fgo'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.fgo_pose), axis=0))
				# grab estimator pose when ground truth is updated
				self.error_results['fgo_gt'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.gt_pose), axis=0))


		self.imu_last_update_time = msg.header.stamp.to_sec()

	def BarCallback(self, msg):
		""" Barometer Callback messages """
		if (not self.bar_last_update_time or
			msg.header.stamp.to_sec() - self.bar_last_update_time > self.bar_interval):
			self.bar_last_update_time = msg.header.stamp.to_sec()

			# sensor measurements
			self.depth = np.array([msg.pose.pose.position.z])

			# add measurements to factor graph
			if self.use_bar and self.use_fgo:
				self.gtsam_fusion.AddBarMeasurement(self.bar_last_update_time, self.depth)
			if not self.use_bar and self.use_fgo:
				return

			# data for plots
			if self.plot_results:
				self.plot_data['BAR'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.depth), axis=0))

	def DvlCallback(self, msg):
		""" DVL Callback messages """
		if (not self.dvl_last_update_time or
			msg.header.stamp.to_sec() - self.dvl_last_update_time > self.dvl_interval):
			self.dvl_last_update_time = msg.header.stamp.to_sec()

			# sensor measurements
			dvl_vel = np.array([msg.velocity.x,
								msg.velocity.y,
								msg.velocity.z])
			# dvl offset array
			dvl_offset = array([self.sen_dvl_offsetX,
								self.sen_dvl_offsetY,
								self.sen_dvl_offsetZ])
			# static frame transformation of DVL wrt IMU
			dvl_enuTransRbtLinVel = np.matmul(self.frameTrans, dvl_vel.T)
			# frame transformation considering dvl linear position offset wrt IMU
			self.dvl_offsetTransRbtLinVel = dvl_enuTransRbtLinVel - np.cross(self.ang_vel.T, dvl_offset).T
			# Convert velocity from robot frame into navigation frame (used for plotting purposes only)
			self.sen_dvl_mapLinVel = np.matmul(self.imu_mapAngPos, self.dvl_offsetTransRbtLinVel).T

			# add measurements to factor graph
			if self.use_dvl and self.use_fgo:
				self.gtsam_fusion.AddDvlMeasurement(self.dvl_last_update_time, dvl_enuTransRbtLinVel)
			if not self.use_dvl and self.use_fgo:
				return

			# data for plots
			if self.plot_results:
				self.plot_data['DVL'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.sen_dvl_mapLinVel), axis=0))

	def GtCallback(self, msg):
		""" GT Callback messages """
		# Change incoming ground truth quarternion data into euler [rad]
		gt_quaternion = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
		gt_euler = euler_from_quaternion(gt_quaternion)
		unwrapEuler = np.unwrap(gt_euler)

		# data
		self.gt_pose = np.array([msg.pose.pose.position.x,
							msg.pose.pose.position.y,
							msg.pose.pose.position.z,
							np.rad2deg(unwrapEuler[0]),
							np.rad2deg(unwrapEuler[1]),
							np.rad2deg(unwrapEuler[2]),
							msg.twist.twist.linear.x,
							msg.twist.twist.linear.y,
							msg.twist.twist.linear.z])

		# data for plots
		if self.plot_results:
			self.plot_data['GT'].append(
				np.concatenate((np.array([msg.header.stamp.to_sec()]), self.gt_pose), axis=0))
		if not self.use_gt:
			return

	def DrCallback(self, msg):
		""" DR Callback messages """
		if (not self.dr_last_update_time or
			msg.header.stamp.to_sec() - self.dr_last_update_time > self.dr_interval):
			self.dr_last_update_time = msg.header.stamp.to_sec()

			# data
			self.dr_pose = np.array([msg.state.x,
								msg.state.y,
								self.depth,
								msg.state.roll,
								msg.state.pitch,
								msg.state.yaw,
								msg.state.vx,
								msg.state.vy,
								msg.state.vz]) # dead reckoning array

			# data for plots
			if self.plot_results:
				self.plot_data['DR'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.dr_pose), axis=0))

			# data for error analysis
			if self.compute_error:
				# grab dead reckoning pose
				self.error_results['dr'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.dr_pose), axis=0))
				# grab ground truth pose when dead reckoning is updated
				self.error_results['dr_gt'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.gt_pose), axis=0))

	def EkfCallback(self, msg):
		""" EKF Callback messages """
		if (not self.ekf_last_update_time or
			msg.header.stamp.to_sec() - self.ekf_last_update_time > self.ekf_interval):
			self.ekf_last_update_time = msg.header.stamp.to_sec()

			# data
			self.ekf_pose = np.array([msg.state.x,
								msg.state.y,
								msg.state.z,
								msg.state.roll,
								msg.state.pitch,
								msg.state.yaw,
								msg.state.vx,
								msg.state.vy,
								msg.state.vz]) # EKF array

			# data for plots
			if self.plot_results:
				self.plot_data['EKF'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.ekf_pose), axis=0))

			# data for error analysis
			if self.compute_error:
				# grab estimator pose
				self.error_results['ekfEst'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.ekf_pose), axis=0))
				# grab estimator pose when ground truth is updated
				self.error_results['ekf_gt'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.gt_pose), axis=0))


	def PublishPose(self, stamp, position, orientation):
		"""Publish GTSAM pose """
		msg = PoseWithCovarianceStamped()
		msg.header.stamp = stamp
		msg.header.frame_id = "world"
		msg.pose.pose.position.x = position[0]
		msg.pose.pose.position.y = position[1]
		msg.pose.pose.position.z = position[2]
		msg.pose.pose.orientation.x = orientation[0]
		msg.pose.pose.orientation.y = orientation[1]
		msg.pose.pose.orientation.z = orientation[2]
		msg.pose.pose.orientation.w = orientation[3]
		self.gtsam_pub.publish(msg)

def main():
	"""Main"""
	node = GtsamEstRosNode()
	if node.is_initialized:
		node.RunGtsam()
	rospy.loginfo("Exiting..")

if __name__ == '__main__':
	main()