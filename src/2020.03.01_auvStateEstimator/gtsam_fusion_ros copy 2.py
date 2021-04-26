#!/usr/bin/env python

# """
# ROS Wrapper: GTSAM state estimation
# """

import rospy
import rosbag
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import PoseWithCovarianceStamped

import numpy as np
from numpy import array, zeros, reshape, matmul, eye, sqrt, cos, sin
import gtsam_fusion_core
import plots
import sys
import os
from sensor_msgs.msg import FluidPressure
from uuv_sensor_ros_plugins_msgs.msg import DVL
from transformations import quaternion_from_euler, euler_from_quaternion
from commons import PressureToDepth, Rot
from auv_estimator.msg import State, Covariance, Inputs, Estimator
import errorAnalysis
from nav_msgs.msg import Odometry


class GtsamFusionRos():
	"""ROS interface to GTSAM FUSION"""
	def __init__(self):
		rospy.init_node('gtsam_fusion', anonymous=True)
		""" ROS parameters """
		# Grab topics
		self.gt_topic = rospy.get_param('~gt_topic', "/rexrov2/pose_gt") # 6DOF pose messages
		self.dr_topic = rospy.get_param('~dr_topic', "/dr/pose") # dead reckoning messages
		self.ekf_topic = rospy.get_param('~ekf_topic', "/est/state") # EKF estimator messages
		self.imu_topic = rospy.get_param('~imu_topic', "/rexrov2/imu") # IMU messages (only raw data is used, not the orientation)
		self.bar_topic = rospy.get_param('~bar_topic', "/bar/depth") # IMU messages (only raw data is used, not the orientation)
		self.dvl_topic = rospy.get_param('~dvl_topic', "/rexrov2/dvl") # IMU messages (only raw data is used, not the orientation)
		# Grab frames
		self.map_frame = rospy.get_param('~map_frame', 'map') # 6DOF pose measurements are in this local map frame
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
		self.bag_secs_to_skip = rospy.get_param('~bag_secs_to_skip', 0.0) # skip data from the start when reading from bag
		self.fixed_dt = rospy.get_param('~fixed_dt', None) # use this to set fixed dt for IMU samples (instead dt calculated from the time stamps)
		# Decide which sensors to be fused
		self.use_dvl = rospy.get_param('~use_dvl', False) # enable / disable the use of DVL messages
		self.use_bar = rospy.get_param('~use_bar', False) # enable / disable the use of barometer messages
		self.use_gt = rospy.get_param('~use_gt', False) # enable / disable the use of GT pose messages
		self.use_dr = rospy.get_param('~use_dr', False) # enable / disable the use of dead reckoning messages
		self.use_ekf = rospy.get_param('~use_ekf', False) # enable / disable the use of EKF estimator messages
		self.use_fgo = rospy.get_param('~use_fgo', False) # enable / disable the use of EKF estimator messages
		# Plot information
		self.plot_results = rospy.get_param('~plot_results', False) # plot results after the bag file is processed
		self.save_dir = rospy.get_param('~save_dir', '/tmp') # directory where the result images are saved
		# Error analysis
		self.compute_error = rospy.get_param('~compute_error', False) # compute error after the bag file is processed

		""" IMU Preintegration Parameters """
		params = {}
		# minimum IMU sample count to accept measurement for optimization
		params['opt_meas_buffer_time'] = rospy.get_param('~opt_meas_buffer_time', 0.3) # Buffer size in [s] for sensor pose measurements
		# optimization
		params['relinearize_th'] = rospy.get_param('~relinearize_th', 0.01)
		params['relinearize_skip'] = rospy.get_param('~relinearize_skip', 10)
		params['factorization'] = rospy.get_param('~relinearize_skip', 'CHOLESKY')
		# IMU preintegration
		params['g'] = rospy.get_param('~g', [0, 0, -9.81])
		params['acc_nd_sigma'] = rospy.get_param('~acc_nd_sigma', [18, 18, 18]) # accelerometer noise density [m/s^2]
		# params['gyro_nd_sigma'] = rospy.get_param('~gyro_nd_sigma', [0.005, 0.005, 0.005]) # gyroscope noise density
		params['gyro_nd_sigma'] = rospy.get_param('~gyro_nd_sigma', [6, 6, 6]) # gyroscope noise density
		params['int_cov_sigma'] = rospy.get_param('~int_cov_sigma', 0) # error associated when integrating position from velocities
		params['setUse2ndOrderCoriolis'] = rospy.get_param('~setUse2ndOrderCoriolis', False)
		params['omega_coriolis'] = rospy.get_param('~omega_coriolis', [0, 0, 0])
		# initial state (default values assume that the robot is statioary at origo)
		params['init_pos'] = rospy.get_param('~init_pos', [0, 0, -25])
		params['init_ori'] = rospy.get_param('~init_ori', [0, 0, 0, 1])
		params['init_vel'] = rospy.get_param('~init_vel', [0, 0, 0])
		params['init_acc_bias'] = rospy.get_param('~init_acc_bias', [0, 0, 0])
		params['init_gyr_bias'] = rospy.get_param('~init_gyr_bias', [0, 0, 0])
		# uncertainty of the initial state
		params['init_pos_cov'] = rospy.get_param('~init_pos_cov', 10)
		params['init_ori_cov'] = rospy.get_param('~init_ori_cov', 10)
		params['init_vel_cov'] = rospy.get_param('~init_vel_cov', 10)
		params['init_acc_bias_cov'] = rospy.get_param('~init_acc_bias_cov', 1.0)
		params['init_gyr_bias_cov'] = rospy.get_param('~init_gyr_bias_cov', 0.1)
		# measurement noise
		params['sigma_pose_pos'] = rospy.get_param('~sigma_pose_pos', [18, 18, 18]) # [m] error in 6DOF pose position
		# params['sigma_pose_rot'] = rospy.get_param('~sigma_pose_rot', [6, 6, 6]) # rpy [rad] error in 6DOF pose rotation
		params['sigma_pose_rot'] = rospy.get_param('~sigma_pose_rot', [np.inf, np.inf, 5.0/180.0*np.pi]) # rpy [rad] error in 6DOF pose rotation
		params['dvl_cov'] = rospy.get_param('~dvl_cov', [0.001, 0.001, 0.001]) # error in dvl position
		params['bar_cov'] = rospy.get_param('~bar_cov', [0.5]) # error in bar position
		params['sigma_acc_bias_evol'] = rospy.get_param('~sigma_acc_bias_evol', [4e-5, 4e-5, 4e-5])
		params['sigma_gyr_bias_evol'] = rospy.get_param('~sigma_gyr_bias_evol', [7e-4, 7e-4, 7e-4])

		# variables
		self.is_initialized = False
		self.imu_last_update_time = None
		self.bar_last_update_time = None
		self.dvl_last_update_time = None
		self.dr_last_update_time = None
		self.ekf_last_update_time = None
		self.last_rbt_pose = None
		self.gt_pose = None
		self.ekf_pose = None
		self.dr_pose = None

		if self.plot_results:
			if self.use_gt:
				self.fusion_items = "POSE + IMU"
			elif self.use_bar:
				self.fusion_items = "BAR + IMU"
			elif self.use_bar and self.use_gt:
				self.fusion_items = "POSE + BAR + IMU"
			elif self.use_dvl:
				self.fusion_items = "DVL + IMU"
			elif self.use_dvl and self.use_bar:
				self.fusion_items = "DVL + BAR + IMU"
			elif self.use_dvl and self.use_bar and self.use_gt:
				self.fusion_items = "DVL + BAR + IMU + POSE"
			self.results = {'IMU': [], 'GT': [], 'BAR': [], 'DVL': [], 'DR': [], 'EKF': [], self.fusion_items: []}

		if self.compute_error:
			self.error_results = {'dr': [], 'dr_gt': [], 'dr_bar': [], 'ekfEst': [], 'ekf_gt': [], 'ekf_bar': [], 'fgo': [], 'fgo_gt': []}

		#############################################################################
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
		#############################################################################

		# publishers
		self.__base_pose_in_map_100hz_pub = rospy.Publisher(
			"base_pose_in_map_100hz", PoseWithCovarianceStamped, queue_size=10)

		# Start fusion core
		self.__fusion_core = gtsam_fusion_core.GtsamFusionCore(params)
		self.is_initialized = True

	def run(self):
		"""Either process bag file via rosbag API or subscribe to topics"""
		if self.bag_file_path: # use rosbag API
			rospy.loginfo("Processing file using rosbag API: {}. Please wait..".format(self.bag_file_path))
			if self.bag_secs_to_skip > 0:
				rospy.loginfo("Skipping {} seconds from the start of the bag file.".format(self.bag_secs_to_skip))
			bag = rosbag.Bag(self.bag_file_path)
			total_time_secs = int(bag.get_end_time() - bag.get_start_time())
			init_t = None
			last_info_time_secs = int(bag.get_start_time())
			for topic, msg, t in bag.read_messages(topics=[self.gt_topic, self.dr_topic, self.ekf_topic, self.imu_topic, self.bar_topic, self.dvl_topic]):
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
				elif topic == self.bar_topic:
					self.BarCallback(msg)
				elif topic == self.dvl_topic:
					self.DvlCallback(msg)
			bag.close()
			rospy.loginfo("Bag processed.")
			if self.compute_error:
				rospy.loginfo("Computing error. Please wait..")
				errorAnalysis.compute_results_error(self.error_results, self.use_fgo, self.use_gt, self.use_dr, self.use_ekf, self.use_bar)
			if self.plot_results:
				rospy.loginfo("Preparing plots. Please wait..")
				if not os.path.exists(self.save_dir):
					os.makedirs(self.save_dir)
				plots.plot_all(self.results, self.fusion_items, self.save_dir, self.use_gt, self.use_dr, self.use_ekf, self.use_bar, self.use_dvl)
		else: # subscribe to topics
			self.sub_dvl = rospy.Subscriber(self.dvl_topic, DVL, self.DvlCallback) 
			self.sub_imu = rospy.Subscriber(self.imu_topic, Imu, self.ImuCallback) 
			self.sub_depth = rospy.Subscriber(self.bar_topic, Odometry, self.BarCallback)
			rospy.spin()

	def ImuCallback(self, msg):
		"""Handle IMU message"""
		if self.imu_last_update_time:
			# Convert to numpy
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

			#############################################################################
			# Setup map orientation array
			euler = euler_from_quaternion([msg.orientation.x,
										msg.orientation.y,
										msg.orientation.z,
										msg.orientation.w]) # Convert IMU data from quarternion to euler
			unwrapEuler = np.unwrap(euler) # unwrap euler angles
			imu_mapEulAng = np.array([[unwrapEuler[0]],
									[unwrapEuler[1]],
									[unwrapEuler[2]]]) # imu angular position array
			### NOTES: Incoming IMU data is in rotation matrix form  ###
			self.imu_mapAngPos = Rot(imu_mapEulAng[0], imu_mapEulAng[1], imu_mapEulAng[2])
			#############################################################################

			dt = msg.header.stamp.to_sec() - self.imu_last_update_time
			if self.fixed_dt:
				dt = self.fixed_dt
			# IMU update
			imu_pos, imu_ori, vel, acc_bias, gyr_bias = self.__fusion_core.AddImuMeasurement(
				msg.header.stamp.to_sec(), self.lin_acc, self.ang_vel, dt)
			# convert pose from IMU frame to robot frame
			rbt_pos = imu_pos
			rbt_ori = imu_ori
			# store internally
			euler_imu_ori = np.asarray(euler_from_quaternion(imu_ori)) / np.pi * 180.
			self.last_rbt_pose = (rbt_pos, rbt_ori)
			self.fgo_pose = np.concatenate((imu_pos, euler_imu_ori, vel), axis=0)
			# publish pose
			self.__publish_pose(
				msg.header.stamp,
				self.map_frame,
				rbt_pos,
				rbt_ori
			)
			# data for plots
			if self.plot_results:
				# store input
				self.results['IMU'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec(), dt]), self.lin_acc, self.ang_vel), axis=0))
				# store output
				self.results[self.fusion_items].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), imu_pos, euler_imu_ori, vel, acc_bias, gyr_bias), axis=0))

		self.imu_last_update_time = msg.header.stamp.to_sec()

	def BarCallback(self, msg):
		""" Handle BAR position Z message """
		if (not self.bar_last_update_time or
			msg.header.stamp.to_sec() - self.bar_last_update_time > self.bar_interval):
			self.bar_last_update_time = msg.header.stamp.to_sec()

			self.depth = np.array([msg.pose.pose.position.z])

			# data for plots
			if self.plot_results:
				self.results['BAR'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.depth), axis=0))
			if not self.use_bar:
				return
			if self.use_bar:
				self.__fusion_core.AddBarMeasurement(self.bar_last_update_time, self.depth)

	# def DvlCallback(self, msg):
	# 	if (not self.dvl_last_update_time or
	# 		msg.header.stamp.to_sec() - self.dvl_last_update_time > self.dvl_interval):
	# 		self.dvl_last_update_time = msg.header.stamp.to_sec()

	# 		dvl_vel = np.array([msg.velocity.x,
	# 							msg.velocity.y,
	# 							msg.velocity.z])

	# 		if self.plot_results:
	# 			self.results['DVL'].append(
	# 				np.concatenate((np.array([msg.header.stamp.to_sec()]), dvl_vel), axis=0))
	# 		if not self.use_dvl:
	# 			return
	# 		if self.use_dvl:
	# 			self.__fusion_core.AddDvlMeasurement(self.dvl_last_update_time, dvl_vel)

	def DvlCallback(self, msg):
		if (not self.dvl_last_update_time or
			msg.header.stamp.to_sec() - self.dvl_last_update_time > self.dvl_interval):
			self.dvl_last_update_time = msg.header.stamp.to_sec()

			#############################################################################
			dvl_vel = np.array([msg.velocity.x,
								msg.velocity.y,
								msg.velocity.z])
			# Setup dvl offset array
			dvl_offset = array([self.sen_dvl_offsetX,
								self.sen_dvl_offsetY,
								self.sen_dvl_offsetZ])

			# Correct DVL coordinate frame wrt to ENU
			dvl_enuTransRbtLinVel = np.matmul(self.frameTrans, dvl_vel.T)
			dvl_enuTransRbtLinVel -= np.cross(self.ang_vel.T, dvl_offset).T
			# Convert velocity from robot frame into map frame
			self.sen_dvl_mapLinVel = np.matmul(self.imu_mapAngPos, dvl_enuTransRbtLinVel).T
			#############################################################################

			if self.plot_results:
				self.results['DVL'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.sen_dvl_mapLinVel), axis=0))
			if not self.use_dvl:
				return
			if self.use_dvl:
				self.__fusion_core.AddDvlMeasurement(self.dvl_last_update_time, self.sen_dvl_mapLinVel)

	def GtCallback(self, msg):
		"""Handle GT message"""
		gt_pos = np.array([msg.pose.pose.position.x,
						msg.pose.pose.position.y,
						msg.pose.pose.position.z])
		gt_ori = np.array([msg.pose.pose.orientation.x,
						msg.pose.pose.orientation.y,
						msg.pose.pose.orientation.z,
						msg.pose.pose.orientation.w])
		gt_vel = np.array([msg.twist.twist.linear.x,
						msg.twist.twist.linear.y,
						msg.twist.twist.linear.z])

		# Change incoming ground truth quarternion data into euler [rad]
		gt_quaternion = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
		gt_euler = euler_from_quaternion(gt_quaternion)
		unwrapEuler = np.unwrap(gt_euler)
		# Convert to numpy
		self.gt_pose = np.array([msg.pose.pose.position.x,
							msg.pose.pose.position.y,
							msg.pose.pose.position.z,
							np.rad2deg(unwrapEuler[0]),
							np.rad2deg(unwrapEuler[1]),
							np.rad2deg(unwrapEuler[2]),
							msg.twist.twist.linear.x,
							msg.twist.twist.linear.y,
							msg.twist.twist.linear.z])

		if self.plot_results:
			euler_gt_ori = np.asarray(euler_from_quaternion(gt_ori)) / np.pi * 180.
			self.results['GT'].append(
				np.concatenate((np.array([msg.header.stamp.to_sec()]), self.gt_pose), axis=0))
		if not self.use_gt:
			return
			
		# data for error analysis
		if self.compute_error:
			# grab estimator pose
			self.error_results['fgo'].append(
				np.concatenate((np.array([msg.header.stamp.to_sec()]), self.fgo_pose), axis=0))
			# grab estimator pose when ground truth is updated
			self.error_results['fgo_gt'].append(
				np.concatenate((np.array([msg.header.stamp.to_sec()]), self.gt_pose), axis=0))

	def DrCallback(self, msg):
		"""Handle DR message """
		if (not self.dr_last_update_time or
			msg.header.stamp.to_sec() - self.dr_last_update_time > self.dr_interval):
			self.dr_last_update_time = msg.header.stamp.to_sec()

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
				self.results['DR'].append(
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
		""" EKF message """
		if (not self.ekf_last_update_time or
			msg.header.stamp.to_sec() - self.ekf_last_update_time > self.ekf_interval):
			self.ekf_last_update_time = msg.header.stamp.to_sec()

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
				self.results['EKF'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.ekf_pose), axis=0))

			# data for error analysis
			if self.compute_error:
				# grab estimator pose
				self.error_results['ekfEst'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.ekf_pose), axis=0))
				# grab estimator pose when ground truth is updated
				self.error_results['ekf_gt'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.gt_pose), axis=0))


	def __publish_pose(self, stamp, frame_id, position, orientation):
		"""Publish PoseWithCovarianceStamped"""
		msg = PoseWithCovarianceStamped()
		msg.header.stamp = stamp
		msg.header.frame_id = frame_id
		msg.pose.pose.position.x = position[0]
		msg.pose.pose.position.y = position[1]
		msg.pose.pose.position.z = position[2]
		msg.pose.pose.orientation.x = orientation[0]
		msg.pose.pose.orientation.y = orientation[1]
		msg.pose.pose.orientation.z = orientation[2]
		msg.pose.pose.orientation.w = orientation[3]
		self.__base_pose_in_map_100hz_pub.publish(msg)

def main():
	"""Main"""
	node = GtsamFusionRos()
	if node.is_initialized:
		node.run()
	rospy.loginfo("Exiting..")

if __name__ == '__main__':
	main()