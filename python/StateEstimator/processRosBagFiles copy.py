#!/usr/bin/python
"""
ROS interface to bag files
"""
import sys
import os
import rospy
import rosbag
import numpy as np
import plots
import errorAnalysis
from transformations import quaternion_from_euler, euler_from_quaternion
from auv_estimator.msg import State, Covariance, Inputs, Estimator


class ProcessRosBag():
	""" ROS interface to bag files """
	def __init__(self):
		""" Node """
		rospy.init_node('processBagFile', anonymous=True)
		""" ROS parameters """
		# Grab topics
		self.gt_topic = rospy.get_param('~gt_topic', "/rexrov2/pose_gt") # ground truth messages
		self.dr_topic = rospy.get_param('~dr_topic', "/dr/pose") # dead reckoning messages
		self.ekf_topic = rospy.get_param('~ekf_topic', "/est/state") # EKF estimator messages
		self.depth_topic = rospy.get_param('~depth_topic', "/bar/depth") # depth messages
		self.imu_topic = rospy.get_param('~imu_topic', "/rexrov2/imu") # imu messages
		# Grab bag file information
		self.bag_file_path = rospy.get_param('~bag_file_path', "") # if this path is set, data is read from the bag file via rosbag API as fast as possible
		self.gt_topic = rospy.get_param('~gt_topic', 0) # set zero to use all available ground truth messages
		self.dr_topic = rospy.get_param('~dr_topic', 0) # set zero to use all available dead reckoning messages
		self.ekf_topic = rospy.get_param('~ekf_topic', 0) # set zero to use all available EKF estimator messages
		self.depth_topic = rospy.get_param('~depth_topic', 0) # set zero to use all available depth messages
		self.imu_topic = rospy.get_param('~imu_topic', 0) # set zero to use all available imu messages
		self.bag_secs_to_skip = rospy.get_param('~bag_secs_to_skip', 0.0) # skip data from the start when reading from bag
		self.fixed_dt = rospy.get_param('~fixed_dt', None) # use this to set fixed dt for IMU samples (instead dt calculated from the time stamps)
		self.use_gt = rospy.get_param('~use_gt', False) # enable / disable the use of ground truth messages
		self.use_dr = rospy.get_param('~use_dr', False) # enable / disable the use of dead reckoning messages
		self.use_ekf = rospy.get_param('~use_ekf', False) # enable / disable the use of EKF estimator messages
		self.use_depth = rospy.get_param('~use_depth', False) # enable / disable the use of depth messages
		self.use_imu = rospy.get_param('~use_imu', False) # enable / disable the use of imu messages
		self.dr_interval = rospy.get_param('~dr_interval', 0) # set zero to use all available dead reckoning messages
		self.ekf_interval = rospy.get_param('~ekf_interval', 0) # set zero to use all available EKF messages
		self.depth_interval = rospy.get_param('~depth_interval', 0) # set zero to use all available depth messages
		self.imu_interval = rospy.get_param('~imu_interval', 0) # set zero to use all available IMU messages

		# Plot information
		self.plot_results = rospy.get_param('~plot_results', False) # plot results after the bag file is processed
		self.save_dir = rospy.get_param('~save_dir', '/tmp') # directory where the result images are saved

		# Error analysis
		self.compute_error = rospy.get_param('~compute_error', False) # compute error after the bag file is processed


		# Variables
		self.is_initialized = False
		self.gt_last_update_time = None
		self.dr_last_update_time = None
		self.ekf_last_update_time = None
		self.depth_last_update_time = None
		self.imu_last_update_time = None
		self.gt_pose = None
		self.ekf_pose = None
		self.dr_pose = None
		self.depth = None

		if self.plot_results:
			self.results = {'ground truth': [], 'dead reckoning': [], 'EKF': [], 'measurements (barometer)': [], 'measurements (IMU)': []}

		if self.compute_error:
			self.error_results = {'dr': [], 'dr_gt': [], 'dr_bar': [], 'ekfEst': [], 'ekf_gt': [], 'ekf_bar': []}

		# Initialize the processing of bag files
		self.is_initialized = True

	def RunBagFiles(self):
		"""Either process bag file via rosbag API"""
		if self.bag_file_path: # use rosbag API
			rospy.loginfo("Processing file using rosbag API: {}. Please wait..".format(self.bag_file_path))
			if self.bag_secs_to_skip > 0:
				rospy.loginfo("Skipping {} seconds from the start of the bag file.".format(self.bag_secs_to_skip))
			bag = rosbag.Bag(self.bag_file_path)
			total_time_secs = int(bag.get_end_time() - bag.get_start_time())
			init_t = None
			last_info_time_secs = int(bag.get_start_time())
			for topic, msg, t in bag.read_messages(topics=[self.gt_topic, self.dr_topic, self.ekf_topic, self.depth_topic, self.imu_topic]):
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
				if topic == self.gt_topic:
					self.GtCallback(msg)
				elif topic == self.dr_topic:
					self.DrCallback(msg)
				elif topic == self.ekf_topic:
					self.EkfCallback(msg)
				elif topic == self.depth_topic:
					self.DepthCallback(msg)
				elif topic == self.imu_topic:
					self.ImuCallback(msg)
			bag.close()
			rospy.loginfo("Bag processed.")
			if self.plot_results:
				rospy.loginfo("Preparing plots. Please wait..")
				if not os.path.exists(self.save_dir):
					os.makedirs(self.save_dir)
				plots.plot_results(self.results, self.save_dir, self.use_gt, self.use_dr, self.use_ekf, self.use_depth, self.use_imu)
			if self.compute_error:
				rospy.loginfo("Computing error. Please wait..")
				errorAnalysis.compute_results_error(self.error_results, self.use_gt, self.use_dr, self.use_ekf, self.use_depth, self.use_imu)
			rospy.spin()

	def GtCallback(self, msg):
		""" Handle GT message """
		if self.gt_last_update_time:
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
			dt = msg.header.stamp.to_sec() - self.gt_last_update_time

			# data for plots
			if self.plot_results:
				self.results['ground truth'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.gt_pose), axis=0))

			# # data for error analysis
			# if self.compute_error:
			# 	# grab estimator pose
			# 	self.error_results['ekfEst'].append(
			# 		np.concatenate((np.array([msg.header.stamp.to_sec()]), self.ekf_pose), axis=0))
			# 	# grab estimator pose when ground truth is updated
			# 	self.error_results['ekf_gt'].append(
			# 		np.concatenate((np.array([msg.header.stamp.to_sec()]), self.gt_pose), axis=0))

		# update time
		self.gt_last_update_time = msg.header.stamp.to_sec()

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
				self.results['dead reckoning'].append(
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

	def DepthCallback(self, msg):
		""" Handle barometer position Z message """
		if (not self.depth_last_update_time or
			msg.header.stamp.to_sec() - self.depth_last_update_time > self.depth_interval):
			self.depth_last_update_time = msg.header.stamp.to_sec()

			self.depth = np.array([msg.pose.pose.position.z])

			# data for plots
			if self.plot_results:
				self.results['measurements (barometer)'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), self.depth), axis=0))

	def ImuCallback(self, msg):
		""" Handle IMU message """
		if (not self.imu_last_update_time or
			msg.header.stamp.to_sec() - self.imu_last_update_time > self.imu_interval):
			self.imu_last_update_time = msg.header.stamp.to_sec()

			# Setup map orientation array
			euler = euler_from_quaternion([msg.orientation.x,
										msg.orientation.y,
										msg.orientation.z,
										msg.orientation.w]) # Convert IMU data from quarternion to euler
			unwrapEuler = np.unwrap(euler) # unwrap euler angles
			imu_rot = np.array([np.rad2deg(unwrapEuler[0]),
									np.rad2deg(unwrapEuler[1]),
									np.rad2deg(unwrapEuler[2])]) # imu angular position array

			# data for plots
			if self.plot_results:
				self.results['measurements (IMU)'].append(
					np.concatenate((np.array([msg.header.stamp.to_sec()]), imu_rot), axis=0))

def main():
	"""Main"""
	node = ProcessRosBag()
	if node.is_initialized:
		node.RunBagFiles()
	rospy.loginfo("Exiting..")

if __name__ == '__main__':
	main()
