#! /usr/bin/env python

# ""
# ROS Wrapper: Dead reckoning state estimation
# """

import sys
import numpy as np
from numpy import array, zeros, reshape, matmul, eye, sqrt, cos, sin
from transformations import quaternion_from_euler, euler_from_quaternion

from deadReckon import DeadReckon
from commons import Rot
import rospy
import rosbag
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from uuv_sensor_ros_plugins_msgs.msg import DVL
from auv_estimator.msg import State, Covariance, Inputs, Estimator

class DeadReckonRos:
	def __init__(self):
		rospy.init_node('DeadReckon', anonymous=True)
		""" ROS Parameters """
		# Grab topics 
		self.imu_topic = rospy.get_param('~imu_topic', "/rexrov2/imu") # IMU messages 
		self.dvl_topic = rospy.get_param('~dvl_topic', "/rexrov2/dvl") # dvl messages 
		# Decide which frame to use (ENU or NED)
		self.useEnu = rospy.get_param("~useEnu") # set to 1 to use ENU frame, set to 0 to NED
		# DVL parameters
		self.sen_dvl_offsetX = rospy.get_param("~dvl_offsetX") # offset relative from the sensor to vehicle center of mass in x-direction
		self.sen_dvl_offsetY = rospy.get_param("~dvl_offsetY") # offset relative from the sensor to vehicle center of mass in y-direction
		self.sen_dvl_offsetZ = rospy.get_param("~dvl_offsetZ") # offset relative from the sensor to vehicle center of mass in z-direction
		# Plot information
		self.plot_results = rospy.get_param('~plot_results', False) # plot results after the bag file is processed
		self.save_dir = rospy.get_param('~save_dir', '/tmp') # directory where the result images are saved
		# Error analysis
		self.compute_error = rospy.get_param('~compute_error', False) # compute error after the bag file is processed
		# variables
		self.is_initialized = False

		""" Setup publishers """
		self.pub_dr_pose = rospy.Publisher('/dr/pose', Estimator, queue_size=1000) 

		""" Instantiate DR variables """
		self.imu_quat = zeros(shape=(4,1))
		self.mapLinPos = zeros(shape=(3,1))
		self.mapLinVel = zeros(shape=(3,1))
		self.mapAngVel = zeros(shape=(3,1))
		self.startTime = rospy.Time.now() # Start 1st ROS timer

		""" Start dead reckoning """
		self.uuvDr = DeadReckon(self.useEnu)
		self.dvl_update = 0
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
			for topic, msg, t in bag.read_messages(topics=[self.imu_topic, self.dvl_topic]):
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
				elif topic == self.dvl_topic:
					self.DvlCallback(msg)
			bag.close()
			rospy.loginfo("Bag processed.")
			# if self.compute_error:
			# 	rospy.loginfo("Computing error. Please wait..")
			# 	errorAnalysis.compute_results_error(self.error_results, self.use_fgo, self.use_gt, self.use_dr, self.use_ekf, self.use_bar)
			if self.plot_results:
				rospy.loginfo("Preparing plots. Please wait..")
				if not os.path.exists(self.save_dir):
					os.makedirs(self.save_dir)
				plots.plot_all(self.results, self.fusion_items, self.save_dir, self.use_gt, self.use_dr, self.use_ekf, self.use_bar, self.use_dvl)
		else: # subscribe to topics
			# Subscribers
			self.sub_dvl = rospy.Subscriber(self.dvl_topic, DVL, self.DvlCallback) 
			self.sub_imu = rospy.Subscriber(self.imu_topic, Imu, self.ImuCallback) 
			rospy.spin()

	""" Raw sensor measurements """
	def DvlCallback(self, msg):
		self.dvl_update = 1
		# Instantiate dvl variables
		currTime = rospy.Time.now() # Start 2nd ROS clock
		dt = (currTime - self.startTime).to_sec() # time step [s]

		# Setup robot velocity array
		dvl_rbtLinVel = array([[msg.velocity.x],
							[msg.velocity.y],
							[msg.velocity.z]])
		# Setup dvl offset array
		dvl_offset = array([[self.sen_dvl_offsetX], 
							[self.sen_dvl_offsetY], 
							[self.sen_dvl_offsetZ]])

		# Initialize DR 
		self.uuvDr.DvlCallback(dvl_rbtLinVel, dvl_offset, dt)
		self.DeadReckoning()

		# Update time
		self.startTime = currTime
		self.dvl_update = 0

	def ImuCallback(self, msg):
		# Setup robot angular velocity array
		imu_rbtAngVel = np.array([[msg.angular_velocity.x],
								[msg.angular_velocity.y],
								[msg.angular_velocity.z]]) 
		# Setup map orientation array
		self.imu_quat = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
		euler = euler_from_quaternion(self.imu_quat) # Convert IMU data from quarternion to euler
		unwrapEuler = np.unwrap(euler) # unwrap euler angles
		self.imu_mapEulAng = np.array([[unwrapEuler[0]],
								[unwrapEuler[1]],
								[unwrapEuler[2]]]) # imu angular position array  
		### NOTES: Incoming IMU data is in rotation matrix form  ###
		imu_mapAngPos = Rot(self.imu_mapEulAng[0], self.imu_mapEulAng[1], self.imu_mapEulAng[2])

		# Initialize DR 
		self.uuvDr.ImuCallback(imu_rbtAngVel, imu_mapAngPos, self.imu_mapEulAng)
		self.DeadReckoning()

	""" Run DeadReckoning """ 
	def DeadReckoning(self):
		if self.dvl_update == 1:
			# Get estimator state
			self.mapLinPos, self.mapLinVel, self.mapAngVel = self.uuvDr.OutputDr()

			# Publish ROS messages 
			self.PubDr() # publish estimator messages

	""" Publish """
	def PubDr(self):
		# Publish dead reckoning state message
		dr_msg = Estimator()
		dr_msg.header.stamp = rospy.Time.now()
		dr_msg.header.frame_id = "world"
		dr_msg.state.x = self.mapLinPos[0] # x
		dr_msg.state.y = self.mapLinPos[1] # y
		dr_msg.state.z = self.mapLinPos[2] # z
		dr_msg.state.roll = np.rad2deg(self.imu_mapEulAng[0]) # roll
		dr_msg.state.pitch = np.rad2deg(self.imu_mapEulAng[1]) # pitch
		dr_msg.state.yaw = np.rad2deg(self.imu_mapEulAng[2]) # yaw
		dr_msg.state.vx = self.mapLinVel[0] # dx
		dr_msg.state.vy = self.mapLinVel[1] # dy
		dr_msg.state.vz = self.mapLinVel[2] # dz
		self.pub_dr_pose.publish(dr_msg) # publish estimator state message

def main():
	"""Main"""
	node = DeadReckonRos()
	if node.is_initialized:
		node.run()
	rospy.loginfo("Exiting..")

if __name__ == '__main__':
	main()