#! /usr/bin/env python
# """
# ROS Wrapper
# """

""" Import libraries """
import sys
import numpy as np
from numpy import array, zeros, reshape, matmul, eye, sqrt, cos, sin
from transformations import quaternion_from_euler, euler_from_quaternion

from deadReckon import DeadReckon
from commons import Rot
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from uuv_sensor_ros_plugins_msgs.msg import DVL
from auv_estimator.msg import State, Covariance, Inputs, Estimator

class DeadReckonRosNode:
	def __init__(self):
		""" ROS Parameters """
		# Grab topics 
		sub_dvl = rospy.get_param("~dvlTopic") 
		sub_imu = rospy.get_param("~imuTopic")
		# Decide which frame to use (ENU or NED)
		self.useEnu = rospy.get_param("~useEnu") # set to 1 to use ENU frame, set to 0 to NED
		# DVL parameters
		self.sen_dvl_offsetX = rospy.get_param("~dvl_offsetX") # offset relative from the sensor to vehicle center of mass in x-direction
		self.sen_dvl_offsetY = rospy.get_param("~dvl_offsetY") # offset relative from the sensor to vehicle center of mass in y-direction
		self.sen_dvl_offsetZ = rospy.get_param("~dvl_offsetZ") # offset relative from the sensor to vehicle center of mass in z-direction

		""" DR driver wrapper """
		self.uuvDr = DeadReckon(self.useEnu)
		self.dvl_update = 0

		""" Setup publishers/subscribers """
		# Subscribers
		self.sub_dvl = rospy.Subscriber(sub_dvl, DVL, self.DvlCallback) 
		self.sub_imu = rospy.Subscriber(sub_imu, Imu, self.ImuCallback) 
		# Publishers
		self.pub_dr_pose = rospy.Publisher('/dr/pose', Estimator, queue_size=1000) 

		""" Instantiate DR variables """
		self.imu_quat = zeros(shape=(4,1))
		self.mapLinPos = zeros(shape=(3,1))
		self.mapLinVel = zeros(shape=(3,1))
		self.mapAngVel = zeros(shape=(3,1))
		self.startTime = rospy.Time.now() # Start 1st ROS timer

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

		# dr_msg = Odometry()
		# dr_msg.header.frame_id = "world"
		# dr_msg.header.stamp = rospy.Time.now()
		# dr_msg.child_frame_id = '/dr/link'
		# dr_msg.pose.pose.position.x = self.mapLinPos[0]
		# dr_msg.pose.pose.position.y = self.mapLinPos[1]
		# dr_msg.pose.pose.position.z = self.mapLinPos[2]
		# dr_msg.twist.twist.linear.x = self.mapLinVel[0]
		# dr_msg.twist.twist.linear.y = self.mapLinVel[1]
		# dr_msg.twist.twist.linear.z = self.mapLinVel[2]
		# dr_msg.twist.twist.angular.x = self.mapAngVel[0]
		# dr_msg.twist.twist.angular.y = self.mapAngVel[1]
		# dr_msg.twist.twist.angular.z = self.mapAngVel[2]
		# dr_msg.pose.pose.orientation.x = self.imu_quat[0]
		# dr_msg.pose.pose.orientation.y = self.imu_quat[1]
		# dr_msg.pose.pose.orientation.z = self.imu_quat[2]
		# dr_msg.pose.pose.orientation.w = self.imu_quat[3]
		# self.pub_dr_pose.publish(dr_msg)

def main(args):
	rospy.init_node('DeadReckon', anonymous=True)
	rospy.loginfo("Starting deadReckonRosNode.py")
	dr = DeadReckonRosNode()

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
		cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)