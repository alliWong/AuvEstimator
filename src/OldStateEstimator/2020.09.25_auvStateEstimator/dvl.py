#!/usr/bin/env python

"""
Converts datas from DVL and IMU into the position of the robot.

Input:	/desisek_saga/dvl 
		/desistek_saga/imu

Output:	/desistek_saga/dvl/position

In __init__, set OFFSET_X, OFFSET_Y and OFFSET_Z equal to the distance in xyz between the DVL and the inertial center of the robot.
		   , set STARTING_X, Y, Z equal to the xyz starting position of the robot, and STARTING_radianX, Y, Z equal to its orientation in radians
"""

import rospy
from uuv_sensor_ros_plugins_msgs.msg import DVL
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

class dvl:
	def __init__(self):
		# 
		sub_dvl = rospy.Subscriber('/desistek_saga/dvl', DVL, self.dvl_sub)
		sub_imu = rospy.Subscriber('/desistek_saga/imu', Imu, self.imu_sub)
		self.pubOdom = rospy.Publisher("/desistek_saga/dvl/position", Odometry, queue_size=1)

		########################## TO BE SET ACCORDING TO THE ROBOT CONFIGURATION #########################
		# The OFFSET corresponds to the position of the dvl relatively to the center of inertia of the robot
		self.OFFSET_X = 0
		self.OFFSET_Y = 0
		self.OFFSET_Z = 0
		# The STARTING parameter corresponds to the starting position of the AUV in the world.
		self.STARTING_X = 0
		self.STARTING_Y = 0
		self.STARTING_Z = -80
		self.STARTING_radianX = 0
		self.STARTING_radianY = 0
		self.STARTING_radianZ = 0
		###################################################################################################

		self.timeDVL = rospy.get_time()
		self.previous_time = 0
		self.dvlseq = 0
		self.dvlsecs = 0
		self.dvlnsecs = 0
		self.dvlX = 0
		self.dvlY = 0
		self.dvlZ = 0
		
		self.timeIMU = rospy.get_time()
		self.quaternionX = 0
		self.quaternionY = 0
		self.quaternionZ = 0
		self.quaternionW = 0
		self.imuX = 0
		self.imuY = 0
		self.imuZ = 0
		self.lastImuX = 0
		self.lastImuY = 0
		self.lastImuZ = 0

		self.dvlReceived = False

		self.estimated_traj_x = self.STARTING_X
		self.estimated_traj_y = self.STARTING_Y
		self.estimated_traj_z = self.STARTING_Z

	# Read the DVL topic
	def dvl_sub(self,msg):
		self.timeDVL = rospy.get_time()
		self.dvlseq = msg.header.seq
		self.dvlsecs = msg.header.stamp.secs
		self.dvlnsecs = msg.header.stamp.nsecs
		self.dvlX = msg.velocity.x
		self.dvlY = msg.velocity.y
		self.dvlZ = msg.velocity.z
		self.dvlReceived = True

	# Read the IMU topic
	def imu_sub(self,msg):
		# Read only if the dvl data has been received
		# This is due to manage the difference of frequency of the 2 sensors.
		# freq(IMU) > freq(DVL)
		if self.dvlReceived == True:
			self.dvlReceived = False
			self.timeIMU = rospy.get_time()
			self.quaternionX = msg.orientation.x
			self.quaternionY = msg.orientation.y
			self.quaternionZ = msg.orientation.z
			self.quaternionW = msg.orientation.w

			# Converts the quaternion to euler
			X,Y,Z = self.quaternion_to_euler(self.quaternionX,self.quaternionY,self.quaternionZ,self.quaternionW)
			self.imuX = X
			self.imuY = Y
			self.imuZ = Z

			# Now that the dvl and the imu have sent data, estimate the position
			self.estimateTraj()


	# Estimates the position of the robot
	def estimateTraj(self):
		dt = float(self.timeDVL - self.previous_time)
		
		X = self.dvlX - self.OFFSET_X*(self.imuZ-self.lastImuZ + self.imuY-self.lastImuY)/dt
		Y = self.dvlY - self.OFFSET_Y*(self.imuZ-self.lastImuZ + self.imuX-self.lastImuX)/dt
		Z = self.dvlZ - self.OFFSET_Z*(self.imuX-self.lastImuX + self.imuY-self.lastImuY)/dt

		self.estimated_traj_x += dt*(X*math.cos(self.imuZ) + Y*math.sin(self.imuZ))
		self.estimated_traj_y -= dt*(-X*math.sin(self.imuZ) + Y*math.cos(self.imuZ))
		self.estimated_traj_z += Z*dt
		
		self.previous_time = self.timeDVL
		self.lastImuX = self.imuX + self.STARTING_radianX
		self.lastImuY = self.imuY + self.STARTING_radianY
		self.lastImuZ = self.imuZ + self.STARTING_radianZ

		self.convert_to_odom()

	# Publish the estimated position
	def convert_to_odom(self):
		odm = Odometry()
		odm.header.seq = self.dvlseq
		rostime = rospy.get_time()
		odm.header.stamp.secs = int(rostime)
		odm.header.stamp.nsecs = 1000000000*(rostime-int(rostime))

		odm.header.frame_id = "world"

		odm.pose.pose.position.x = self.estimated_traj_x
		odm.pose.pose.position.y = self.estimated_traj_y
		odm.pose.pose.position.z = self.estimated_traj_z

		odm.pose.pose.orientation.x = self.quaternionX
		odm.pose.pose.orientation.y = self.quaternionY
		odm.pose.pose.orientation.z = self.quaternionZ
		odm.pose.pose.orientation.w = self.quaternionW
		self.pubOdom.publish(odm)

	# Converts the quaternion to euler
	def quaternion_to_euler(self,x,y,z,w):
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		X = math.degrees(math.atan2(t0, t1))

		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		Y = math.degrees(math.asin(t2))

		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		Z = math.atan2(t3, t4)

		return X, Y, Z

def main(args):
	rospy.init_node('read_dvl', anonymous=True)
	rospy.loginfo("Starting desistek_dvl.py")
	dvlog = dvl()

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
