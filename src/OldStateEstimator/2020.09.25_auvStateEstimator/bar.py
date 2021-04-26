#! /usr/bin/env python

"""
Converts depth data from pressure_sensor into the depth of the AUV.

Input:	/rexrov2/pressure

Output:	/rexrov2/depth
"""

import rospy
from sensor_msgs.msg import FluidPressure
from nav_msgs.msg import Odometry
import sys

class BarSensor:
	def __init__(self):
		self.standPressure = 101.325 # standard pressure (1 atm) [kPa]
		self.kPaPerM = 9.804139432 # pressure per meter [kPa/m]

        # ROS Interface
		self.sub_pressure = rospy.Subscriber('/rexrov2/pressure', FluidPressure, self.pressure_sub)
		self.pub = rospy.Publisher('/rexrov2/depth', Odometry, queue_size=1)

	def pressure_sub(self,msg):
		pressure = msg.fluid_pressure # ros message for pressure reading [Pa]
		depth = (pressure-self.standPressure)/self.kPaPerM # calculated depth from auv pressure data

		bar = Odometry()
		bar.pose.pose.position.z = -depth
		self.pub.publish(bar)

def main(args):
	rospy.init_node('bar_sensor',anonymous=True)
	rospy.loginfo("Starting bar.py")
	BarSensor()

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)


