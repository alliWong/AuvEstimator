#! /usr/bin/env python
"""
Converts pressure from pressure_sensor into the depth of the AUV.

Input:	/rexrov2/pressure

Output:	/est/bar/map/z
"""

import sys
import rospy
from sensor_msgs.msg import FluidPressure
from nav_msgs.msg import Odometry
from commons import PressureToDepth

class Pressure:
    def __init__(self):
        # Grab ROS parameters
        sub_bar = rospy.get_param("~barTopic") # grab pressure topic
        self.sen_bar_offsetZ = rospy.get_param("~barOffset") # grab barometer z offset

        # Subscribe to pressure topic
        self.sub_bar = rospy.Subscriber(sub_bar, FluidPressure, self.bar_callback)

        # Create publisher topic to store depth data
        self.pub_bar_map_z = rospy.Publisher('/bar/depth', Odometry, queue_size=2)  

    def bar_callback(self, msg):
        # Compute depth
        self.sen_bar_mapLinPos = PressureToDepth(msg.fluid_pressure, self.sen_bar_offsetZ)

        # ROS Interface
        bar_msg = Odometry()
        bar_msg.header.stamp = rospy.Time.now()
        bar_msg.header.frame_id = "/map/bar_link"
        bar_msg.pose.pose.position.z = self.sen_bar_mapLinPos
        self.pub_bar_map_z.publish(bar_msg)

def main(args):
	rospy.init_node('Barometer', anonymous=True)
	rospy.loginfo("Starting barometer.py")
	press = Pressure()

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
