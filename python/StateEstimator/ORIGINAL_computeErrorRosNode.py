#!/usr/bin/env python 
"""
This file computes the x-error, y-error, z-error, and distance between the estimator and dead reckoning against ground truth.
"""
import sys
import numpy as np
from numpy import array, zeros
from transformations import quaternion_from_euler, euler_from_quaternion

import rospy
import message_filters
from nav_msgs.msg import Odometry
from auv_estimator.msg import State, Covariance, Inputs, Estimator, Error
from originalComputeError import ErrorAnalysis
from sklearn.metrics import mean_squared_error

""" Compute error driver wrapper """
Err = ErrorAnalysis()

class ErrorAnalysisRosNode(object):
	def __init__(self):
		""" ROS Parameters """
		# Grab topics 
		sub_pose_gt = rospy.get_param("~groundTruthTopic") # grab ground truth ros parameters
		sub_pose_est = rospy.get_param("~estimatorTopic") # grab estimator ros parameters
		sub_pose_dr = rospy.get_param("~deadReckoningTopic") # grab estimator ros parameters
		
		""" Setup publishers/subscribers """
		# Subscribers
		self.sub_gt_pose = rospy.Subscriber(sub_pose_gt, Odometry, self.GtCallback) # subscribe to ground truth pose
		self.sub_dr_pose = rospy.Subscriber(sub_pose_dr, Odometry, self.DrCallback) # subscribe to estimator pose
		self.sub_est_pose = rospy.Subscriber(sub_pose_est, Estimator, self.EstCallback) # subscribe to estimator pose
		# Publishers
		self.pub_dr_err = rospy.Publisher('/dr/error2', Error, queue_size=1000) # publish gt vs est error
		self.pub_est_err = rospy.Publisher('/est/error2', Error, queue_size=1000) # publish gt vs est error

		""" Variables """
		self.dr_update = 0
		self.est_update = 0

	def GtCallback(self, msg):
		# Change incoming ground truth quarternion data into euler [rad]
		gt_quaternion = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
		gt_euler = euler_from_quaternion(gt_quaternion) # Convert gt data from quarternion to euler
		unwrapEuler = np.unwrap(gt_euler) # unwrap euler angles
		gt_mapEulAng = np.array([[unwrapEuler[0]],
								[unwrapEuler[1]],
								[unwrapEuler[2]]]) # gt angular position array  

		gt_pose = array([[msg.pose.pose.position.x],
						[msg.pose.pose.position.y],
						[msg.pose.pose.position.z],
						[np.rad2deg(gt_mapEulAng[0])],
						[np.rad2deg(gt_mapEulAng[1])],
						[np.rad2deg(gt_mapEulAng[2])],
						[msg.twist.twist.linear.x],
						[msg.twist.twist.linear.y],
						[msg.twist.twist.linear.z]]) # ground truth array

		# Initialize compute error
		Err.GtCallback(gt_pose)
		self.ErrAnalysis()

	def DrCallback(self, msg):
		self.dr_update = 1
		# Change incoming dead reckoning quarternion data into euler [rad]
		dr_quaternion = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
		dr_euler = euler_from_quaternion(dr_quaternion)
		unwrapEuler = np.unwrap(dr_euler) # unwrap euler angles
		dr_mapEulAng = np.array([[unwrapEuler[0]],
								[unwrapEuler[1]],
								[unwrapEuler[2]]]) # dr angular position array  

		dr_pose = array([[msg.pose.pose.position.x],
						[msg.pose.pose.position.y],
						[msg.pose.pose.position.z],
						[np.rad2deg(dr_mapEulAng[0])],
						[np.rad2deg(dr_mapEulAng[1])],
						[np.rad2deg(dr_mapEulAng[2])],
						[msg.twist.twist.linear.x],
						[msg.twist.twist.linear.y],
						[msg.twist.twist.linear.z]]) # dead reckoning array

		# Initialize compute error
		Err.DrCallback(dr_pose)
		self.ErrAnalysis()
		self.dr_update = 0

	def EstCallback(self, msg):
		self.est_update = 1
		est_pose = array([[msg.state.x],
						[msg.state.y],
						[msg.state.z],
						[msg.state.roll],
						[msg.state.pitch],
						[msg.state.yaw],
						[msg.state.vx],
						[msg.state.vy],
						[msg.state.vz]]) # estimator array

		# Initialize compute error
		Err.EstCallback(est_pose)
		self.ErrAnalysis()
		self.est_update = 0

	def ErrAnalysis(self):
		if self.dr_update == 1:
			self.dr_error_pose, self.dr_rmse, self.dr_dist_error, self.dr_dist_error_rmse = Err.OutputDr()
			# Publish data
			self.PublishDr()
		# if self.est_update == 1:
		# 	self.est_error_pose, self.est_rmse, self.est_dist_error, self.est_dist_error_rmse = Err.OutputEst()
		# 	# Publish data
		# 	self.PublishEst()

	def PublishDr(self):
		# DR error
		dr_err_msg = Error()
		dr_err_msg.header.stamp = rospy.Time.now()
		dr_err_msg.x = self.dr_error_pose[0]
		dr_err_msg.y = self.dr_error_pose[1]
		dr_err_msg.z = self.dr_error_pose[2]
		dr_err_msg.roll = self.dr_error_pose[3]
		dr_err_msg.pitch = self.dr_error_pose[4]
		dr_err_msg.yaw = self.dr_error_pose[5]
		dr_err_msg.vx = self.dr_error_pose[6]
		dr_err_msg.vy = self.dr_error_pose[7]
		dr_err_msg.vz = self.dr_error_pose[8]
		dr_err_msg.rmse_x = self.dr_rmse[0]
		dr_err_msg.rmse_y = self.dr_rmse[1]
		dr_err_msg.rmse_z = self.dr_rmse[2]
		dr_err_msg.rmse_roll = self.dr_rmse[3]
		dr_err_msg.rmse_pitch = self.dr_rmse[4]
		dr_err_msg.rmse_yaw = self.dr_rmse[5]
		dr_err_msg.rmse_vx = self.dr_rmse[6]
		dr_err_msg.rmse_vy = self.dr_rmse[7]
		dr_err_msg.rmse_vz = self.dr_rmse[8]
		dr_err_msg.rmse_dist = self.dr_dist_error_rmse
		dr_err_msg.dist_error = self.dr_dist_error
		# dr_err_msg.dist_traveled = self.dist_traveled
		self.pub_dr_err.publish(dr_err_msg)

	def PublishEst(self):
		# Est error
		est_err_msg = Error()
		est_err_msg.header.stamp = rospy.Time.now()
		est_err_msg.x = self.est_error_pose[0]
		est_err_msg.y = self.est_error_pose[1]
		est_err_msg.z = self.est_error_pose[2]
		est_err_msg.roll = self.est_error_pose[3]
		est_err_msg.pitch = self.est_error_pose[4]
		est_err_msg.yaw = self.est_error_pose[5]
		est_err_msg.vx = self.est_error_pose[6]
		est_err_msg.vy = self.est_error_pose[7]
		est_err_msg.vz = self.est_error_pose[8]
		est_err_msg.rmse_x = self.est_rmse[0]
		est_err_msg.rmse_y = self.est_rmse[1]
		est_err_msg.rmse_z = self.est_rmse[2]
		est_err_msg.rmse_roll = self.est_rmse[3]
		est_err_msg.rmse_pitch = self.est_rmse[4]
		est_err_msg.rmse_yaw = self.est_rmse[5]
		est_err_msg.rmse_vx = self.est_rmse[6]
		est_err_msg.rmse_vy = self.est_rmse[7]
		est_err_msg.rmse_vz = self.est_rmse[8]
		est_err_msg.rmse_dist = self.est_dist_error_rmse
		est_err_msg.dist_error = self.est_dist_error
		# est_err_msg.dist_traveled = self.dist_traveled
		self.pub_est_err.publish(est_err_msg)

def main(args):
	rospy.init_node('ComputeError2', anonymous=True)
	rospy.loginfo("Starting ORIGINAL_computeErrorRosNode.py")
	err = ErrorAnalysisRosNode()

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
		cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)