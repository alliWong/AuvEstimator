#!/usr/bin/env python 
"""
This file computes the x-error, y-error, z-error, and distance between 
the estimator and dead reckoning against ground truth.
"""
import sys
import numpy as np
from numpy import array, zeros
from transformations import quaternion_from_euler, euler_from_quaternion
from sklearn.metrics import mean_squared_error

import rospy
import message_filters
from nav_msgs.msg import Odometry
from commons import EuclideanDistance, SkewSymmetric, Rot, TrapIntegrate, MapAngVelTrans, PressureToDepth, Rmse
from auv_estimator.msg import State, Covariance, Inputs, Estimator, Error
from computeError import ErrorAnalysis

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
		# Publishers
		self.pub_est_err = rospy.Publisher('/est/error', Error, queue_size=2) # publish gt vs est error
		self.pub_dr_err = rospy.Publisher('/dr/error', Error, queue_size=2) # publish gt vs est error
		# Subscribers
		self.sub_gt_pose = message_filters.Subscriber(sub_pose_gt, Odometry) # subscribe to ground truth pose
		self.sub_dr_pose = message_filters.Subscriber(sub_pose_dr, Odometry) # subscribe to estimator pose
		self.sub_est_pose = message_filters.Subscriber(sub_pose_est, Estimator) # subscribe to estimator pose
		# Approximately synchronizes messages from subscribers by their timestamps
		# self.dr_time_synch = message_filters.ApproximateTimeSynchronizer([self.sub_gt_pose, self.sub_dr_pose], 
		# 					queue_size=1000, slop=0.1, allow_headerless=True)
		# self.dr_time_synch.registerCallback(self.dr_err_callback) # register multiple callbacks

		self.est_time_synch = message_filters.ApproximateTimeSynchronizer([self.sub_gt_pose, self.sub_est_pose], 
							queue_size=1000, slop=0.1, allow_headerless=True)
		self.est_time_synch.registerCallback(self.est_err_callback) # register multiple callbacks

	def dr_err_callback(self, gt_msg, dr_msg):
		# Change incoming ground truth quarternion data into euler [rad]
		gt_quaternion = (gt_msg.pose.pose.orientation.x, gt_msg.pose.pose.orientation.y, gt_msg.pose.pose.orientation.z, gt_msg.pose.pose.orientation.w)
		gt_euler = np.unwrap(euler_from_quaternion(gt_quaternion))

		# Change incoming dead reckoning quarternion data into euler [rad]
		dr_quaternion = (dr_msg.pose.pose.orientation.x, dr_msg.pose.pose.orientation.y, dr_msg.pose.pose.orientation.z, dr_msg.pose.pose.orientation.w)
		dr_euler = euler_from_quaternion(dr_quaternion)

		# Instantiate arrays and variables
		self.gt_pose = array([[gt_msg.pose.pose.position.x],
							[gt_msg.pose.pose.position.y],
							[gt_msg.pose.pose.position.z],
							[np.rad2deg(gt_euler[0])],
							[np.rad2deg(gt_euler[1])],
							[np.rad2deg(gt_euler[2])],
							[gt_msg.twist.twist.linear.x],
							[gt_msg.twist.twist.linear.y],
							[gt_msg.twist.twist.linear.z]]) # ground truth array
		self.dr_pose = array([[dr_msg.pose.pose.position.x],
							[dr_msg.pose.pose.position.y],
							[dr_msg.pose.pose.position.z],
							[np.rad2deg(dr_euler[0])],
							[np.rad2deg(dr_euler[1])],
							[np.rad2deg(dr_euler[2])],
							[dr_msg.twist.twist.linear.x],
							[dr_msg.twist.twist.linear.y],
							[dr_msg.twist.twist.linear.z]]) # dead reckoning array

		self.dr_dist_traveled, self.dr_dist_error_rmse = Err.DrAnalysis(self.gt_pose, self.dr_pose)

		# Publish 
		self.PublishDr()

	def est_err_callback(self, gt_msg, est_msg):
		# Change incoming ground truth quarternion data into euler [rad]
		gt_quaternion = (gt_msg.pose.pose.orientation.x, gt_msg.pose.pose.orientation.y, gt_msg.pose.pose.orientation.z, gt_msg.pose.pose.orientation.w)
		gt_euler = np.unwrap(euler_from_quaternion(gt_quaternion))

		# Instantiate arrays and variables
		self.gt_pose = array([[gt_msg.pose.pose.position.x],
							[gt_msg.pose.pose.position.y],
							[gt_msg.pose.pose.position.z],
							[np.rad2deg(gt_euler[0])],
							[np.rad2deg(gt_euler[1])],
							[np.rad2deg(gt_euler[2])],
							[gt_msg.twist.twist.linear.x],
							[gt_msg.twist.twist.linear.y],
							[gt_msg.twist.twist.linear.z]]) # ground truth array
		self.est_pose = array([[est_msg.state.x],
								[est_msg.state.y],
								[est_msg.state.z],
								[est_msg.state.roll],
								[est_msg.state.pitch],
								[est_msg.state.yaw],
								[est_msg.state.vx],
								[est_msg.state.vy],
								[est_msg.state.vz]]) # estimator array

		self.est_dist_traveled, self.est_dist_error_rmse = Err.EstAnalysis(self.gt_pose, self.est_pose)

		# Publish 
		self.PublishEst()

	def PublishDr(self):
		# DR error
		dr_err_msg = Error()
		dr_err_msg.header.stamp = rospy.Time.now()
		dr_err_msg.rmse_dist = self.dr_dist_error_rmse
		dr_err_msg.dist_traveled = self.dr_dist_traveled
		self.pub_dr_err.publish(dr_err_msg)

	def PublishEst(self):
		# Est error
		est_err_msg = Error()
		est_err_msg.header.stamp = rospy.Time.now()
		est_err_msg.rmse_dist = self.est_dist_error_rmse
		est_err_msg.dist_traveled = self.est_dist_traveled
		self.pub_est_err.publish(est_err_msg)

def main(args):
    rospy.init_node('ComputeError', anonymous=True)
    rospy.loginfo("Starting computeErrorRosNode.py")
    err = ErrorAnalysisRosNode()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)