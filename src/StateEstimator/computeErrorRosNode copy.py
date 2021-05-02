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

class ErrorAnalysis(object):
	def __init__(self):
		sub_pose_gt = rospy.get_param("~groundTruthTopic") # grab ground truth ros parameters
		sub_pose_est = rospy.get_param("~estimatorTopic") # grab estimator ros parameters
		sub_pose_dr = rospy.get_param("~deadReckoningTopic") # grab estimator ros parameters
		self.sub_gt_pose = message_filters.Subscriber(sub_pose_gt, Odometry) # subscribe to ground truth pose
		self.sub_est_pose = message_filters.Subscriber(sub_pose_est, Estimator) # subscribe to estimator pose
		self.sub_dr_pose = message_filters.Subscriber(sub_pose_dr, Odometry) # subscribe to estimator pose
		self.pub_est_err = rospy.Publisher('/est/error', Error, queue_size=2) # publish gt vs est error
		self.pub_dr_err = rospy.Publisher('/dr/error', Error, queue_size=2) # publish gt vs est error

		# Approximately synchronizes messages from subscribers by their timestamps
		# self.time_synch = message_filters.ApproximateTimeSynchronizer([self.sub_gt_pose, self.sub_est_pose, self.sub_dr_pose], 
		# 					# queue_size=2, slop=0.005, allow_headerless=True)
		# 					queue_size=2, slop=0.1, allow_headerless=True)
		self.time_synch = message_filters.ApproximateTimeSynchronizer([self.sub_gt_pose, self.sub_dr_pose], 
							queue_size=2, slop=1, allow_headerless=True)
		self.time_synch.registerCallback(self.err_callback) # register multiple callbacks
		
		# Instantiate variables
		self.est_x_errorArray = []
		self.est_y_errorArray = []
		self.est_z_errorArray = []
		self.est_roll_errorArray = []
		self.est_pitch_errorArray = []
		self.est_yaw_errorArray = []
		self.est_vx_errorArray = []
		self.est_vy_errorArray = []
		self.est_vz_errorArray = []
		self.dr_x_errorArray = []
		self.dr_y_errorArray = []
		self.dr_z_errorArray = []
		self.dr_roll_errorArray = []
		self.dr_pitch_errorArray = []
		self.dr_yaw_errorArray = []
		self.dr_vx_errorArray = []
		self.dr_vy_errorArray = []
		self.dr_vz_errorArray = []

		self.est_dist_errorList = []
		self.dr_dist_errorList = []
		self.dist_traveled = 0 # distance traveled by vehicle

		# GT variables
		self.gt_pose_prev = zeros(shape=(15,1)) # ground truth pose array
		# DR variables
		self.dr_dist_error = 0 # dead reckoning distance error
		self.dr_error_pose = zeros(shape=(15,1)) # dead reckoning error pose array
		# Est variables
		self.est_dist_error = 0 # estimator distance error
		self.est_error_pose = zeros(shape=(15,1)) # estimator error pose array

		self.gt_x_list = array([],dtype='int32')
		self.gt_y_list = array([],dtype='int32')
		self.dr_x_list = array([],dtype='int32')
		self.dr_y_list = array([],dtype='int32')

	def err_callback(self, gt_msg, dr_msg):
		# print('COMPUTE ERROR')
		# Change incoming ground truth quarternion data into euler [rad]
		gt_quaternion = (gt_msg.pose.pose.orientation.x, gt_msg.pose.pose.orientation.y, gt_msg.pose.pose.orientation.z, gt_msg.pose.pose.orientation.w)
		gt_euler = euler_from_quaternion(gt_quaternion)
		unwrapEuler = np.unwrap(gt_euler)
		gt_angPosRoll = unwrapEuler[0]
		gt_angPosPitch = unwrapEuler[1]
		gt_angPosYaw = unwrapEuler[2]

		# Change incoming dead reckoning quarternion data into euler [rad]
		dr_quaternion = (dr_msg.pose.pose.orientation.x, dr_msg.pose.pose.orientation.y, dr_msg.pose.pose.orientation.z, dr_msg.pose.pose.orientation.w)
		dr_euler = euler_from_quaternion(dr_quaternion)

		# Instantiate arrays and variables
		self.gt_pose = array([
            [gt_msg.pose.pose.position.x],
            [gt_msg.pose.pose.position.y],
            [gt_msg.pose.pose.position.z],
            [np.rad2deg(gt_angPosRoll)],
            [np.rad2deg(gt_angPosPitch)],
            [np.rad2deg(gt_angPosYaw)],
            [gt_msg.twist.twist.linear.x],
            [gt_msg.twist.twist.linear.y],
            [gt_msg.twist.twist.linear.z],
			]) # ground truth array
		self.dr_pose = array([
            [dr_msg.pose.pose.position.x],
            [dr_msg.pose.pose.position.y],
            [dr_msg.pose.pose.position.z],
            [np.rad2deg(dr_euler[0])],
            [np.rad2deg(dr_euler[1])],
            [np.rad2deg(dr_euler[2])],
            [dr_msg.twist.twist.linear.x],
            [dr_msg.twist.twist.linear.y],
            [dr_msg.twist.twist.linear.z],
			]) # dead reckoning array

		''' TEST '''
		print('shape',self.gt_x_list.shape)
		print('gtx',self.gt_pose[0])
		print('gty',self.gt_pose[1])
		print('drx',self.dr_pose[0])
		print('drx',self.dr_pose[1])

		# Append coordinates
		self.gt_x_list = np.append(self.gt_x_list, self.gt_pose[0], axis=0)
		self.gt_y_list = np.append(self.gt_y_list, self.gt_pose[1], axis=0)
		self.dr_x_list = np.append(self.dr_x_list, self.dr_pose[0], axis=0)
		self.dr_y_list = np.append(self.dr_y_list, self.dr_pose[1], axis=0)
		
		# Absolute mean error
		x_rmse = mean_squared_error(self.gt_x_list, self.dr_x_list, squared=False)
		y_rmse = mean_squared_error(self.gt_y_list, self.dr_y_list, squared=False)
		# dr_x_rmse = Rmse(np.array(self.dr_x_list), np.array(self.gt_x_list))
		# dr_y_rmse = Rmse(np.array(self.dr_y_list), np.array(self.gt_y_list))
		self.dr_dist_error_rmse = np.sqrt(x_rmse**2+y_rmse**2)
		''' TEST '''

		# Compute distance traveled by the vehicle
		dx = EuclideanDistance(self.gt_pose[0], self.gt_pose[1], self.gt_pose_prev[0], self.gt_pose_prev[1])
		self.dist_traveled = self.dist_traveled + dx
		# Update gt pose
		self.gt_pose_prev[0] = self.gt_pose[0]
		self.gt_pose_prev[1] = self.gt_pose[1]
		self.gt_pose_prev[2] = self.gt_pose[2]

		# Publish data
		self.publish()



	# def err_callback(self, gt_msg, est_msg, dr_msg):
	# 	# print('COMPUTE ERROR')
	# 	# Change incoming ground truth quarternion data into euler [rad]
	# 	gt_quaternion = (gt_msg.pose.pose.orientation.x, gt_msg.pose.pose.orientation.y, gt_msg.pose.pose.orientation.z, gt_msg.pose.pose.orientation.w)
	# 	gt_euler = euler_from_quaternion(gt_quaternion)
	# 	unwrapEuler = np.unwrap(gt_euler)
	# 	gt_angPosRoll = unwrapEuler[0]
	# 	gt_angPosPitch = unwrapEuler[1]
	# 	gt_angPosYaw = unwrapEuler[2]

	# 	# Change incoming dead reckoning quarternion data into euler [rad]
	# 	dr_quaternion = (dr_msg.pose.pose.orientation.x, dr_msg.pose.pose.orientation.y, dr_msg.pose.pose.orientation.z, dr_msg.pose.pose.orientation.w)
	# 	dr_euler = euler_from_quaternion(dr_quaternion)

	# 	# Instantiate arrays and variables
	# 	self.gt_pose = array([
    #         [gt_msg.pose.pose.position.x],
    #         [gt_msg.pose.pose.position.y],
    #         [gt_msg.pose.pose.position.z],
    #         [np.rad2deg(gt_angPosRoll)],
    #         [np.rad2deg(gt_angPosPitch)],
    #         [np.rad2deg(gt_angPosYaw)],
    #         [gt_msg.twist.twist.linear.x],
    #         [gt_msg.twist.twist.linear.y],
    #         [gt_msg.twist.twist.linear.z],
	# 		]) # ground truth array
	# 	self.est_pose = array([
    #         [est_msg.state.x],
    #         [est_msg.state.y],
    #         [est_msg.state.z],
    #         [est_msg.state.roll],
    #         [est_msg.state.pitch],
    #         [est_msg.state.yaw],
    #         [est_msg.state.vx],
    #         [est_msg.state.vy],
    #         [est_msg.state.vz]
	# 		]) # estimator array
	# 	self.dr_pose = array([
    #         [dr_msg.pose.pose.position.x],
    #         [dr_msg.pose.pose.position.y],
    #         [dr_msg.pose.pose.position.z],
    #         [np.rad2deg(dr_euler[0])],
    #         [np.rad2deg(dr_euler[1])],
    #         [np.rad2deg(dr_euler[2])],
    #         [dr_msg.twist.twist.linear.x],
    #         [dr_msg.twist.twist.linear.y],
    #         [dr_msg.twist.twist.linear.z],
	# 		]) # dead reckoning array

	# 	### Method 1 ###	
	# 	# # EVALUATE: Compute the difference between the ground truth and estimator coordinates
	# 	# self.est_error_pose[0] = np.abs(gt_pose[0] - est_pose[0]) # x
	# 	# self.est_error_pose[1] = np.abs(gt_pose[1] - est_pose[1]) # y
	# 	# self.est_error_pose[2] = np.abs(gt_pose[2] - est_pose[2]) # z
	# 	# self.est_error_pose[3] = np.abs(gt_pose[3] - est_pose[3]) # roll
	# 	# self.est_error_pose[4] = np.abs(gt_pose[4] - est_pose[4]) # pitch
	# 	# self.est_error_pose[5] = np.abs(gt_pose[5] - est_pose[5]) # yaw
	# 	# self.est_error_pose[6] = np.abs(gt_pose[6] - est_pose[6]) # vx
	# 	# self.est_error_pose[7] = np.abs(gt_pose[7] - est_pose[7]) # vy
	# 	# self.est_error_pose[8] = np.abs(gt_pose[8] - est_pose[8]) # vz
	# 	# # EVALUATE: Compute the euclidean distance between the ground truth and estimator coordinates
	# 	# self.est_dist_error = EuclideanDistance(gt_pose[0], gt_pose[1], est_pose[0], est_pose[1])	
		
	# 	# # EVALUATE: Compute the difference between the ground truth and DR coordinates
	# 	# self.dr_error_pose[0] = np.abs(gt_pose[0] - dr_pose[0]) # x
	# 	# self.dr_error_pose[1] = np.abs(gt_pose[1] - dr_pose[1]) # y
	# 	# self.dr_error_pose[2] = np.abs(gt_pose[2] - dr_pose[2]) # z
	# 	# self.dr_error_pose[3] = np.abs(gt_pose[3] - dr_pose[3]) # roll
	# 	# self.dr_error_pose[4] = np.abs(gt_pose[4] - dr_pose[4]) # pitch
	# 	# self.dr_error_pose[5] = np.abs(gt_pose[5] - dr_pose[5]) # yaw
	# 	# self.dr_error_pose[6] = np.abs(gt_pose[6] - dr_pose[6]) # vx
	# 	# self.dr_error_pose[7] = np.abs(gt_pose[7] - dr_pose[7]) # vy
	# 	# self.dr_error_pose[8] = np.abs(gt_pose[8] - dr_pose[8]) # vz
	# 	# # EVALUATE: Compute the euclidean distance between the ground truth and DR coordinates
	# 	# self.dr_dist_error = EuclideanDistance(gt_pose[0], gt_pose[1], dr_pose[0], dr_pose[1])

	# 	# # Distance error
	# 	# self.est_dist_errorList.append(self.est_dist_error)
	# 	# self.est_dist_errorArray = np.array(self.est_dist_errorList)
	# 	# self.est_dist_error_rmse = Rmse(self.est_dist_errorArray)

	# 	# self.dr_dist_errorList.append(self.dr_dist_error)
	# 	# self.dr_dist_errorArray = np.array(self.dr_dist_errorList)
	# 	# self.dr_dist_error_rmse = Rmse(self.dr_dist_errorArray)

	# 	# # EVALUATE: estimator RMSE
	# 	# self.est_x_errorArray.append(self.est_error_pose[0])
	# 	# self.est_y_errorArray.append(self.est_error_pose[1])
	# 	# self.est_z_errorArray.append(self.est_error_pose[2])
	# 	# self.est_roll_errorArray.append(self.est_error_pose[3])
	# 	# self.est_pitch_errorArray.append(self.est_error_pose[4])
	# 	# self.est_yaw_errorArray.append(self.est_error_pose[5])
	# 	# self.est_vx_errorArray.append(self.est_error_pose[6])
	# 	# self.est_vy_errorArray.append(self.est_error_pose[7])
	# 	# self.est_vz_errorArray.append(self.est_error_pose[8])
	# 	# self.est_x_rmse = Rmse(self.est_x_errorArray)
	# 	# self.est_y_rmse = Rmse(self.est_y_errorArray)
	# 	# self.est_z_rmse = Rmse(self.est_z_errorArray)
	# 	# self.est_roll_rmse = Rmse(self.est_roll_errorArray)
	# 	# self.est_pitch_rmse = Rmse(self.est_pitch_errorArray)
	# 	# self.est_yaw_rmse = Rmse(self.est_yaw_errorArray)
	# 	# self.est_vx_rmse = Rmse(self.est_vx_errorArray)
	# 	# self.est_vy_rmse = Rmse(self.est_vy_errorArray)
	# 	# self.est_vz_rmse = Rmse(self.est_vz_errorArray)

	# 	# # EVALUATE: dead reckoning RMSE
	# 	# self.dr_x_errorArray.append(self.dr_error_pose[0])
	# 	# self.dr_y_errorArray.append(self.dr_error_pose[1])
	# 	# self.dr_z_errorArray.append(self.dr_error_pose[2])
	# 	# self.dr_roll_errorArray.append(self.dr_error_pose[3])
	# 	# self.dr_pitch_errorArray.append(self.dr_error_pose[4])
	# 	# self.dr_yaw_errorArray.append(self.dr_error_pose[5])
	# 	# self.dr_vx_errorArray.append(self.dr_error_pose[6])
	# 	# self.dr_vy_errorArray.append(self.dr_error_pose[7])
	# 	# self.dr_vz_errorArray.append(self.dr_error_pose[8])
	# 	# self.dr_x_rmse = Rmse(self.dr_x_errorArray)
	# 	# self.dr_y_rmse = Rmse(self.dr_y_errorArray)
	# 	# self.dr_z_rmse = Rmse(self.dr_z_errorArray)
	# 	# self.dr_roll_rmse = Rmse(self.dr_roll_errorArray)
	# 	# self.dr_pitch_rmse = Rmse(self.dr_pitch_errorArray)
	# 	# self.dr_yaw_rmse = Rmse(self.dr_yaw_errorArray)
	# 	# self.dr_vx_rmse = Rmse(self.dr_vx_errorArray)
	# 	# self.dr_vy_rmse = Rmse(self.dr_vy_errorArray)
	# 	# self.dr_vz_rmse = Rmse(self.dr_vz_errorArray)

	# 	''' TEST '''
	# 	# Append coordinates
	# 	self.gt_x_list.append(self.gt_pose[0])
	# 	self.gt_y_list.append(self.gt_pose[1])
	# 	self.dr_x_list.append(self.dr_pose[0])
	# 	self.dr_y_list.append(self.dr_pose[1])
	# 	self.est_x_list.append(self.est_pose[0])
	# 	self.est_y_list.append(self.est_pose[1])
		
	# 	# Absolute mean error
	# 	# x_rmse = mean_squared_error(self.gt_x_list, self.dr_x_list, squared=False)
	# 	# y_rmse = mean_squared_error(self.gt_y_list, self.dr_y_list, squared=False)
	# 	dr_x_rmse = Rmse(np.array(self.dr_x_list), np.array(self.gt_x_list))
	# 	dr_y_rmse = Rmse(np.array(self.dr_y_list), np.array(self.gt_y_list))
	# 	self.dr_dist_error_rmse = np.sqrt(dr_x_rmse**2+dr_y_rmse**2)

	# 	est_x_rmse = Rmse(np.array(self.est_x_list), np.array(self.gt_x_list))
	# 	est_y_rmse = Rmse(np.array(self.est_y_list), np.array(self.gt_y_list))
	# 	self.est_dist_error_rmse = np.sqrt(est_x_rmse**2+est_y_rmse**2)

	# 	''' TEST '''

	# 	# Compute distance traveled by the vehicle
	# 	dx = EuclideanDistance(self.gt_pose[0], self.gt_pose[1], self.gt_pose_prev[0], self.gt_pose_prev[1])
	# 	self.dist_traveled = self.dist_traveled + dx
	# 	# Update gt pose
	# 	self.gt_pose_prev[0] = self.gt_pose[0]
	# 	self.gt_pose_prev[1] = self.gt_pose[1]
	# 	self.gt_pose_prev[2] = self.gt_pose[2]

	# 	# Publish data
	# 	self.publish()

	def publish(self):
		# # Estimator error
		# est_err_msg = Error()
		# est_err_msg.header.stamp = rospy.Time.now()
		# est_err_msg.x = self.est_error_pose[0]
		# est_err_msg.y = self.est_error_pose[1]
		# est_err_msg.z = self.est_error_pose[2]
		# est_err_msg.roll = self.est_error_pose[3]
		# est_err_msg.pitch = self.est_error_pose[4]
		# est_err_msg.yaw = self.est_error_pose[5]
		# est_err_msg.vx = self.est_error_pose[6]
		# est_err_msg.vy = self.est_error_pose[7]
		# est_err_msg.vz = self.est_error_pose[8]
		# # est_err_msg.rmse_x = self.est_x_rmse
		# # est_err_msg.rmse_y = self.est_y_rmse
		# # est_err_msg.rmse_z = self.est_z_rmse
		# # est_err_msg.rmse_roll = self.est_roll_rmse
		# # est_err_msg.rmse_pitch = self.est_pitch_rmse
		# # est_err_msg.rmse_yaw = self.est_yaw_rmse
		# # est_err_msg.rmse_vx = self.est_vx_rmse
		# # est_err_msg.rmse_vy = self.est_vy_rmse
		# # est_err_msg.rmse_vz = self.est_vz_rmse
		# est_err_msg.rmse_dist = self.est_dist_error_rmse
		# est_err_msg.dist_error = self.est_dist_error
		# est_err_msg.dist_traveled = self.dist_traveled
		# self.pub_est_err.publish(est_err_msg)

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
		# dr_err_msg.rmse_x = self.dr_x_rmse
		# dr_err_msg.rmse_y = self.dr_y_rmse
		# dr_err_msg.rmse_z = self.dr_z_rmse
		# dr_err_msg.rmse_roll = self.dr_roll_rmse
		# dr_err_msg.rmse_pitch = self.dr_pitch_rmse
		# dr_err_msg.rmse_yaw = self.dr_yaw_rmse
		# dr_err_msg.rmse_vx = self.dr_vx_rmse
		# dr_err_msg.rmse_vy = self.dr_vy_rmse
		# dr_err_msg.rmse_vz = self.dr_vz_rmse
		dr_err_msg.rmse_dist = self.dr_dist_error_rmse
		dr_err_msg.dist_error = self.dr_dist_error
		dr_err_msg.dist_traveled = self.dist_traveled
		self.pub_dr_err.publish(dr_err_msg)

def main(args):
    rospy.init_node('ComputeError', anonymous=True)
    rospy.loginfo("Starting computeErrorRosNode.py")
    err = ErrorAnalysis()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)