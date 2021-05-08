#!/usr/bin/env python 
"""
This file computes the x-error, y-error, z-error, and distance between 
the estimator and dead reckoning against ground truth.
"""
import sys
import numpy as np
from numpy import array, zeros
# import tf
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
from transformations import quaternion_from_euler, euler_from_quaternion

import rospy
import message_filters
from nav_msgs.msg import Odometry
from commons import Distance, SkewSymmetric, Rot, TrapIntegrate, MapAngVelTrans, PressureToDepth, Rmse, EuclideanDistance
from auv_estimator.msg import State, Covariance, Inputs, Estimator, Error
from sklearn.metrics import mean_squared_error

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
		self.time_synch = message_filters.ApproximateTimeSynchronizer([self.sub_gt_pose, self.sub_est_pose, self.sub_dr_pose], 
							queue_size=2, slop=0.01, allow_headerless=True)
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
		self.dist_traveled = 0 # distance traveled by vehicle

		self.est_dist_error = 0 # estimator distance error
		self.dr_dist_error = 0 # dead reckoning distance error
		self.est_error_pose = zeros(shape=(15,1)) # estimator error pose array
		self.dr_error_pose = zeros(shape=(15,1)) # dead reckoning error pose array
		self.gt_pose_prev = zeros(shape=(15,1)) # ground truth pose array
	
		self.gt_x_list = array([],dtype='int32')
		self.gt_y_list = array([],dtype='int32')
		self.dr_x_list = array([],dtype='int32')
		self.dr_y_list = array([],dtype='int32')
		self.est_x_list = array([],dtype='int32')
		self.est_y_list = array([],dtype='int32')

	def err_callback(self, gt_msg, est_msg, dr_msg):
		print('COMPUTE ERROR')
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
		self.est_pose = array([
            [est_msg.state.x],
            [est_msg.state.y],
            [est_msg.state.z],
            [est_msg.state.roll],
            [est_msg.state.pitch],
            [est_msg.state.yaw],
            [est_msg.state.vx],
            [est_msg.state.vy],
            [est_msg.state.vz]
			]) # estimator array
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

		# ### Method 1 ###	
		# Append variables
		self.gt_x_list = np.append(self.gt_x_list, self.gt_pose[0], axis=0)
		self.gt_y_list = np.append(self.gt_y_list, self.gt_pose[1], axis=0)
		self.dr_x_list = np.append(self.dr_x_list, self.dr_pose[0], axis=0)
		self.dr_y_list = np.append(self.dr_y_list, self.dr_pose[1], axis=0)
		self.est_x_list = np.append(self.est_x_list, self.est_pose[0], axis=0)
		self.est_y_list = np.append(self.est_y_list, self.est_pose[1], axis=0)

		# Calculate DR RMSE
		x_rmse = mean_squared_error(self.gt_x_list, self.dr_x_list, squared=False) # calculate y rmse
		y_rmse = mean_squared_error(self.gt_y_list, self.dr_y_list, squared=False) # calculate x rmse
		self.dr_dist_error_rmse = Distance(x_rmse, y_rmse) # calculate distance
		# Calculate Est RMSE
		est_x_rmse = mean_squared_error(self.gt_x_list, self.est_x_list, squared=False) # calculate y rmse
		est_y_rmse = mean_squared_error(self.gt_y_list, self.est_y_list, squared=False) # calculate x rmse
		self.est_dist_error_rmse = Distance(est_x_rmse, est_y_rmse) # calculate distance

		# Compute distance traveled by the vehicle
		dx = EuclideanDistance(self.gt_pose[0], self.gt_pose[1], self.gt_pose_prev[0], self.gt_pose_prev[1])
		self.dist_traveled = self.dist_traveled + dx
		# Update gt pose
		self.gt_pose_prev[0] = self.gt_pose[0]
		self.gt_pose_prev[1] = self.gt_pose[1]
		self.gt_pose_prev[2] = self.gt_pose[2]

		# Publish data
		self.publish()

	def publish(self):
		# Estimator error
		est_err_msg = Error()
		est_err_msg.header.stamp = rospy.Time.now()
		# est_err_msg.x = self.est_error_pose[0]
		# est_err_msg.y = self.est_error_pose[1]
		# est_err_msg.z = self.est_error_pose[2]
		# est_err_msg.roll = self.est_error_pose[3]
		# est_err_msg.pitch = self.est_error_pose[4]
		# est_err_msg.yaw = self.est_error_pose[5]
		# est_err_msg.vx = self.est_error_pose[6]
		# est_err_msg.vy = self.est_error_pose[7]
		# est_err_msg.vz = self.est_error_pose[8]
		# est_err_msg.rmse_x = self.est_x_rmse
		# est_err_msg.rmse_y = self.est_y_rmse
		# est_err_msg.rmse_z = self.est_z_rmse
		# est_err_msg.rmse_roll = self.est_roll_rmse
		# est_err_msg.rmse_pitch = self.est_pitch_rmse
		# est_err_msg.rmse_yaw = self.est_yaw_rmse
		# est_err_msg.rmse_vx = self.est_vx_rmse
		# est_err_msg.rmse_vy = self.est_vy_rmse
		# est_err_msg.rmse_vz = self.est_vz_rmse
		est_err_msg.rmse_dist = self.est_dist_error_rmse
		est_err_msg.dist_error = self.est_dist_error
		est_err_msg.dist_traveled = self.dist_traveled
		self.pub_est_err.publish(est_err_msg)

		# DR error
		dr_err_msg = Error()
		dr_err_msg.header.stamp = rospy.Time.now()
		# dr_err_msg.x = self.dr_error_pose[0]
		# dr_err_msg.y = self.dr_error_pose[1]
		# dr_err_msg.z = self.dr_error_pose[2]
		# dr_err_msg.roll = self.dr_error_pose[3]
		# dr_err_msg.pitch = self.dr_error_pose[4]
		# dr_err_msg.yaw = self.dr_error_pose[5]
		# dr_err_msg.vx = self.dr_error_pose[6]
		# dr_err_msg.vy = self.dr_error_pose[7]
		# dr_err_msg.vz = self.dr_error_pose[8]
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
    rospy.loginfo("Starting computeError.py")
    err = ErrorAnalysis()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)










# #!/usr/bin/env python 
# """
# This file computes the x-error, y-error, z-error, and distance between 
# the estimator and dead reckoning against ground truth.
# """
# import sys
# import numpy as np
# from numpy import array, zeros
# from transformations import quaternion_from_euler, euler_from_quaternion
# from sklearn.metrics import mean_squared_error

# import rospy
# import message_filters
# from nav_msgs.msg import Odometry
# from commons import EuclideanDistance, SkewSymmetric, Rot, TrapIntegrate, MapAngVelTrans, PressureToDepth, Rmse
# from auv_estimator.msg import State, Covariance, Inputs, Estimator, Error
# from computeError import ErrorAnalysis

# """ Compute error driver wrapper """
# Err = ErrorAnalysis()

# class ErrorAnalysisRosNode(object):
# 	def __init__(self):
# 		""" ROS Parameters """
# 		# Grab topics 
# 		sub_pose_gt = rospy.get_param("~groundTruthTopic") # grab ground truth ros parameters
# 		sub_pose_est = rospy.get_param("~estimatorTopic") # grab estimator ros parameters
# 		sub_pose_dr = rospy.get_param("~deadReckoningTopic") # grab estimator ros parameters

# 		""" Setup publishers/subscribers """
# 		# Publishers
# 		self.pub_est_err = rospy.Publisher('/est/error', Error, queue_size=2) # publish gt vs est error
# 		self.pub_dr_err = rospy.Publisher('/dr/error', Error, queue_size=2) # publish gt vs est error
# 		# Subscribers
# 		self.sub_gt_pose = message_filters.Subscriber(sub_pose_gt, Odometry) # subscribe to ground truth pose
# 		self.sub_dr_pose = message_filters.Subscriber(sub_pose_dr, Odometry) # subscribe to estimator pose
# 		self.sub_est_pose = message_filters.Subscriber(sub_pose_est, Estimator) # subscribe to estimator pose
# 		# Approximately synchronizes messages from subscribers by their timestamps
# 		# self.dr_time_synch = message_filters.ApproximateTimeSynchronizer([self.sub_gt_pose, self.sub_dr_pose], 
# 		# 					queue_size=1000, slop=0.1, allow_headerless=True)
# 		# self.dr_time_synch.registerCallback(self.dr_err_callback) # register multiple callbacks

# 		self.est_time_synch = message_filters.ApproximateTimeSynchronizer([self.sub_gt_pose, self.sub_est_pose], 
# 							queue_size=1000, slop=0.1, allow_headerless=True)
# 		self.est_time_synch.registerCallback(self.est_err_callback) # register multiple callbacks

# 	def dr_err_callback(self, gt_msg, dr_msg):
# 		# Change incoming ground truth quarternion data into euler [rad]
# 		gt_quaternion = (gt_msg.pose.pose.orientation.x, gt_msg.pose.pose.orientation.y, gt_msg.pose.pose.orientation.z, gt_msg.pose.pose.orientation.w)
# 		gt_euler = np.unwrap(euler_from_quaternion(gt_quaternion))

# 		# Change incoming dead reckoning quarternion data into euler [rad]
# 		dr_quaternion = (dr_msg.pose.pose.orientation.x, dr_msg.pose.pose.orientation.y, dr_msg.pose.pose.orientation.z, dr_msg.pose.pose.orientation.w)
# 		dr_euler = euler_from_quaternion(dr_quaternion)

# 		# Instantiate arrays and variables
# 		self.gt_pose = array([[gt_msg.pose.pose.position.x],
# 							[gt_msg.pose.pose.position.y],
# 							[gt_msg.pose.pose.position.z],
# 							[np.rad2deg(gt_euler[0])],
# 							[np.rad2deg(gt_euler[1])],
# 							[np.rad2deg(gt_euler[2])],
# 							[gt_msg.twist.twist.linear.x],
# 							[gt_msg.twist.twist.linear.y],
# 							[gt_msg.twist.twist.linear.z]]) # ground truth array
# 		self.dr_pose = array([[dr_msg.pose.pose.position.x],
# 							[dr_msg.pose.pose.position.y],
# 							[dr_msg.pose.pose.position.z],
# 							[np.rad2deg(dr_euler[0])],
# 							[np.rad2deg(dr_euler[1])],
# 							[np.rad2deg(dr_euler[2])],
# 							[dr_msg.twist.twist.linear.x],
# 							[dr_msg.twist.twist.linear.y],
# 							[dr_msg.twist.twist.linear.z]]) # dead reckoning array

# 		self.dr_dist_traveled, self.dr_dist_error_rmse = Err.DrAnalysis(self.gt_pose, self.dr_pose)

# 		# Publish 
# 		self.PublishDr()

# 	def est_err_callback(self, gt_msg, est_msg):
# 		# Change incoming ground truth quarternion data into euler [rad]
# 		gt_quaternion = (gt_msg.pose.pose.orientation.x, gt_msg.pose.pose.orientation.y, gt_msg.pose.pose.orientation.z, gt_msg.pose.pose.orientation.w)
# 		gt_euler = np.unwrap(euler_from_quaternion(gt_quaternion))

# 		# Instantiate arrays and variables
# 		self.gt_pose = array([[gt_msg.pose.pose.position.x],
# 							[gt_msg.pose.pose.position.y],
# 							[gt_msg.pose.pose.position.z],
# 							[np.rad2deg(gt_euler[0])],
# 							[np.rad2deg(gt_euler[1])],
# 							[np.rad2deg(gt_euler[2])],
# 							[gt_msg.twist.twist.linear.x],
# 							[gt_msg.twist.twist.linear.y],
# 							[gt_msg.twist.twist.linear.z]]) # ground truth array
# 		self.est_pose = array([[est_msg.state.x],
# 								[est_msg.state.y],
# 								[est_msg.state.z],
# 								[est_msg.state.roll],
# 								[est_msg.state.pitch],
# 								[est_msg.state.yaw],
# 								[est_msg.state.vx],
# 								[est_msg.state.vy],
# 								[est_msg.state.vz]]) # estimator array

# 		self.est_dist_traveled, self.est_dist_error_rmse = Err.EstAnalysis(self.gt_pose, self.est_pose)

# 		# Publish 
# 		self.PublishEst()

# 	def PublishDr(self):
# 		# DR error
# 		dr_err_msg = Error()
# 		dr_err_msg.header.stamp = rospy.Time.now()
# 		dr_err_msg.rmse_dist = self.dr_dist_error_rmse
# 		dr_err_msg.dist_traveled = self.dr_dist_traveled
# 		self.pub_dr_err.publish(dr_err_msg)

# 	def PublishEst(self):
# 		# Est error
# 		est_err_msg = Error()
# 		est_err_msg.header.stamp = rospy.Time.now()
# 		est_err_msg.rmse_dist = self.est_dist_error_rmse
# 		est_err_msg.dist_traveled = self.est_dist_traveled
# 		self.pub_est_err.publish(est_err_msg)

# def main(args):
#     rospy.init_node('ComputeError', anonymous=True)
#     rospy.loginfo("Starting computeErrorRosNode.py")
#     err = ErrorAnalysisRosNode()
    
#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         print("Shutting down")
#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main(sys.argv)