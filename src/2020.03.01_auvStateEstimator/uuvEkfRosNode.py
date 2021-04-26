#! /usr/bin/env python
# """
# ROS Wrapper
# """

""" Import libraries """
import sys
import numpy as np
from numpy import array, zeros, reshape, matmul, eye, sqrt, cos, sin
from transformations import quaternion_from_euler, euler_from_quaternion
from uuvEkfEstimator import EkfEstimator
from commons import Rot

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from uuv_sensor_ros_plugins_msgs.msg import DVL
from auv_estimator.msg import State, Covariance, Inputs, Estimator

class EkfEstRosNode:
  def __init__(self):
    """ ROS Parameters """
    # Grab topics 
    sub_dvl = rospy.get_param("~dvlTopic") 
    sub_imu = rospy.get_param("~imuTopic")
    sub_depth = rospy.get_param("~depthTopic")
    # Decide which frame to use (ENU or NED)
    self.useEnu = rospy.get_param("~useEnu") # set to 1 to use ENU frame, set to 0 to NED
    # Time step property
    self.est_dt = rospy.get_param("~est_dt") # estimator time step [s]
    # DVL parameters
    self.sen_dvl_offsetX = rospy.get_param("~dvl_offsetX") # offset relative from the sensor to vehicle center of mass in x-direction
    self.sen_dvl_offsetY = rospy.get_param("~dvl_offsetY") # offset relative from the sensor to vehicle center of mass in y-direction
    self.sen_dvl_offsetZ = rospy.get_param("~dvl_offsetZ") # offset relative from the sensor to vehicle center of mass in z-direction
    # Imu parameters
    self.sen_imu_accBiasX = rospy.get_param("~imu_rbtAccBiasX")  # initial accel bias in x-direction
    self.sen_imu_accBiasY = rospy.get_param("~imu_rbtAccBiasY")  # initial accel bias in y-direction
    self.sen_imu_accBiasZ = rospy.get_param("~imu_rbtAccBiasZ")  # initial accel bias in z-direction
    self.sen_imu_gyrBiasX = rospy.get_param("~imu_rbtGyrBiasX")  # initial gyro bias in x-direction
    self.sen_imu_gyrBiasY = rospy.get_param("~imu_rbtGyrBiasY")  # initial gyro bias in y-direction
    self.sen_imu_gyrBiasZ = rospy.get_param("~imu_rbtGyrBiasZ")  # initial gyro bias in z-direction
    # Decide which sensors to use for pseudo linear position calculation    
    self.est_useDvlLinPos = rospy.get_param("~est_useDvlLinPos") # set to 1 to use DVL trapezoidal method, set to 0 to ignore
    self.est_useImuLinPos = rospy.get_param("~est_useImuLinPos") # set to 1 to use IMU trapezoidal method, set to 0 to ignore
    # Robot frame estimator measurement noise covariance
    self.est_R_linPos = rospy.get_param("~est_R_linPos") # linear position [m]
    self.est_R_linPosZ = rospy.get_param("~est_R_linPosZ") # linear position [m]
    self.est_R_angPos = rospy.get_param("~est_R_angPos") # rotation [rad]
    self.est_R_linVel = rospy.get_param("~est_R_linVel") # linear velocity [m/s]
    self.est_R_acBias = rospy.get_param("~est_R_acBias") # acceleration bias [m^2/s]
    self.est_R_gyBias = rospy.get_param("~est_R_gyBias") # gyro bias [deg/s]
    # Robot frame estimator process noise covariance
    self.est_Q_linPos = rospy.get_param("~est_Q_linPos") # linear position [m]
    self.est_Q_angPos = rospy.get_param("~est_Q_angPos") # rotation [rad]
    self.est_Q_linVel = rospy.get_param("~est_Q_linVel") # linear velocity [m/s]
    self.est_Q_acBias = rospy.get_param("~est_Q_acBias") # acceleration bias [m^2/s]
    self.est_Q_gyBias = rospy.get_param("~est_Q_gyBias") # gyro bias [deg/s]

    """ Estimator driver wrapper """
    # Decide which sensor to calculate pseudo position estimate
    if self.est_useDvlLinPos == 1:
      pseudoLinPos = 0
    if self.est_useImuLinPos == 1:
      pseudoLinPos = 1
    if self.est_useDvlLinPos == 1 and self.est_useImuLinPos == 1:
      pseudoLinPos = 2
    # Initialize estimator
    self.uuvEkfEst = EkfEstimator(self.est_dt, self.useEnu, pseudoLinPos)
    # Construct process and sensor covariance matrices
    self.uuvEkfEst.ComputeQ(self.est_Q_linPos, self.est_Q_angPos, self.est_Q_linVel, self.est_Q_acBias, self.est_Q_gyBias)
    self.uuvEkfEst.ComputeR(self.est_R_linPosZ, self.est_R_angPos, self.est_R_linVel, self.est_R_acBias, self.est_R_gyBias)
    
    """ Setup publishers/subscribers """
    # Subscribers
    self.sub_dvl = rospy.Subscriber(sub_dvl, DVL, self.DvlCallback) 
    self.sub_imu = rospy.Subscriber(sub_imu, Imu, self.ImuCallback) 
    self.sub_depth = rospy.Subscriber(sub_depth, Odometry, self.DepthCallback)
    # Publishers
    self.pub_est_pose = rospy.Publisher('/est/map/pose', Odometry, queue_size=2)
    self.pub_est_state = rospy.Publisher('/est/state', Estimator, queue_size=2)

  """ Raw sensor measurements """
  def DvlCallback(self, msg):
    # Instantiate dvl variables
    dvl_time = rospy.get_time()

    # Setup robot velocity array
    dvl_rbtLinVel = array([[msg.velocity.x],
                          [msg.velocity.y],
                          [msg.velocity.z]])
    # Setup dvl offset array
    dvl_offset = array([[self.sen_dvl_offsetX], 
                        [self.sen_dvl_offsetY], 
                        [self.sen_dvl_offsetZ]])

    # Initialize estimator 
    self.uuvEkfEst.DvlCallback(dvl_rbtLinVel, dvl_offset, dvl_time)
    self.Estimator()

  def ImuCallback(self, msg):
    # Instantiate imu variables
    imu_time = rospy.get_time()

    # Setup robot acceleration array
    imu_rbtLinAcc = array([[msg.linear_acceleration.x],
                          [msg.linear_acceleration.y],
                          [msg.linear_acceleration.z]])
    # Setup robot angular velocity array
    imu_rbtAngVel = np.array([[msg.angular_velocity.x],
                              [msg.angular_velocity.y],
                              [msg.angular_velocity.z]]) 
    # Setup biases array
    imu_accBias = array([[self.sen_imu_accBiasX], 
                        [self.sen_imu_accBiasY], 
                        [self.sen_imu_accBiasZ]]) # initial accel bias
    imu_gyrBias = array([[self.sen_imu_gyrBiasX], 
                        [self.sen_imu_gyrBiasY], 
                        [self.sen_imu_gyrBiasZ]]) # initial gyro bias
    # Setup map orientation array
    euler = euler_from_quaternion([msg.orientation.x,
                                  msg.orientation.y,
                                  msg.orientation.z,
                                  msg.orientation.w]) # Convert IMU data from quarternion to euler
    unwrapEuler = np.unwrap(euler) # unwrap euler angles
    imu_mapEulAng = np.array([[unwrapEuler[0]],
                              [unwrapEuler[1]],
                              [unwrapEuler[2]]]) # imu angular position array  
    ### NOTES: Incoming IMU data is in rotation matrix form  ###
    imu_mapAngPos = Rot(imu_mapEulAng[0], imu_mapEulAng[1], imu_mapEulAng[2])

    # Initialize estimator
    self.uuvEkfEst.ImuCallback(imu_rbtLinAcc, imu_rbtAngVel, imu_mapAngPos, imu_mapEulAng, imu_accBias, imu_gyrBias, imu_time)
    self.Estimator()

  def DepthCallback(self, msg):
    # Instantiate barometer variables
    sen_bar_time = rospy.get_time()

    # Setup robot z position array
    sen_bar_mapLinPos = msg.pose.pose.position.z

    # Initialize estimator
    self.uuvEkfEst.DepthCallback(sen_bar_mapLinPos, sen_bar_time)
    self.Estimator()

  """ Run Estimator """ 
  def Estimator(self):
    # Get estimator state
    self.est_x, self.est_P, self.est_u = self.uuvEkfEst.OutputEst()

    # Publish ROS messages 
    self.PubEstimator() # publish estimator messages

  """ Publish """
  def PubEstimator(self):
    # Publish estimator state message
    est_msg = Estimator()
    est_msg.header.stamp = rospy.Time.now()
    est_msg.header.frame_id = "world"
    est_msg.state.x = self.est_x[0] # x
    est_msg.state.y = self.est_x[1] # y
    est_msg.state.z = self.est_x[2] # z
    est_msg.state.roll = np.rad2deg(self.est_x[3]) # roll
    est_msg.state.pitch = np.rad2deg(self.est_x[4]) # pitch
    est_msg.state.yaw = np.rad2deg(self.est_x[5]) # yaw
    est_msg.state.vx = self.est_x[6] # dx
    est_msg.state.vy = self.est_x[7] # dy
    est_msg.state.vz = self.est_x[8] # dz
    est_msg.state.bias_ax = self.est_x[9] # x accel bias
    est_msg.state.bias_ay = self.est_x[10] # y accel bias
    est_msg.state.bias_az = self.est_x[11] # z accel bias
    est_msg.state.bias_gx = self.est_x[12] # x gyro bias
    est_msg.state.bias_gy = self.est_x[13] # y gyro bias
    est_msg.state.bias_gz = self.est_x[14] # z gyro bias
    est_msg.covariance.x = self.est_P[0,0] # 
    est_msg.covariance.y = self.est_P[1,1] # 
    est_msg.covariance.z = self.est_P[2,2] # 
    est_msg.covariance.roll = self.est_P[3,3] #
    est_msg.covariance.pitch = self.est_P[4,4] # 
    est_msg.covariance.yaw = self.est_P[5,5] # 
    est_msg.covariance.vx = self.est_P[6,6] # 
    est_msg.covariance.vy = self.est_P[7,7] #
    est_msg.covariance.vz = self.est_P[8,8] #
    est_msg.covariance.bias_ax = self.est_P[9,9] # 
    est_msg.covariance.bias_ay = self.est_P[10,10] #
    est_msg.covariance.bias_az = self.est_P[11,11] # 
    est_msg.covariance.bias_gx = self.est_P[12,12] # x gyro bias
    est_msg.covariance.bias_gy = self.est_P[13,13] # y gyro bias
    est_msg.covariance.bias_gz = self.est_P[14,14] # z gyro bias
    est_msg.inputs.ax = self.est_u[0] # 
    est_msg.inputs.ay = self.est_u[1] # 
    est_msg.inputs.az = self.est_u[2] # 
    est_msg.inputs.gx = self.est_u[3] # 
    est_msg.inputs.gy = self.est_u[4] # 
    est_msg.inputs.gz = self.est_u[5] # 
    self.pub_est_state.publish(est_msg) # publish estimator state message

def main(args):
    rospy.init_node('EkfEst', anonymous=True)
    rospy.loginfo("Starting uuvEkfEstRosNode.py")
    est = EkfEstRosNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)