#! /usr/bin/env python
"""
This file is a compilation of functions.
"""
import numpy as np
from numpy import matrix, cos, sin, tan, power, mean, sqrt
import gtsam


# Skew symmetric function
def SkewSymmetric(s):
  return np.matrix([[    0, -s[2], s[1]],  
                  [ s[2],     0, -s[0]],
                  [-s[1],  s[0],   0]])

# Convert frame references function
def Rot(phi, theta, psi):
    Rx = np.array([
        [1,       0,       0],
        [0,  cos(phi), -sin(phi)],
        [0,  sin(phi),   cos(phi)]], dtype=object)
    Ry = np.array([
        [ cos(theta),  0,  sin(theta)],
        [      0,  1,       0],
        [-sin(theta),  0,  cos(theta)]], dtype=object)
    Rz = np.array([
        [cos(psi),  -sin(psi), 0],
        [sin(psi),   cos(psi), 0],
        [     0,        0, 1]], dtype=object)
    bodyRot = Rz.dot(Ry).dot(Rx)
    return bodyRot

# Convert angular velocity from robot to map frame
def MapAngVelTrans(phi, theta, psi):
    T = np.array([
        [1, sin(phi)*tan(theta), cos(phi)*tan(theta)],
        [0, cos(phi), -sin(phi)],
        [0, sin(phi)/cos(theta), cos(phi)/cos(theta)]], dtype=object)
    return T

# Trapezoidal integration function
def TrapIntegrate(currentTime, currentData, previousState, previousTime, previousData):
    # Calculate change in time from current and previous data 
    dt = currentTime - previousTime
    # Apply the trapezoidal integration method
    currentState = previousState+(1/2)*(previousData+currentData)*dt
    return currentState

# Distance function
def EuclideanDistance(x1, y1, x2, y2):
    dx = x2-x1
    dy = y2-y1
    dist = np.power((dx**2+dy**2), 0.5)
    return dist

# Distance function
def Distance(dx, dy):
    dist = np.sqrt(dx**2+dy**2)
    return dist

# Root mean square error function
# def Rmse(differences):
#     # differences_squared = np.power(differences,2)
#     differences_squared = differences**2
#     mean_of_differences_squared = np.mean(differences_squared) # Take mean
#     rmse_val = np.sqrt(mean_of_differences_squared)
#     return rmse_val

def Rmse(targets, predictions):
    differences = predictions - targets                      
    differences_squared = np.power(differences, 2)                    
    mean_of_differences_squared = np.mean(differences_squared)
    rmse_val = np.sqrt(mean_of_differences_squared)           
    return rmse_val    

# def Rmse(predictions, targets):
#     n = len(predictions)
#     rmse = np.linalg.norm(predictions-targets)/np.sqrt(n)     
#     return rmse    

# Pressure [Pa] to depth [m] calculation 
def PressureToDepth(pressure, barOffset):
    standPressure = 101.325 # standard pressure [1 atm]
    kPaPerM = 9.804139432 # pressure per meter [kPa/m]

    # Raw depth calculation
    sen_bar_depth = (pressure-standPressure)/kPaPerM # calculated depth from auv pressure data
    # Total depth calculation including barometer offset
    depth = -sen_bar_depth-barOffset
    
    return depth

# Create 3d double numpy np.array
def vector3(x, y, z):
    return np.array([x, y, z], dtype=np.float)


def gtsam_pose_to_numpy(gtsam_pose):
	"""Convert GTSAM pose to numpy arrays (position, orientation)"""
	position = np.array([gtsam_pose.x(),
						 gtsam_pose.y(),
						 gtsam_pose.z()])
	q = gtsam_pose.rotation().quaternion()
	orientation = np.array([q[1], q[2], q[3], q[0]]) # xyzw
	return position, orientation

def numpy_pose_to_gtsam(position, orientation):
	"""Convert numpy arrays (position, orientation) to GTSAM pose"""
	return gtsam.Pose3(gtsam.Rot3.Quaternion(orientation[3],
											 orientation[0],
											 orientation[1],
											 orientation[2]),
								gtsam.Point3(position[0],
											 position[1],
											 position[2]))