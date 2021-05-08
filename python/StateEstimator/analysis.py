#!/usr/bin/env python 
"""
This file analyze the estimator. Two methods are applied
1) Error comparison between estimator and ground truth at the same time step (computeError.py)
   Euclidean distance comparison between estimator and ground truth at the same time step (computeError.py)
   RMSE between the estimator and ground truth
"""
import csv
import numpy as np
from commons import Rmse

# CSV file name 
filename = "/home/allison/Workspace/AuvWs/src/auv_estimator/Data/Trial3ComputeError.csv"

# Initialize list
est_x_error = []
est_y_error = []
est_z_error = []
est_roll_error = []
est_pitch_error = []
est_yaw_error = []
est_vx_error = []
est_vy_error = []
est_vz_error = []
est_dist_error = []


dr_x_error = []
dr_y_error = []
dr_z_error = []
dr_roll_error = []
dr_pitch_error = []
dr_yaw_error = []
dr_vx_error = []
dr_vy_error = []
dr_vz_error = []
dr_dist_error = []

# Read csv file 
with open(filename, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) # Creating a csv reader object 
    next(csvreader) # Skip first row (header row)

    for row in csvreader:
        # EST
        est_x_error.append(float(row[1]))
        est_y_error.append(float(row[2]))
        est_z_error.append(float(row[3]))
        est_roll_error.append(float(row[4]))
        est_pitch_error.append(float(row[5]))
        est_yaw_error.append(float(row[6]))
        est_vx_error.append(float(row[7]))
        est_vy_error.append(float(row[8]))
        est_vz_error.append(float(row[9]))

        # DR
        dr_x_error.append(float(row[1]))
        dr_y_error.append(float(row[2]))
        dr_z_error.append(float(row[3]))
        dr_roll_error.append(float(row[4]))
        dr_pitch_error.append(float(row[5]))
        dr_yaw_error.append(float(row[6]))
        dr_vx_error.append(float(row[7]))
        dr_vy_error.append(float(row[8]))
        dr_vz_error.append(float(row[9]))
        
        # Distance
        est_dist_error.append(float(row[19]))
        dr_dist_error.append(float(row[19]))

    # Get total number of rows (data points)
    print("Total no. of data: %d"%(csvreader.line_num)) 

### Method 1 ###
# RMSE calculation
est_x_rmse = rmse(est_x_error)
est_y_rmse = rmse(est_y_error)
est_z_rmse = rmse(est_z_error)
est_roll_rmse = rmse(est_roll_error)
est_pitch_rmse = rmse(est_pitch_error)
est_yaw_rmse = rmse(est_yaw_error)
est_vx_rmse = rmse(est_vx_error)
est_vy_rmse = rmse(est_vy_error)
est_vz_rmse = rmse(est_vz_error)
est_dist_rmse = rmse(est_dist_error)

dr_x_rmse = rmse(dr_x_error)
dr_y_rmse = rmse(dr_y_error)
dr_z_rmse = rmse(dr_z_error)
dr_roll_rmse = rmse(dr_roll_error)
dr_pitch_rmse = rmse(dr_pitch_error)
dr_yaw_rmse = rmse(dr_yaw_error)
dr_vx_rmse = rmse(dr_vx_error)
dr_vy_rmse = rmse(dr_vy_error)
dr_vz_rmse = rmse(dr_vz_error)
dr_dist_rmse = rmse(dr_dist_error)

# Print message
print("est x rmse error is: " + str(est_x_rmse) + " m")
print("est y rmse error is: " + str(est_y_rmse) + " m")
print("est z rmse error is: " + str(est_z_rmse) + " m")
print("est roll rmse error is: " + str(est_roll_rmse) + " deg")
print("est pitch rmse error is: " + str(est_pitch_rmse) + " deg")
print("est yaw rmse error is: " + str(est_yaw_rmse) + " deg")
print("est vx rmse error is: " + str(est_vx_rmse) + " m/s")
print("est vy rmse error is: " + str(est_vy_rmse) + " m/s")
print("est vz rmse error is: " + str(est_vz_rmse) + " m/s")
print("est distance rmse error is: " + str(est_dist_rmse) + " m")


### Method 2 ###
absErrorStdEstMapX = np.std(est_x_error)
absErrorStdEstMapY = np.std(est_y_error)
absErrorStdEstMapXY = np.sqrt(np.power(absErrorStdEstMapX,2) + np.power(absErrorStdEstMapY,2))
absErrorStdDrMapX = np.std(dr_x_error)
absErrorStdDrMapY = np.std(dr_y_error)
absErrorStdDrMapXY = np.sqrt(np.power(absErrorStdDrMapX,2) + np.power(absErrorStdDrMapY,2))

gnssToEstMeanErrorChange = (est_dist_rmse-dr_dist_rmse)/dr_dist_rmse
gnssToEstStdErrorChange = (absErrorStdEstMapXY-absErrorStdDrMapXY)/absErrorStdDrMapXY

print(gnssToEstMeanErrorChange)
print(gnssToEstStdErrorChange)