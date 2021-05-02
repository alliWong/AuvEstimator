# AuvEstimator
# Unmanned under water vehicle state estimator using EKF and FGO
*** WORK IN PROGRESS ***
## GTSAM

### Graph formulation 
![factorGraph](/Pictures/factorGraph.png)

### Sensors
* DVL
* barometer
* IMU

### Input measurements into GTSAM
* Robot body frame velocity (DVL)
* Navigation frame linear position z (barometer)
* Robot body frame linear acceleration + angular velocity (IMU)

### Output data from GTSAM
* Navigation frame linear position x, y, z (IMU preintegration - NavState)
* Navigation frame linear velocity x, y, z (DVL)
* Navigation frame angular position (IMU preintegration - NavState)
* Robot body frame acceleration bias (IMU preintegration - NavState)
* Robot body frame angular velocity bias position (IMU preintegration - NavState)


### List of relevant program files:
File | Description
-----|------------
processBag.launch | File to run gtsam core files on bag data
![uuvGtsam.py](/src/StateEstimator/uuvGtsam.py) | GTSAM state estimator script.
![uuvGtsamRosNode.py](/src/StateEstimator/uuvGtsamRosNode.py) | Ros wrapper for GTSAM base code (uuvGtsam.py)
![plots.py](/src/StateEstimator/plots.py) | File to generate plots.
![errorAnalysis.py](/src/StateEstimator/errorAnalysis.py) | Computes RMSE between the EKF, FGO, and dead reckoning against ground truth.
![commons.py](/src/StateEstimator/commons.py) | General helpful functions.
![transformations.py](/src/StateEstimator/transformations.py) | File for frame transformations (e.g., quaternion, rotations, euler). Had trouble importing tf2_ros (probably because python 2.7), so generated hardcoded file.
![PriorFactorPose3Z.h](/include/factors/PriorFactorPose3Z.h) | Barometer custom factor.
![PriorFactorVel.h](/include/factors/PriorFactorVel.h) | DVL custom factor.

## EKF
### Position calculation
1) Dead reckoning where position is computed by performing euler integration on velocity measurements obtained by DVL
2) Estimator position is computed by performing trapezoidal integration method on velocity measurements obtained by DVL and IMU

## Tranforming between frames (robot body to map frame) using Fixed xyz rotation
![FrameTransformation](/images/FrameTrans.png)

## EKF detail
6 DoF loosely coupled EKF

![EKF](/images/ekf.PNG)

### 15 state
```
   x - map
   y - map
   z - map
   roll - map
   pitch - map
   yaw - map
   x velocity (u) - map
   y velocity (v) - map
   z velocity (w) - map
   x accel bias - robot
   y accel bias - robot
   z accel bias - robot
   x gyro bias - robot
   y gyro bias - robot
   z gyro bias - robot
```
### Predict step
![xPredictStep](/images/xPredictStep.png)
![pPredictStep](/images/pPredictStep.png)

### List of relevant program files:
File | Description
-----|------------
ekfTest.launch | File to launch the SensorNode.py, deadReckoning.py, and computeError.py. This file also contains ros parameters that are used in the python files. 
uuvEkfEstimator.py | Script that setup and performs the state estimation of the vehicle with EKF using DVL, IMU, and barometer. 
uuvEkfRosNode.py | Ros wrapper for EKF base code (uuvEkfEstimator.py)
deadReckon.py | Performs dead reckoning with velocity data from DVL and orientation data from IMU.
deadReckonRosNode.py | Ros wrapper for EKF base code (deadReckon.py)
ekf.py | Script that contains the math behind EKF.
computeError.py | Computes the x-error, y-error, z-error, and distance between the estimator and dead reckoning against ground truth in real time.


## System Requirements
* ROS kinetic or melodic:
* Ubuntu 18.04 LTS:
* Python 2.7 environment

## Relevant Packages
Required: Ubuntu 16.04 with ROS Kinetic (http://wiki.ros.org/kinetic). Also, the following packages are required: 

* UUV-simulator (simulator):

  https://github.com/uuvsimulator/uuv_simulator

* rexrov2 vehicle (vehicle simulator):

  https://github.com/uuvsimulator/rexrov2
  
* plotjuggler (plotting tool):

  https://github.com/facontidavide/PlotJuggler
  
## Launch Procedures (need update)
1) Install all relevant packages stated above
