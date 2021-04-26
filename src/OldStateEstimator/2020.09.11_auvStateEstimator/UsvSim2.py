#! /usr/bin/env python
import math
import matplotlib.pyplot as plt
import numpy as np

"""
Simulation user inputs
"""
# Time properties
sim_dt = 0.001  # simulation time step [s]
startTime = 0   # simulation start time [s]
endTime = 50    # simulation end time [s]

# Map frame estimator measurement noise covariance
map_R_linPos = 2        # linear position [m]
map_R_angPos = 0.075	# angular position [rad]
map_R_linVel = 2		# linear velocity [m/s]
map_R_angVel = 0.075	# angular velocity [rad/s]

# Map frame estimator process noise covariance
map_Q_linPos = map_R_linPos*1e1     # linear position [m]

# Map frame estimator observation/observability
H_linPos = 1	# linear position (measured)
H_angPos = 1	# angular position (measured)
H_linVel = 1	# linear velocity (pseudo-measured)
H_angVel = 1	# angular velocity (measured)

"""
Estimator user inputs
"""
# Time step properties
dt = 0.25  # estimator time step [s]

# Map frame covariance, observation, state, and input matrices
map_Q = np.array([
    [map_Q_linPos, 0, 0, 0, 0, 0],
    [0, map_Q_linPos, 0, 0, 0, 0],
    [0, 0, 1e5, 0, 0, 0],
    [0, 0, 0, 1e5, 0, 0],
    [0, 0, 0, 0, 1e5, 0],
    [0, 0, 0, 0, 0, 1e5],
    ])                              # process noise covariance matrix
map_R = np.array([
    [map_R_linPos, 0, 0, 0, 0, 0],
	[0, map_R_linPos, 0, 0, 0, 0],
	[0, 0, map_R_angPos, 0, 0, 0],
	[0, 0, 0, map_R_linVel, 0, 0],
	[0, 0, 0, 0, map_R_linVel, 0],
	[0, 0, 0, 0, 0, map_R_angVel]
    ])                              # measurement noise covariance

#  Simulation parameter
# INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
# GPS_NOISE = np.diag([0.5, 0.5]) ** 2
# INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0), 1.0, 1.0, 1.0, 1.0]) ** 2
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0), 1.0]) ** 2
GPS_NOISE = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) ** 2

show_animation = True

def calc_input():
    # Control input vector
    u = np.array([[1], [1], [0.1]])   # [surge, sway, yaw]
    return u

def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(6, 1)

    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(3, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud


def motion_model(x, u):
    F = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])    # discrete state matrix

    B = np.zeros((6, 3))    # input matrix

    x = F @ x + B @ u

    return x

def observation_model(x):
    map_H = np.array([							
		[H_linPos, 0, 0, 0, 0, 0],
		[0, H_linPos, 0, 0, 0, 0],
		[0, 0, H_angPos, 0, 0, 0],
		[0, 0, 0, H_linVel, 0, 0],
		[0, 0, 0, 0, H_linVel, 0],
		[0, 0, 0, 0, 0, H_angVel]       # observation matrix
	])	

    z = map_H @ x

    return z


def jacob_f(x, u):
    """
    Jacobian of Motion Model

    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1, 0, -dt * v * math.sin(yaw), dt * math.cos(yaw), 0, 0],
        [0, 1, dt * v * math.cos(yaw), dt * math.sin(yaw), 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
        ])

    return jF


def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
		[1, 0, 0, 0, 0, 0],
		[0, 1, 0, 0, 0, 0],
		[0, 0, 1, 0, 0, 0],
		[0, 0, 0, 1, 0, 0],
		[0, 0, 0, 0, 1, 0],
		[0, 0, 0, 0, 0, 1]  
    ])

    return jH


def ekf_estimation(xEst, PEst, z, u):
    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + map_Q

    #  Update
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + map_R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst


def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    rot = np.array([[math.cos(angle), math.sin(angle)],
                    [-math.sin(angle), math.cos(angle)]])
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")


def main():
    print(__file__ + " start!!")

    time = 0.0

    # State Vector [x y yaw v]'
    xEst = np.zeros((6, 1))
    xTrue = np.zeros((6, 1))
    PEst = np.eye(6)

    xDR = np.zeros((6, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    for i in range (0, endTime):
        time += dt
        u = calc_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = z

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g", label = 'positioning by GPS')
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b", label = 'truth')
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k", label = 'dead reckoning')
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r", label = 'estimated')
            plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.pause(0.001)

if __name__ == '__main__':
    main()
