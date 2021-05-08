#! /usr/bin/env python
import numpy as np
from numpy import matmul, eye, dot
from numpy.linalg import inv

class ExtendedKalmanFilter:
    def __init__(self, xDim, x, P, z, u, A, Q, R, H):
        self.xDim = xDim
        self.x = x  # estimator state estimate
        self.P = P  # covariance matrx
        self.z = z  # measurement matrix
        self.u = u  # control input matrix
        self.A = A  # jacobian of state matrix
        self.Q = Q  # process noise covariance matrix
        self.R = R  # measurement noise covariance matrix
        self.H = H  # jacobian of measurement matrix wrt state
        self.I = eye(self.xDim) # identity matrix

    def predict(self, P):
        # Prediction for state vector and covariance
        P = self.A.dot(P).dot(self.A.T) + self.Q 
        return P

    def update(self, x, P, z):
        # Compute Kalman Gain
        K = P.dot(self.H.T).dot(np.linalg.inv(self.R + self.H.dot(P).dot(self.H.T)))

        # Correction based on observation
        y = z - dot(self.H, x)
        x = x + dot(K, y)
        P = dot(self.I - dot(K, self.H), P)
        return x, K, P
