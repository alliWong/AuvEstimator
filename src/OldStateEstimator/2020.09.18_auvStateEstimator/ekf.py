#! /usr/bin/env python

import numpy as np
from numpy.linalg import inv


class ExtendedKalmanFilter:
    """
    Extended Kalman filter
    """
    def __init__(self, x, P, z, u, A, B, Q, R, H):
        self.x = x  # estimator state estimate
        self.P = P  # covariance matrx
        self.z = z  # measurement matrix
        self.u = u  # control input m atrix
        self.A = A  # state matrix
        self.B = B  # input matrix
        self.Q = Q  # process noise covariance matrix
        self.R = R  # measurement noise covariance matrix
        self.H = H  # jacobian of measurement matrix wrt state
    
    # F (A) is different
    def predict(self, x, P, u):
        # Prediction for state vector and covariance
        x = self.A * x + self.B * u # x = f(x, u)
        P = self.A * P * (self.A.T) + self.Q    # same
        return x, P

    # H is different
    def update(self, x, P, z):
        # Compute Kalman Gain
        K = P * (self.H.T) * inv(self.H * P * (self.H.T) + self.R)  # same

        # Correction based on observation
        x = x + K * (z - self.H * x)
        P = P - K * self.H * P # same
        return x, K, P

