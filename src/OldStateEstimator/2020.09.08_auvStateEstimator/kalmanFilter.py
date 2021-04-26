#! /usr/bin/env python

import numpy as np
from numpy.linalg import inv


class KalmanFilter:
    """
    Simple Kalman filter

    Control term has been omitted for now
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
        self.H = H  # observation matrix


    def predict(self, x, P, u):
        # Prediction for state vector and covariance
        x = self.A * x + self.B * u
        P = self.A * P * (self.A.T) + self.Q
        return x, P

    def update(self, x, P, z):
        # Compute Kalman Gain
        K = P * (self.H.T) * inv(self.H * P * (self.H.T) + self.R)

        # Correction based on observation
        x += K * (z - self.H * x)
        P = P - K * self.H * P
        return x, K, P