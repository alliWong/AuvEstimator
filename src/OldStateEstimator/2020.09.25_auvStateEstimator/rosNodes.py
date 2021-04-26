#! /usr/bin/env python

"""
TASKS:
1) Test in UUV simulator with ros nodes
"""

import rospy
from numpy.linalg import inv
from numpy import identity
from numpy import matrix, diag, random, array, zeros, block, mod, reshape
from math import sqrt, cos, sin, atan2, log
import math
import matplotlib.pyplot as plt
import numpy as np
import genPlots
from kf import KalmanFilter
from ekf import ExtendedKalmanFilter
from inputs import timeInput, simInputs, senInputs, estInputs
t = timeInput()
sim = simInputs()
sen = senInputs()
est = estInputs()

