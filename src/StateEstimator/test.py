#!/usr/bin/env python

import numpy as np
from commons import PressureToDepth
import gtsam
import gtsam_example

g = gtsam_example.Greeting()
g.sayHello()

# ################################################################
# Rot = np.array([[-0.835035, 0.501546, -0.226206],
#                 [-0.0851015, -0.523926, -0.847502],
#                 [-0.543576, -0.688443, 0.480179]
#                 ])



# v1 = 7.93416e+06
# v2 = -899729
# v3 = 2.24528e+06

# n_vel = np.array([[v1],
#                 [v2],
#                 [v3]])

# b_vel = np.matmul(Rot.T, n_vel)

# # b_vel_vec = np.array([[0.470324],
# #                 [-0.210995],
# #                 [0.962962]]) 

# print('b_vel\n', b_vel)
# # print('err\n', b_vel-b_vel_vec)

#################################################################


# Rot = np.array([[0.0577141, -0.998328, 0.00316392],
#                 [0.997996,  0.0577766,  0.0257905],
#                 [-0.0259302,  0.0016691,   0.999662]])

# b_vel = np.array([[-0.82252034],
#                 [-0.06793869],
#                 [0.66993921]])

# n_vel = np.matmul(Rot, b_vel)

# print('n_vel\n', n_vel)
# print('err\n', b_vel-b_vel_vec)


