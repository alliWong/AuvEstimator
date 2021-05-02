# ! /usr/bin/env python
from sympy import symbols, sin, cos, diff, subs
from sympy import *
import numpy as np
from numpy import array, vectorize
import math
# from commons import frameTrans

def dRot(input_phi, input_theta, input_psi, input_dphi, input_dtheta, input_dpsi):
    # Create symbolic variables
    x, y, z, phi, theta, psi, u, v, w = symbols("x, y, z, phi, theta, psi, u, v, w ") 

    # Rotation matrices
    Rx = lambda t: array([
        [1,       0,       0],
        [0,  cos(t), -sin(t)],
        [0,  sin(t),   cos(t)]]) # rotation about x axis
    Ry = lambda t: array([
        [ cos(t),  0,  sin(t)],
        [      0,  1,       0],
        [-sin(t),  0,  cos(t)]]) # rotation about y axis
    Rz = lambda t: array([
        [cos(t),  -sin(t), 0],
        [sin(t),   cos(t), 0],
        [     0,        0, 1]]) # rotation about z axis

    # Three dimensional rotation matrix
    RFxyz = np.dot(Rz(psi), Ry(theta)).dot(Rx(phi)) # fixed xyz

    # Transformation matrix for angular velocity related by the Jacobian
    J = array([
        [1, sin(phi)*tan(theta), cos(phi)*tan(theta)],
        [0, cos(phi), -sin(phi)],
        [0, sin(phi)/cos(theta), cos(phi)/cos(theta)]
    ])

    # State transition matrix
    A = np.array([
        [x, 0, 0, u*RFxyz[0,0], v*RFxyz[0, 1], w*RFxyz[0, 2], u*RFxyz[0,0], v*RFxyz[0, 1], w*RFxyz[0, 2], 0, 0, 0, 0, 0, 0], # x
        [0, y, 0, u*RFxyz[1,0], v*RFxyz[1, 1], w*RFxyz[1, 2], u*RFxyz[1,0], v*RFxyz[1, 1], w*RFxyz[1, 2], 0, 0, 0, 0, 0, 0], # y
        [0, 0, z, u*RFxyz[2,0], v*RFxyz[2, 1], w*RFxyz[2, 2], u*RFxyz[2,0], v*RFxyz[2, 1], w*RFxyz[2, 2], 0, 0, 0, 0, 0, 0], # z
        [0, 0, 0, phi, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # phi
        [0, 0, 0, 0, theta, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # theta
        [0, 0, 0, 0, 0, psi, 0, 0, 0, 0, 0, 0, 0, 0, 0], # psi
        [0, 0, 0, 0, 0, 0, u, 0, 0, 0, 0, 0, 0, 0, 0], # vx
        [0, 0, 0, 0, 0, 0, 0, v, 0, 0, 0, 0, 0, 0, 0], # vy
        [0, 0, 0, 0, 0, 0, 0, 0, w, 0, 0, 0, 0, 0, 0], # vz
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # bax
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # bay
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # baz
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # bgx
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # bgy
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # bgz
    ])

    func = x
    dx = diff(func,x)
    # Derivative of the three dimensional rotation matrix
    dA = np.array([
        [dx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # phi
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # phi
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # phi
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # phi
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # theta
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # psi
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # vx
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # vy
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # vz
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # bax
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # bay
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # baz
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # bgx
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # bgy
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # bgz
    ])

    Rot = np.vectorize(lambda t: t.subs({phi:input_phi, theta:input_theta, psi:input_psi}))(RFxyz)
    # print('Rot \n', Rot)
    dRot = dA.subs({phi:input_phi, theta:input_theta, psi:input_psi, dphi:input_dphi, dtheta:input_dtheta, dpsi:input_dpsi})
    # print('dRot \n', dRot)

    return dRot

phi, theta, psi = symbols("phi, theta, psi") 

# Using function
phi, theta, psi, dphi, dtheta, dpsi = phi, 0, 0, 0, 0, 0
testFunc = dRot(phi, theta, psi, dphi, dtheta, dpsi)
print('testF', testFunc)






################## TEST NON FUNCTION #############################
# # Create symbolic variables
# phi, theta, psi = symbols("phi, theta, psi") 
# dphi, dtheta, dpsi = symbols("dphi, dtheta, dpsi") 
# x, y, z, x_pre, y_pre, z_pre = symbols("x, y, z, x_pre, y_pre, z_pre") 
# dx_pre, dy_pre, dz_pre = symbols("dx_pre, dy_pre, dz_pre")
# dt = symbols("dt")


# # Rotation matrices
# Rx = lambda t: array([
#     [1,       0,       0],
#     [0,  cos(t), -sin(t)],
#     [0,  sin(t),   cos(t)]]) # rotation about x axis
# Ry = lambda t: array([
#     [ cos(t),  0,  sin(t)],
#     [      0,  1,       0],
#     [-sin(t),  0,  cos(t)]]) # rotation about y axis
# Rz = lambda t: array([
#     [cos(t),  -sin(t), 0],
#     [sin(t),   cos(t), 0],
#     [     0,        0, 1]]) # rotation about z axis

# # Three dimensional rotation matrix
# RFxyz = np.dot(Rz(psi), Ry(theta)).dot(Rx(phi)) # fixed xyz

# # Derivative of the three dimensional rotation matrix
# dRFxyz = diff(RFxyz,phi)*dphi+diff(RFxyz,theta)*dtheta+diff(RFxyz,psi)*dpsi

# # Print expressions
# # print('RFxyz \n', RFxyz)
# # print('dRFxyz \n', dRFxyz)

# # Print equations
# # Rot = np.vectorize(lambda t: t.subs({phi:0, theta:theta, psi:0}))(RFxyz)
# # dRot = dRFxyz.subs({phi:0, theta: theta, psi:0, dphi:0, dtheta:0, dpsi:0})
# # print('Rot \n', Rot)
# # print('dRot \n', dRot)

# pos = np.array([
#     [x],
#     [y],
#     [z]
# ])
# pos_pre = np.array([
#     [x_pre],
#     [y_pre],
#     [z_pre]
# ])
# vel_pre = np.array([
#     [dx_pre],
#     [dy_pre],
#     [dz_pre]
# ])
# pos = (pos_pre + vel_pre*RFxyz*dt)

# print('pos', pos[2])



################## TEST FUNCTION #############################
# def dRot(input_phi, input_theta, input_psi, input_dphi, input_dtheta, input_dpsi):
#     # Create symbolic variables
#     phi, theta, psi = symbols("phi, theta, psi") 
#     dphi, dtheta, dpsi = symbols("dphi, dtheta, dpsi") 

#     # Rotation matrices
#     Rx = lambda t: array([
#         [1,       0,       0],
#         [0,  cos(t), -sin(t)],
#         [0,  sin(t),   cos(t)]]) # rotation about x axis
#     Ry = lambda t: array([
#         [ cos(t),  0,  sin(t)],
#         [      0,  1,       0],
#         [-sin(t),  0,  cos(t)]]) # rotation about y axis
#     Rz = lambda t: array([
#         [cos(t),  -sin(t), 0],
#         [sin(t),   cos(t), 0],
#         [     0,        0, 1]]) # rotation about z axis

#     # Three dimensional rotation matrix
#     RFxyz = np.dot(Rz(psi), Ry(theta)).dot(Rx(phi)) # fixed xyz

#     # Derivative of the three dimensional rotation matrix
#     dRFxyz = diff(RFxyz,phi)*dphi+diff(RFxyz,theta)*dtheta+diff(RFxyz,psi)*dpsi

#     # Method 1) Calculate map velocity using time derivative Rotation matrix
#     Rot = np.vectorize(lambda t: t.subs({phi:input_phi, theta:input_theta, psi:input_psi}))(RFxyz)
#     # print('Rot \n', Rot)
#     dRot = dRFxyz.subs({phi:input_phi, theta:input_theta, psi:input_psi, dphi:input_dphi, dtheta:input_dtheta, dpsi:input_dpsi})
#     # print('dRot \n', dRot)

#     return dRot

# # Using function
# phi, theta, psi, dphi, dtheta, dpsi = 0, 45, 0, 0, 0, 0
# testFunc = dRot(phi, theta, psi, dphi, dtheta, dpsi)
# print('testF', testFunc)

