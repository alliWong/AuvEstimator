import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, atan2

def rbt2map(xrf,yrf,xr0,yr0,psi0,xm0,ym0):
    # Converts pose in the robot frame to pose in the map frame

    # Calculate translations and rotations in robot frame
    Txr = xrf-xr0
    Tyr = yrf-yr0

    # Calculate intermediate length and angle
    li = sqrt(Txr**2+Tyr**2)
    psii = atan2(yrf-yr0, xrf-xr0)  # atan or atan2

    # Calculate translations and rotations in map frame
    Txm = cos(psii+psi0)*li
    Tym = sin(psii+psi0)*li

    # Calculate individual components in the map frame
    xmf = xm0+Txm
    ymf = ym0+Tym

    # print('Txr', Txr)
    # print('Tyr', Tyr)
    # print('li', li)
    # print('psii', psii)
    # print('Txm', Txm)
    # print('Tym', Tym)

    return xmf, ymf

[xmf, ymf] = rbt2map(
    1,
    1,
    0,
    0,
    90,
    0,
    0
)

print(1e1)

img = np.array([[1, 2], [3, 4]])
stacked_img = np.stack((img,)*3, axis=-1)
print(stacked_img)
 # array([[[1, 1, 1],
 #         [2, 2, 2]],
 #        [[3, 3, 3],
 #         [4, 4, 4]]])
