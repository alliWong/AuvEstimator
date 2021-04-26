import numpy as np
import matplotlib.pyplot as plt

a = np.matrix([
    [1, 2, 3, 4, 5], 
    [2, 2, 3, 4, 5], 
    [3, 2, 3, 4, 5], 
    [4, 2, 3, 4, 5], 
    [5, 2, 3, 4, 5] 
])

print(a[:, -1])
# print(a.shape)
# est_map_x = np.matrix(np.zeros((6,2000)))              # map frame estimator state estimate

# print(est_map_x.shape)

# a = 0

# for a in range (0, 10, 1):
#     if np.mod(a, 1) == 0:
#         a = a+1
#         b = a
#         print(b)


# endTime = 50
# dt = 0.001
# startTime = 0
# t_now = startTime
# t_N = round(endTime/dt - startTime/dt)
# est_dt = 0.025
# i = 1

# for k in range (0, t_N-1, 1):
#     if np.round(np.mod(k*dt, est_dt), decimals=2) == 0:
#         print(np.mod(k*dt, est_dt))
#     # print(k)
    # if (k*dt/est_dt) == 0:
    # if np.mod(k*dt, est_dt) == 0:

        # print(q)
        # print(k)
        # print(i)

# # Example data
# t = np.arange(0.0, 1.0 + 0.01, 0.01)
# s = np.cos(4 * np.pi * t) + 2

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.plot(t, s)

# plt.xlabel(r'\textbf{time} (s)')
# plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
# # plt.title(r"\TeX\ is Number "
# #           r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
# #           fontsize=16, color='gray')
# plt.title(r'\textit{voltage} (mV)',fontsize=16)
# # Make room for the ridiculously large title.
# plt.subplots_adjust(top=0.8)

# plt.savefig('tex_demo')
# plt.show()



# import numpy as np
# from numpy import array, block, zeros, matrix, identity, reshape
# import math

# ################# Simulation user input ##################

# # Time properties
# dt = 0.001      # simulation time step [s]
# startTime = 0
# endTime = 1    # simulation end time [s]
# t_N = round(endTime/dt) # total number of time steps [s]
# t_now = startTime

# # USV lumped parameters
# sim_m = 225
# sim_I = 100
# sim_bxr = 40
# sim_byr = 400
# sim_bpr = 300

# # Sensor setup
# gnss_rr = 1      # refresh rate of GPS sensor [Hz]
# gnss_linPosVar = 1
# imu_rr = 20     # refresh rate of IMU sensor [Hz]
# imu_angPosVar = 0.05
# imu_angVelVar = 0.05
# imu_linAccVar = 0.1

# ################## Simulation Setup ##################
# # State, output, and input vectors
# sim_xr = zeros((6, t_N))
# sim_yr = zeros((9, t_N))
# sim_xm = zeros((3, t_N))
# sim_u = zeros((3, t_N))

# # Convect into matrix
# sim_xr = np.matrix(sim_xr)

# ################## Create Plant ##################
# sim_A = matrix([
#     [0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 1],
#     [0, 0, 0, -sim_bxr/sim_m, 0, 0],
#     [0, 0, 0, 0, -sim_byr/sim_m, 0],
#     [0, 0, 0, 0, 0, -sim_bpr/sim_I]         
# ])

# sim_B = matrix([
#     [0, 0, 0],
#     [0, 0, 0],
#     [0, 0, 0],
#     [1/sim_m, 0, 0],
#     [0, 1/sim_m, 0],
#     [0, 0, 1/sim_I]
#     ])

# sim_C = block([
#     [1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     [sim_A[3:6, :], zeros((3,3))]
# ])

# sim_D = block([
#     [0, 0, 0],
#     [0, 0, 0],
#     [0, 0, 0],
#     [0, 0, 0],
#     [0, 0, 0],
#     [0, 0, 0],  
#     [sim_B[3:6, :]] 
# ])

# F = identity(6)+sim_A*dt
# G = sim_B*dt
# H = sim_C
# J = sim_D
# k = 1

# for k in range (0, t_N):
#     if k == 1:   
#         sim_xr[:, k+1] = F*sim_xr[:, k]+G*matrix([[sim_u[0, k]], [sim_u[1, k]], [sim_u[2, k]]])
#         sim_yr[:, k+1] = H*block([[sim_xr[:, k+1]], [zeros((3,1))]])+J*matrix([[sim_u[0, k]], [sim_u[1, k]], [sim_u[2, k]]])

#         print('F:', F)
#         print(F.shape)
#         print('sim xr:', reshape(sim_xr[:, k], (6, 1)))
#         print(sim_xr[:, k].shape)
#         test = F*reshape(sim_xr[:, k], (6, 1))
#         print(test.shape)

#         # print(G.shape)
#         # stuff = array([[sim_u[0, k]], [sim_u[1, k]], [sim_u[2, k]]])
#         # print(stuff)
#         # print(stuff.shape)
#         # test2 = G*matrix([[sim_u[0, k]], [sim_u[1, k]], [sim_u[2, k]]])
#         # test3 = test + test2
#         # print(test3.shape)
#         # print(test3)

#         # sim_xr[:, k+1] = test+test2
#         # print(sim_xr[:, k+1])

#         # print(H)
#         # print(H.shape)
#         # test = block([
#         #     [sim_xr[:, k+1]],
#         #     [zeros((3,1))]
#         #     ])

#         # print(test.shape)
#         # print(test)
