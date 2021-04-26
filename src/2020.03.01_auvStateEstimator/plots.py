#!/usr/bin/python
""" Plot input and output """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from mpl_toolkits import mplot3d
import os
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D

""" Plot properties """
titleFontSize = 30
defaultFontSize = 30
markerSize = 3
legendSize = 20
lineWidth = 2
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes', labelsize=defaultFontSize) # fontsize of the x and y labels
plt.rc('xtick', labelsize=defaultFontSize) # fontsize of the tick labels
plt.rc('ytick', labelsize=defaultFontSize) # fontsize of the tick labels

def plot_results(data, save_dir, use_gt, use_dr, use_est, use_depth, use_imu):
	"""
	Plotting estimators, ground truth, and dead reckoning trajectories
	"""
	""" Input is a dict containing the input and output """
	# convert list to np array for easier operation
	data_np = {}
	for meas_type, samples in data.items():
		if samples:
			data_np[meas_type] = np.asarray(samples)
	if not data_np:
		print("Nothing to plot..")
		return
	# find min time
	min_time = None
	for meas_type, sample_array in data_np.items():
		if not min_time or sample_array[0, 0] < min_time:
			min_time = sample_array[0, 0]

	""" Map XY """
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for meas_type, sample_array in data_np.items():
		if (meas_type == "ground truth" and use_gt):
			plt.plot(sample_array[:, 1], sample_array[:, 2],  sample_array[:, 3], 'o', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'k', label='Ground truth')
		elif (meas_type == "DR" and use_dr):
			plt.plot(sample_array[:, 1], sample_array[:, 2],  sample_array[:, 3], 'o', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'r', label='Dead reckoning')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 1], sample_array[:, 2],  sample_array[:, 3], 'o', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'g', label='Extended Kalman filter')
	trialTitleString = r'Trajectory 1 \textit{3D} Position'
	plt.title(trialTitleString,fontsize = titleFontSize)
	ax.set_xlabel(r'$x_m$ [m]', labelpad=30)
	ax.set_ylabel(r'$y_m$ [m]', labelpad=30)
	ax.set_zlabel(r'$z_m$ [m]', labelpad=30)
	# plt.legend(prop={'size': legendSize})
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
	ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
	ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None]) # press esc key to exit figure
	# Save plot to directory
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'xy.png'))

	""" Map Linear Position """
	### X ###
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for meas_type, sample_array in data_np.items():
		if (meas_type == "ground truth" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 1], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'k', label='Ground truth')
		elif (meas_type == "DR" and use_dr):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 1], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'r', label='Dead reckoning')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 1], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'g', label='Extended Kalman filter')
	trialTitleString = r'Map \textit{$x_m$} position'
	plt.title(trialTitleString,fontsize = titleFontSize)
	plt.xlabel(r'time [s]')
	plt.ylabel(r'$x_m$ [m]')
	# plt.legend(prop={'size': legendSize})
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	plt.grid(False)
	# # Set figure size in pixels
	# fig = plt.gcf()
	# DPI = fig.get_dpi()
	# fig.set_size_inches(1920.0/float(DPI),1280.0/float(DPI))
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	# Save plot to directory
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'X.png'))

	### Y ###
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for meas_type, sample_array in data_np.items():
		if (meas_type == "ground truth" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 2], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'k', label='Ground truth')
		elif (meas_type == "DR" and use_dr):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 2], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'r', label='Dead reckoning')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 2], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'g', label='Extended Kalman filter')
	trialTitleString = r'Map \textit{$y_m$} position'
	plt.title(trialTitleString,fontsize = titleFontSize)
	plt.xlabel(r'time [s]')
	plt.ylabel(r'$y_m$ [m]')
	# plt.legend(prop={'size': legendSize})
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	plt.grid(False)
	# # Set figure size in pixels
	# fig = plt.gcf()
	# DPI = fig.get_dpi()
	# fig.set_size_inches(1920.0/float(DPI),1280.0/float(DPI))
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None]) # press esc key to exit figure
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'Y.png')) # Save plot to directory

	### Z ###
	fig = plt.figure()
	ax = fig.add_subplot(111)
	legends = []
	for meas_type, sample_array in data_np.items():
		# if (meas_type == "ground truth" and use_gt):
		# 	plt.plot(sample_array[:, 0] - min_time, sample_array[:, 3], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'k', label='Ground truth')
		if (meas_type == "DR" and use_dr):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 3], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'r', label='measurements')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 3], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'g', label='Extended Kalman filter')
	trialTitleString = r'Map \textit{$z_m$} position'
	plt.title(trialTitleString,fontsize = titleFontSize)
	plt.xlabel(r'time [s]')
	plt.ylabel(r'$z_m$ [m]')
	# plt.legend(prop={'size': legendSize})
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	plt.grid(False)
	# # Set figure size in pixels
	# fig = plt.gcf()
	# DPI = fig.get_dpi()
	# fig.set_size_inches(1920.0/float(DPI),1280.0/float(DPI))
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	# Save plot to directory
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'Z.png'))

	""" Map Angular Position """
	### Roll ###
	fig = plt.figure()
	ax = fig.add_subplot(111)
	legends = []
	for meas_type, sample_array in data_np.items():
		if (meas_type == "ground truth" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 4], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'k', label='Ground truth')
		elif (meas_type == "measurements (IMU)" and use_imu):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 1], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'r',  label='measurements')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 4], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'g', label='Extended Kalman filter')
	trialTitleString = r'Roll'
	plt.title(trialTitleString,fontsize = titleFontSize)
	plt.xlabel(r'time [s]')
	plt.ylabel(r'Roll [degrees]')
	# plt.legend(prop={'size': legendSize})
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	plt.grid(False)
	# # Set figure size in pixels
	# fig = plt.gcf()
	# DPI = fig.get_dpi()
	# fig.set_size_inches(1920.0/float(DPI),1280.0/float(DPI))
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	# Save plot to directory
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'roll.png'))

	### Pitch ###
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for meas_type, sample_array in data_np.items():
		if (meas_type == "ground truth" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 5], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'k', label='Ground truth')
		elif (meas_type == "measurements (IMU)" and use_imu):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 2], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'r', label='measurements')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 5], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'g', label='Extended Kalman filter')
	trialTitleString = r'Pitch'
	plt.title(trialTitleString,fontsize = titleFontSize)
	plt.xlabel(r'time [s]')
	plt.ylabel(r'Pitch [degrees]')
	# plt.legend(prop={'size': legendSize})
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	plt.grid(False)
	# # Set figure size in pixels
	# fig = plt.gcf()
	# DPI = fig.get_dpi()
	# fig.set_size_inches(1920.0/float(DPI),1280.0/float(DPI))
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	# Save plot to directory
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'pitch.png')) # Save plot to directory

	### Yaw ###
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for meas_type, sample_array in data_np.items():
		if (meas_type == "ground truth" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 6], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'k', label='Ground truth')
		elif (meas_type == "measurements (IMU)" and use_imu):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 3], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'r', label='measurements')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 6], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'g', label='Extended Kalman filter')
	trialTitleString = r'Yaw'
	plt.title(trialTitleString,fontsize = titleFontSize)
	plt.xlabel(r'time [s]')
	plt.ylabel(r'Yaw [degrees]')
	# plt.legend(prop={'size': legendSize})
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	plt.grid(False)
	# # Set figure size in pixels
	# fig = plt.gcf()
	# DPI = fig.get_dpi()
	# fig.set_size_inches(1920.0/float(DPI),1280.0/float(DPI))
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	# Save plot to directory
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'yaw.png')) # Save plot to directory

	""" Linear Velocity """
	### X Linear Velocity ###
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for meas_type, sample_array in data_np.items():
		if (meas_type == "ground truth" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 7], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'k', label='Ground truth')
		elif (meas_type == "DR" and use_dr):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 7], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'r', label='measurements')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 7], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'g', label='Extended Kalman filter')
	trialTitleString = r'Map \textit{$\dot{x}_m$} velocity'
	plt.title(trialTitleString,fontsize = titleFontSize)
	plt.xlabel(r'time [s]')
	plt.ylabel(r'$\dot{x}_m$ [m/s]')
	# plt.legend(prop={'size': legendSize})
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	plt.grid(False)
	# Set figure size in pixels
	# fig = plt.gcf()
	# DPI = fig.get_dpi()
	# fig.set_size_inches(1920.0/float(DPI),1280.0/float(DPI))
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	# Save plot to directory
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'vx.png'))

	### Y Linear Velocity ###
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for meas_type, sample_array in data_np.items():
		if (meas_type == "ground truth" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 8], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'k', label='Ground truth')
		elif (meas_type == "DR" and use_dr):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 8], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'r', label='measurements')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 8], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'g', label='Extended Kalman filter')
	trialTitleString = r'Map \textit{$\dot{y}_m$} velocity'
	plt.title(trialTitleString,fontsize = titleFontSize)
	plt.xlabel(r'time [s]')
	plt.ylabel(r'$\dot{y}_m$ [m/s]')
	# plt.legend(prop={'size': legendSize})
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	plt.grid(False)
	# Set figure size in pixels
	# fig = plt.gcf()
	# DPI = fig.get_dpi()
	# fig.set_size_inches(1920.0/float(DPI),1280.0/float(DPI))
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	# Save plot to directory
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'vy.png'))

	### Z Linear Velocity ##
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for meas_type, sample_array in data_np.items():
		if (meas_type == "ground truth" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 9], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'k', label='Ground truth')
		elif (meas_type == "DR" and use_dr):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 9], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'r', label='measurements')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 9], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'g', label='Extended Kalman filter')
	trialTitleString = r'Map \textit{$\dot{z}_m$} velocity'
	plt.title(trialTitleString,fontsize = titleFontSize)
	plt.xlabel(r'time [s]')
	plt.ylabel(r'$\dot{z}_m$ [m/s]')
	# plt.legend(prop={'size': legendSize})
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	plt.grid(False)
	# Set figure size in pixels
	# fig = plt.gcf()
	# DPI = fig.get_dpi()
	# fig.set_size_inches(1920.0/float(DPI),1280.0/float(DPI))
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	# Save plot to directory
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'vz.png'))

	""" Show plots """
	plt.show()

########################################################################################################################
def plot_gtsam(data, fusion_items, save_dir, use_gt, use_bar, use_dvl):
	"""
	Plotting isam2
	"""
	""" Input is a dict containing the input and output in IMU frame """
	# convert list to np array for easier operation
	data_np = {}
	for meas_type, samples in data.items():
		if samples:
			data_np[meas_type] = np.asarray(samples)
	if not data_np:
		print("Nothing to plot..")
		return
	# find min time
	min_time = None
	for meas_type, sample_array in data_np.items():
		if not min_time or sample_array[0, 0] < min_time:
			min_time = sample_array[0, 0]

	""" Map XY """
	plt.figure(figsize=(8, 8))
	for meas_type, sample_array in data_np.items():
		if meas_type == fusion_items:
			plt.plot(sample_array[:, 1], sample_array[:, 2], '.', markersize=markerSize, color = 'g', label='Estimator')
		elif (meas_type == "GT" and use_gt):
			plt.plot(sample_array[:, 1], sample_array[:, 2], 'o', markersize=markerSize, color = 'k', label='Ground truth')
	plt.xlabel('x [m]')
	plt.ylabel('y [m]')
	plt.title('Map XY')
	plt.legend()
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
	plt.grid(True)
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'xy.png'))

	""" Linear Position """
	# X vs Time
	fig = plt.figure(figsize=(15,15))
	ax1 = fig.add_subplot(221)
	for meas_type, sample_array in data_np.items():
		if meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 1], '.', markersize=markerSize, color = 'g', label='Estimator')
		elif (meas_type == "GT" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 1], '.', markersize=markerSize, color = 'k', label='Ground truth')
	plt.xlabel('Time [s]')
	plt.ylabel('X [m]')
	plt.title('Map X vs Time')
	plt.legend()
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
	plt.grid(True)
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'x.png'))
	# Y vs Time
	fig.add_subplot(222)
	for meas_type, sample_array in data_np.items():
		if meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 2], '.', markersize=markerSize, color = 'g', label='Estimator')
		elif (meas_type == "GT" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 2], '.', markersize=markerSize, color = 'k', label='Ground truth')
	plt.xlabel('Time [s]')
	plt.ylabel('Y [m]')
	plt.title('Map Y vs Time')
	plt.legend()
	plt.tight_layout(pad=1, w_pad=1, h_pad=5.0)
	plt.grid(True)
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'y.png'))
	# Z
	fig.add_subplot(223)
	for meas_type, sample_array in data_np.items():
		if meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 3], '.', markersize=markerSize, color = 'g', label='Estimator')
		elif (meas_type == "GT" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 3], '.', markersize=markerSize, color = 'k', label='Ground truth')
		elif meas_type == "BAR" and use_bar:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 1], '.', markersize=markerSize, color = 'r', label='measurements')
	plt.xlabel('Time [s]')
	plt.ylabel('Z [m]')
	plt.title('Map Z vs Time')
	plt.legend()
	plt.tight_layout(pad=1, w_pad=1, h_pad=5.0)
	plt.grid(True)
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'z.png'))

	""" Angular Position """
	# Roll vs Time
	fig = plt.figure(figsize=(15,15))
	ax1 = fig.add_subplot(221)
	for meas_type, sample_array in data_np.items():
		if meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 4], '-', markersize=markerSize, color = 'g', label='Estimator')
		elif meas_type == "GT" and use_gt:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 4], '-', markersize=markerSize, color = 'k', label='Ground truth')
		elif meas_type == "IMU":
			roll_vehicle = 180 * np.arctan2(sample_array[:, 3], np.sqrt(np.power(sample_array[:, 2], 2) + np.power(sample_array[:, 4], 2))) / np.pi
			plt.plot(sample_array[:, 0] - min_time, roll_vehicle, '-', markersize=markerSize, color='r', label='measurements')
	plt.xlabel('Time [s]')
	plt.ylabel('Roll [deg]')
	plt.title('Roll vs Time')
	plt.legend()
	plt.grid(True)
	plt.tight_layout(pad=1, w_pad=1, h_pad=5.0)
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'roll.png'))
	# Pitch vs Time
	fig.add_subplot(222)
	for meas_type, sample_array in data_np.items():
		if meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 5], '-', markersize=markerSize, color = 'g', label='Estimator')
		elif meas_type == "GT" and use_gt:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 5], '-', markersize=markerSize, color = 'k', label='Ground truth')
		elif meas_type == "IMU":
			pitch_vehicle = 180 * np.arctan2(-sample_array[:, 2], np.sqrt(np.power(sample_array[:, 3], 2) + np.power(sample_array[:, 4], 2))) / np.pi
			plt.plot(sample_array[:, 0] - min_time, pitch_vehicle, '-', markersize=markerSize, color='r', label='measurements')
	plt.xlabel('Time [s]')
	plt.ylabel('Pitch [deg]')
	plt.title('Pitch vs Time')
	plt.legend()
	plt.grid(True)
	plt.tight_layout(pad=1, w_pad=1, h_pad=5.0)
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'pitch.png'))
	# Yaw vs Time
	fig.add_subplot(223)
	for meas_type, sample_array in data_np.items():
		if meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 6] + 180.0, '-', markersize=markerSize, color = 'g', label='Estimator')
		elif meas_type == "GT" and use_gt:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 6] + 180.0, '-', markersize=markerSize,  color = 'k', label='Ground truth')
	plt.xlabel('Time [s]')
	plt.ylabel('Yaw [deg]')
	plt.title('Yaw vs Time')
	plt.legend()
	plt.grid(True)
	plt.tight_layout(pad=1, w_pad=1, h_pad=5.0)
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'Yaw.png'))
	# Show plots
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None]) # press esc key to exit figure

	""" Linear Velocity """
	# X Linear Velocity vs Time
	fig = plt.figure()
	ax = fig.add_subplot(221)
	for meas_type, sample_array in data_np.items():
		if (meas_type == "GT" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 7], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'k', label='Ground truth')
		elif meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 7], '-', markersize=markerSize, color = 'g', label='Estimator')
		elif (meas_type == "DVL" and use_dvl):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 1], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'r', label='measurements')
	trialTitleString = r'Map \textit{$\dot{x}_m$} velocity'
	plt.title(trialTitleString,fontsize = titleFontSize)
	plt.xlabel(r'Time [s]')
	plt.ylabel(r'$\dot{x}_m$ [m/s]')
	plt.legend()
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	plt.grid(False)
	# Save plot to directory
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'vx.png'))
	# Y Linear Velocity vs Time
	ax = fig.add_subplot(222)
	for meas_type, sample_array in data_np.items():
		if (meas_type == "GT" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 8], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'k', label='Ground truth')
		elif meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 8], '-', markersize=markerSize, color = 'g', label='Estimator')
		elif (meas_type == "DVL" and use_dvl):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 2], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'r', label='measurements')
	trialTitleString = r'Map \textit{$\dot{y}_m$} velocity'
	plt.title(trialTitleString,fontsize = titleFontSize)
	plt.xlabel(r'Time [s]')
	plt.ylabel(r'$\dot{y}_m$ [m/s]')
	plt.legend()
	# plt.legend(prop={'size': legendSize})
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	plt.grid(False)
	# Save plot to directory
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'vy.png'))
	# Z Linear Velocity vs Time
	ax = fig.add_subplot(223)
	for meas_type, sample_array in data_np.items():
		if (meas_type == "GT" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 9], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'k', label='Ground truth')
		elif meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 9], '-', markersize=markerSize, color = 'g', label='Estimator')
		elif (meas_type == "DVL" and use_dvl):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 3], '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'r', label='measurements')
	trialTitleString = r'Map \textit{$\dot{z}_m$} velocity'
	plt.title(trialTitleString,fontsize = titleFontSize)
	plt.xlabel(r'Time [s]')
	plt.ylabel(r'$\dot{z}_m$ [m/s]')
	plt.legend()
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	plt.grid(False)
	# Save plot to directory
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'vz.png'))
	# Show plots
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None]) # press esc key to exit figure
	plt.show()

########################################################################################################################
def plot_all(data, fusion_items, save_dir, use_gt, use_dr, use_est, use_bar, use_dvl):
	"""
	Plotting isam2, ekf, dr, and ground truth
	"""
	# convert list to np array for easier operation
	data_np = {}
	for meas_type, samples in data.items():
		if samples:
			data_np[meas_type] = np.asarray(samples)
	if not data_np:
		print("Nothing to plot..")
		return
	# find min time
	min_time = None
	for meas_type, sample_array in data_np.items():
		if not min_time or sample_array[0, 0] < min_time:
			min_time = sample_array[0, 0]

	""" Map XY """
	plt.figure(figsize=(8, 8))
	for meas_type, sample_array in data_np.items():
		if meas_type == fusion_items:
			plt.plot(sample_array[:, 1], sample_array[:, 2], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'g', label='FGO')
		elif (meas_type == "GT" and use_gt):
			plt.plot(sample_array[:, 1], sample_array[:, 2], 'o', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'k', label='GT')
		elif (meas_type == "DR" and use_dr):
			plt.plot(sample_array[:, 1], sample_array[:, 2], 'o', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'r', label='DR')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 1], sample_array[:, 2], 'o', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'm', label='EKF')
	plt.xlabel('x [m]')
	plt.ylabel('y [m]')
	plt.title('Map XY')
	plt.legend()
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
	plt.grid(True)
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'xy.png'))

	""" Linear Position """
	# X vs Time
	fig = plt.figure(figsize=(15,15))
	ax1 = fig.add_subplot(221)
	for meas_type, sample_array in data_np.items():
		if meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 1], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'g', label='FGO')
		elif (meas_type == "GT" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 1], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'k', label='GT')
		elif (meas_type == "DR" and use_dr):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 1], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'r', label='DR')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 1], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'm', label='EKF')
	plt.xlabel('Time [s]')
	plt.ylabel('X [m]')
	plt.title('Map X vs Time')
	plt.legend()
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
	plt.grid(True)
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'x.png'))
	# Y vs Time
	fig.add_subplot(222)
	for meas_type, sample_array in data_np.items():
		if meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 2], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'g', label='FGO')
		elif (meas_type == "GT" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 2], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'k', label='GT')
		elif (meas_type == "DR" and use_dr):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 2], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'r', label='DR')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 2], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'm', label='EKF')
	plt.xlabel('Time [s]')
	plt.ylabel('Y [m]')
	plt.title('Map Y vs Time')
	plt.legend()
	plt.tight_layout(pad=1, w_pad=1, h_pad=5.0)
	plt.grid(True)
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'y.png'))
	# Z
	fig.add_subplot(223)
	for meas_type, sample_array in data_np.items():
		if meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 3], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'g', label='FGO')
		elif (meas_type == "GT" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 3], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'k', label='GT')
		elif meas_type == "BAR" and use_bar:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 1], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'r', label='measurements')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 3], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'm', label='EKF')
	plt.xlabel('Time [s]')
	plt.ylabel('Z [m]')
	plt.title('Map Z vs Time')
	plt.legend()
	plt.tight_layout(pad=1, w_pad=1, h_pad=5.0)
	plt.grid(True)
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'z.png'))

	""" Angular Position """
	# Roll vs Time
	fig = plt.figure(figsize=(15,15))
	ax1 = fig.add_subplot(221)
	for meas_type, sample_array in data_np.items():
		if meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 4], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'g', label='FGO')
		elif meas_type == "GT" and use_gt:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 4], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'k', label='GT')
		elif meas_type == "IMU":
			roll_vehicle = 180 * np.arctan2(sample_array[:, 3], np.sqrt(np.power(sample_array[:, 2], 2) + np.power(sample_array[:, 4], 2))) / np.pi
			plt.plot(sample_array[:, 0] - min_time, roll_vehicle, '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color='r', label='measurements')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 4], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'm', label='EKF')
	plt.xlabel('Time [s]')
	plt.ylabel('Roll [deg]')
	plt.title('Roll vs Time')
	plt.legend()
	plt.grid(True)
	plt.tight_layout(pad=1, w_pad=1, h_pad=5.0)
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'roll.png'))
	# Pitch vs Time
	fig.add_subplot(222)
	for meas_type, sample_array in data_np.items():
		if meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 5], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'g', label='FGO')
		elif meas_type == "GT" and use_gt:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 5], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'k', label='GT')
		elif meas_type == "IMU":
			pitch_vehicle = 180 * np.arctan2(-sample_array[:, 2], np.sqrt(np.power(sample_array[:, 3], 2) + np.power(sample_array[:, 4], 2))) / np.pi
			plt.plot(sample_array[:, 0] - min_time, pitch_vehicle, '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color='r', label='measurements')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 5], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'm', label='EKF')
	plt.xlabel('Time [s]')
	plt.ylabel('Pitch [deg]')
	plt.title('Pitch vs Time')
	plt.legend()
	plt.grid(True)
	plt.tight_layout(pad=1, w_pad=1, h_pad=5.0)
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'pitch.png'))
	# Yaw vs Time
	fig.add_subplot(223)
	for meas_type, sample_array in data_np.items():
		if meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 6] + 180.0, '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'g', label='FGO')
		elif meas_type == "GT" and use_gt:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 6] + 180.0, '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'k', label='GT')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 6] + 180.0, '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'm', label='EKF')
	plt.xlabel('Time [s]')
	plt.ylabel('Yaw [deg]')
	plt.title('Yaw vs Time')
	plt.legend()
	plt.grid(True)
	plt.tight_layout(pad=1, w_pad=1, h_pad=5.0)
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'Yaw.png'))
	# Show plots
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None]) # press esc key to exit figure

	""" Linear Velocity """
	# X Linear Velocity vs Time
	fig = plt.figure()
	ax = fig.add_subplot(221)
	for meas_type, sample_array in data_np.items():
		if (meas_type == "GT" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 7], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'k', label='GT')
		elif meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 7], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'g', label='FGO')
		elif (meas_type == "DVL" and use_dvl):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 1], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'r', label='measurements')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 7], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'm', label='EKF')
	trialTitleString = r'Map \textit{$\dot{x}_m$} velocity'
	plt.title(trialTitleString,fontsize = titleFontSize)
	plt.xlabel(r'Time [s]')
	plt.ylabel(r'$\dot{x}_m$ [m/s]')
	plt.legend()
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	plt.grid(False)
	# Save plot to directory
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'vx.png'))
	# Y Linear Velocity vs Time
	ax = fig.add_subplot(222)
	for meas_type, sample_array in data_np.items():
		if (meas_type == "GT" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 8], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'k', label='GT')
		elif meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 8], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'g', label='FGO')
		elif (meas_type == "DVL" and use_dvl):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 2], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'r', label='measurements')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 8], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'm', label='EKF')
	trialTitleString = r'Map \textit{$\dot{y}_m$} velocity'
	plt.title(trialTitleString,fontsize = titleFontSize)
	plt.xlabel(r'Time [s]')
	plt.ylabel(r'$\dot{y}_m$ [m/s]')
	plt.legend()
	# plt.legend(prop={'size': legendSize})
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	plt.grid(False)
	# Save plot to directory
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'vy.png'))
	# Z Linear Velocity vs Time
	ax = fig.add_subplot(223)
	for meas_type, sample_array in data_np.items():
		if (meas_type == "GT" and use_gt):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 9], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'k', label='GT')
		elif meas_type == fusion_items:
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 9], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'g', label='FGO')
		elif (meas_type == "DVL" and use_dvl):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 3], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'r', label='measurements')
		elif (meas_type == "EKF" and use_est):
			plt.plot(sample_array[:, 0] - min_time, sample_array[:, 9], '.', markersize=markerSize, linestyle='-', linewidth=lineWidth, color = 'm', label='EKF')
	trialTitleString = r'Map \textit{$\dot{z}_m$} velocity'
	plt.title(trialTitleString,fontsize = titleFontSize)
	plt.xlabel(r'Time [s]')
	plt.ylabel(r'$\dot{z}_m$ [m/s]')
	plt.legend()
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	plt.grid(False)
	# Save plot to directory
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'vz.png'))
	# Show plots
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None]) # press esc key to exit figure
	plt.show()

########################################################################################################################
def plot_error(dr_time, ekf_time, absErrorMeanDrMapXY, absErrorMeanEkfMapXY):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(dr_time, absErrorMeanDrMapXY, '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'k', label='Ground truth')
	# plt.plot(ekf_time, absErrorMeanEkfMapXY, '.', markersize=markerSize,  linestyle='-', linewidth=lineWidth, color = 'r', label='EKF')
	trialTitleString = r'Map \textit{$\dot{z}_m$} velocity'
	plt.title(trialTitleString,fontsize = titleFontSize)
	plt.xlabel(r'time [s]')
	plt.ylabel(r'$\dot{z}_m$ [m/s]')
	# plt.legend(prop={'size': legendSize})
	ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
	plt.grid(False)
	# Set figure size in pixels
	# fig = plt.gcf()
	# DPI = fig.get_dpi()
	# fig.set_size_inches(1920.0/float(DPI),1280.0/float(DPI))
	# Press esc key to exit figure
	plt.gcf().canvas.mpl_connect('key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	# Save plot to directory
	# if save_dir:
	# 	plt.savefig(os.path.join(save_dir, 'vz.png'))
	plt.show()



