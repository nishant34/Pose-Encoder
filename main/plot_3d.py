import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import cv2
import numpy as np

import config as cfg
import data_3dpw_joint_info as joints 


def plot_lim(data, dim):
	poses = np.reshape(data, [-1,dim])
	min_lim = np.min(poses, axis=0)
	max_lim = np.max(poses, axis=0)
	#print('min_lim : ', min_lim, ' max_lim :', max_lim)
	return min_lim, max_lim

def plot_3d_poses(poses, figsize=(10,10), rad_thr=0):
	fig = plt.figure(figsize=figsize, frameon=False)
	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(azim=-110, elev=20)

	for pose in poses:
		for i in np.arange(len(joints.joint_start)):
			x, z, y = [np.array( [pose[joints.joint_start[i], j], pose[joints.joint_end[i], j]] ) for j in range(3)]
			ax.plot(x, -y, z, lw=2, c=joints.limb_col[i])
		for i in range(joints.num_joints):
			ax.scatter(pose[i, 0], -pose[i, 2], pose[i, 1], s=6, c=joints.joint_dot_col[i])

	#ax.invert_xaxis()

	#ax.invert_yaxis()
	# ax.set_xlabel('x')
	# ax.set_ylabel('y')
	# ax.set_zlabel('z')

	if cfg.PY_VERSION == 3:
		ax.get_xaxis().set_ticklabels([])
		ax.get_yaxis().set_ticklabels([])
		ax.get_zaxis().set_ticklabels([])
		# ax.set_zticks([])

	min_lim, max_lim = plot_lim(poses, 3)
	avg_lim = min_lim + (max_lim - min_lim) / 2
	RADIUS = np.max(max_lim - min_lim)/2 + rad_thr

	ax.set_xlim3d([-RADIUS+avg_lim[0], RADIUS+avg_lim[0]])
	ax.set_zlim3d([-RADIUS+avg_lim[1], RADIUS+avg_lim[1]])
	ax.set_ylim3d([-(RADIUS+avg_lim[2]), RADIUS-avg_lim[2]])
	plt.show()
	

def plot_3d_cam(poses, figsize=(5,5), rad_thr=0):
	fig = plt.figure(figsize=figsize, frameon=False)
	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(azim=-110, elev=20)

	for pose in poses:
		for i in np.arange(len(joints.joint_start)):
			x, z, y = [np.array( [pose[joints.joint_start[i], j], pose[joints.joint_end[i], j]] ) for j in range(3)]
			ax.plot(x, y, -z, lw=2, c=joints.limb_col[i])
		for i in range(joints.num_joints):
			ax.scatter(pose[i, 0], pose[i, 2], -pose[i, 1], s=4, c=joints.joint_dot_col[i])

	ax.set_xlabel('x')
	ax.set_ylabel('z')
	ax.set_zlabel('y')

	if cfg.PY_VERSION == 3:
# 		ax.set_xlabel('')
# 		ax.set_ylabel('')
# 		ax.set_zlabel('')

		ax.get_xaxis().set_ticklabels([])
		ax.get_yaxis().set_ticklabels([])
		ax.get_zaxis().set_ticklabels([])

	poses = poses[:,:,[0,2,1]]
	poses *= [1, 1, -1]

	min_lim, max_lim = plot_lim(poses, 3)
	avg_lim = min_lim + (max_lim - min_lim) / 2
	RADIUS = np.max(max_lim - min_lim)/2 + rad_thr

	ax.set_xlim3d([-RADIUS+avg_lim[0], RADIUS+avg_lim[0]])
	ax.set_ylim3d([-RADIUS+avg_lim[1], RADIUS+avg_lim[1]])
	ax.set_zlim3d([-RADIUS+avg_lim[2], RADIUS+avg_lim[2]])
	plt.show()
	# fig.savefig('test.png', transparent=True, bbox_inches='tight', pad_inches=0)

def bone_length(pose):

	for i in range(joints.num_joints-1):
		start_joint = joints.joint_name.get(str(joints.joint_start[i]))
		end_joint = joints.joint_name.get(str(joints.joint_end[i]))
		length = np.linalg.norm(pose[joints.joint_start[i]]-pose[joints.joint_end[i]])
		print(i, '{}_to_{} :'.format(start_joint, end_joint), '{0:.2f}'.format(length))


colors = np.array([[0,0,255], [0,255,0], [255,0,0], [255,0,255], [0,255,255], [255,255,0], [127,127,0], [0,127,0], [100,0,100],
              [255,0,255], [0,255,0], [0,0,255], [255,255,0], [127,127,0], [100,0,100], [175,100,195],
              [0,0,255], [0,255,0], [255,0,0], [255,0,255], [0,255,255], [255,255,0], [127,127,0], [0,127,0], [100,0,100],
              [255,0,255], [0,255,0], [0,0,255], [255,255,0], [127,127,0], [100,0,100], [175,100,195]])

def cam_helper(ax, poses, rad_thr=0):
	for pose in poses:
		for i in np.arange(len(joints.joint_start)):
			x, z, y = [np.array( [pose[joints.joint_start[i], j], pose[joints.joint_end[i], j]] ) for j in range(3)]
			ax.plot(x, y, -z, lw=2, c=joints.limb_col[i])
		for i in range(joints.num_joints):
			ax.scatter(pose[i, 0], pose[i, 2], -pose[i, 1], s=6, c=joints.joint_dot_col[i])

	# ax.set_xlabel('x')
	# ax.set_ylabel('z')
	# ax.set_zlabel('y')

	poses = poses[:,:,[0,2,1]]
	poses *= [1, 1, -1]

	min_lim, max_lim = plot_lim(poses, 3)
	avg_lim = min_lim + (max_lim - min_lim) / 2
	RADIUS = np.max(max_lim - min_lim)/2 + rad_thr

	ax.set_xlim3d([-RADIUS+avg_lim[0], RADIUS+avg_lim[0]])
	ax.set_ylim3d([-RADIUS+avg_lim[1], RADIUS+avg_lim[1]])
	ax.set_zlim3d([-RADIUS+avg_lim[2], RADIUS+avg_lim[2]])

	if cfg.PY_VERSION == 3:
		ax.get_xaxis().set_ticklabels([])
		ax.get_yaxis().set_ticklabels([])
		ax.get_zaxis().set_ticklabels([])

def plot_3d_gt_pred(gt_poses, pred_poses, figsize=(10,5), rad_thr=0, in_poses=False):

	if in_poses:
		pred_poses = pred_poses * [1, -1, -1]


	fig = plt.figure(figsize=figsize, frameon=False)
	
	ax = fig.add_subplot(121, projection='3d')
	ax.view_init(azim=-110, elev=20)
	ax.title.set_text('GT')
	cam_helper(ax, gt_poses, rad_thr)
	
	ax = fig.add_subplot(122, projection='3d')
	ax.view_init(azim=-110, elev=20)
	ax.title.set_text('PRED')
	cam_helper(ax, pred_poses, rad_thr)

	plt.show()


def pose_grid(poses):

	fig = plt.figure(figsize=(16, 16), frameon=False)

	idx = range(16)

	for i in range(1, 5):
		for j in range(1, 5):
				ax = plt.subplot(4, 4, (i-1)*4+j, projection='3d')
				cam_helper(ax, poses[idx[(i-1)*4+j-1]:idx[(i-1)*4+j-1]+1])

				ax_lim = 0.6
				ax.set_xlim3d([-ax_lim, ax_lim])
				ax.set_ylim3d([-ax_lim, ax_lim])
				ax.set_zlim3d([-ax_lim, ax_lim])
				ax.set_title('{}'.format((i-1)*4+j-1))


def pose_grid_recon(poses_gt, poses_recon):

	for p in range(4):

		fig = plt.figure(figsize=(16, 4), frameon=False)
		for i in range(p*4, (p+1)*4):
			ax = plt.subplot(1, 4, i+1-p*4, projection='3d')
			cam_helper(ax, poses_gt[i:i+1])

			ax_lim = 0.6
			ax.set_xlim3d([-ax_lim, ax_lim])
			ax.set_ylim3d([-ax_lim, ax_lim])
			ax.set_zlim3d([-ax_lim, ax_lim])
			ax.set_title('GT_{}'.format(i))

		fig = plt.figure(figsize=(16, 4), frameon=False)
		for i in range(p*4, (p+1)*4):
			ax = plt.subplot(1, 4, i+1-p*4, projection='3d')
			cam_helper(ax, poses_recon[i:i+1])

			ax_lim = 0.6
			ax.set_xlim3d([-ax_lim, ax_lim])
			ax.set_ylim3d([-ax_lim, ax_lim])
			ax.set_zlim3d([-ax_lim, ax_lim])
			ax.set_title('PRED_{}'.format(i))

