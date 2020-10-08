import numpy as np

#2D joints follow mscoco format
#3D joints follow SMPL 24 joints format

#SMPL 24 joint ordering for 3D

# 'hips',
# 'leftUpLeg',
# 'rightUpLeg',
# 'spine',
# 'leftLeg',
# 'rightLeg',
# 'spine1',
# 'leftFoot',
# 'rightFoot',
# 'spine2',
# 'leftToeBase',
# 'rightToeBase',
# 'neck',
# 'leftShoulder',
# 'rightShoulder',
# 'head',
# 'leftArm',
# 'rightArm',
# 'leftForeArm',
# 'rightForeArm',
# 'leftHand',
# 'rightHand',
# 'leftHandIndex1',
# 'rightHandIndex1'

joint_name = {
                   '0':'hips', 
                   '1':'leftUpLeg',
                   '2':'rightUpLeg', 
                   '3':'spine', 
                   '4':'leftLeg', 
                   '5':'rightLeg',
                   '6':'spine1',
                   '7':'leftFoot',
                   '8':'rightFoot',
                   '9':'spine2',
                  '10':'leftToeBase',
                  '11':'rightToeBase',
                  '12':'neck',
                  '13':'leftShoulder',
                  '14':'rightShoulder',
                  '15':'head',
                  '16':'leftArm',
                  '17':'rightArm',
                  '18':'leftForeArm', 
                  '19':'rightForeArm',
                  '20':'leftHand',
                  '21':'rightHand',
                  '22':'leftHandIndex1',
                  '23':'rightHandIndex1'
             }

joint_idx = {
                        'hips':            0,
                        'leftUpLeg':       1,
                        'rightUpLeg':      2,
                        'spine':           3,
                        'leftLeg':         4,
                        'rightLeg':        5,
                        'spine1':          6,
                        'leftFoot':        7,
                        'rightFoot':       8,
                        'spine2':          9,
                        'leftToeBase':    10,
                        'rightToeBase':   11,
                        'neck':           12,
                        'leftShoulder' :  13,
                        'rightShoulder':  14,
                        'head':           15,
                        'leftArm':        16,
                        'rightArm':       17,
                        'leftForeArm':    18,
                        'rightForeArm':   19,
                        'leftHand':       20,
                        'rightHand':      21,
                        'leftHandIndex1': 22,
                        'rightHandIndex1':23
            }

#number of joints
num_joints = 24

parent_id = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21]

#limb start point and end point
joint_start = np.array([ 0,  1,  4,  7,  0,  2,  5,  8,  0,  3,  6,   9, 12, 12, 13, 16, 18, 20, 12, 14, 17, 19, 21])
joint_end   = np.array([ 1,  4,  7, 10,  2,  5,  8, 11,  3,  6,  9,  12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23])


joint_pairs = list(zip(
[ 0,  1,  4,  7,  0,  2,  5,  8,  0,  3,  6,   9, 12, 12, 13, 16, 18, 20, 12, 14, 17, 19, 21],
[ 1,  4,  7, 10,  2,  5,  8, 11,  3,  6,  9,  12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]))

#color of limbs
limb_col =        [
                            '#C4FFC4',  # pelvis to left hip
                            '#00FF00',  # left hip to left knee
                            '#005900',  # left knee to left ankle
                            '#002900',  # left ankle to left toe
    
                            '#6262FF',  # pelvis to right_hip
                            '#0000FF',  # right hip to right_knee
                            '#00009D',  # right knee to right ankle
                            '#00009D',  # right ankle to right toe
    
                            '#FFBF00',  # pelvis to spine
                            '#FFBF00',  # spine to spine1
                            '#FFBF00',  # spine1 to spine2
                            '#FFBF00',  # spine2 to neck
    
                            '#FFBF00',  # headtop to neck
    
                            '#FF8989',  # neck to left shoulder (upper)
                            '#FFA9A9',  # neck to left shoulder 
                            '#FF0000',  # left shoulder to left elbow
                            '#890000',  # left elbow to left wrist
                            '#390000',  # left wrist to left hand index

                            '#C489FF',  # neck to right_shoulder (upper)
                            '#A459FF',  # neck to right_shoulder
                            '#8000FF',  # right_shoulder to right elbow
                            '#310062',  # right elbow to right wrist
                            '#110032',  # right wrist to right hand index
    
                        ]

#color of dots
joint_dot_col = [
                            '#000000',   # 'hips', 
                            '#C4FFC4',   # 'leftUpLeg',
                            '#6262FF',   # 'rightUpLeg', 
                            '#000000',   # 'spine', 
                            '#00FF00',   # 'leftLeg', 
                            '#0000FF',   # 'rightLeg',
                            '#000000',   # 'spine1',
                            '#005900',   # 'leftFoot',
                            '#00009D',   # 'rightFoot',
                            '#000000',   # 'spine2',
                            '#002900',  # 'leftToeBase',
                            '#00009D',  # 'rightToeBase',
                            '#FFBF00',  # 'neck',
                            '#FF8989',  # 'leftShoulder',
                            '#C489FF',  # 'rightShoulder',
                            '#000000',  # 'head',
                            '#FFA9A9',  # 'leftArm',
                            '#A459FF',  # 'rightArm',
                            '#FF0000',  # 'leftForeArm', 
                            '#8000FF',  # 'rightForeArm',
                            '#890000',  # 'leftHand',
                            '#310062',  # 'rightHand',
                            '#390000',  # 'leftHandIndex1',
                            '#110032',  # 'rightHandIndex1'
                        ]

def bone_length(pose):

  for i in range(num_joints-1):
    start_joint = joint_name.get(str(joint_start[i]))
    end_joint = joint_name.get(str(joint_end[i]))
    length = np.linalg.norm(pose[joint_start[i]]-pose[joint_end[i]])
    print(i, '{}_to_{} :'.format(start_joint, end_joint), '{0:.2f}'.format(length))

def to_mupots(poses):

#mupots
#0  head_top
#1  neck
#2  right_shoulder
#3  right_elbow
#4  right_wrist
#5  left_shoulder
#6  left_elbow
#7  left_wrist
#8  right_hip
#9  right_knee
#10 right_ankle
#11 left_hip
#12 left_knee
#13 left_ankle
#14 pelvis
#15 spine
#16 mid_neck


  mupots_poses = np.zeros((poses.shape[0], 17, 3))
  pose_idx = 0

  for pose in poses:
    head_top = pose[joint_idx['head']]
    neck = (pose[joint_idx['leftShoulder']]+pose[joint_idx['rightShoulder']]) / 2
    right_shoulder = pose[joint_idx['rightArm']]
    right_elbow= pose[joint_idx['rightForeArm']]
    right_wrist = pose[joint_idx['rightHand']]
    left_shoulder = pose[joint_idx['leftArm']]
    left_elbow= pose[joint_idx['leftForeArm']]
    left_wrist = pose[joint_idx['leftHand']]
    right_hip = pose[joint_idx['rightUpLeg']]
    right_knee = pose[joint_idx['rightLeg']]
    right_ankle = pose[joint_idx['rightFoot']]
    left_hip = pose[joint_idx['leftUpLeg']]
    left_knee = pose[joint_idx['leftLeg']]
    left_ankle = pose[joint_idx['leftFoot']]
    pelvis = (pose[joint_idx['rightUpLeg']] + pose[joint_idx['leftUpLeg']]) / 2
    spine = pose[joint_idx['spine1']]
    mid_neck = pose[joint_idx['neck']]

    mupots_poses[pose_idx][0] = head_top
    mupots_poses[pose_idx][1] = neck
    mupots_poses[pose_idx][2] = right_shoulder
    mupots_poses[pose_idx][3] = right_elbow
    mupots_poses[pose_idx][4] = right_wrist
    mupots_poses[pose_idx][5] = left_shoulder
    mupots_poses[pose_idx][6] = left_elbow
    mupots_poses[pose_idx][7] = left_wrist
    mupots_poses[pose_idx][8] = right_hip
    mupots_poses[pose_idx][9] = right_knee
    mupots_poses[pose_idx][10] = right_ankle
    mupots_poses[pose_idx][11] = left_hip
    mupots_poses[pose_idx][12] = left_knee
    mupots_poses[pose_idx][13] = left_ankle
    mupots_poses[pose_idx][14] = pelvis
    mupots_poses[pose_idx][15] = spine
    mupots_poses[pose_idx][16] = mid_neck

    pose_idx += 1
    #mupots_poses[:,1] = -mupots_poses[:,1]
    #mupots_poses[:,2] = mupots_poses[:,2]

  return mupots_poses

def readable(poses):
  #pose : (24 x 3)
  #convert to readable
  keypoints_cam_0 = to_mupots(poses)
  tmp_kpt_3d = copy.deepcopy(keypoints_cam_0)
  tmp_kpt_3d[:, :, 1] = -keypoints_cam_0[[0]][0][:,2]
  tmp_kpt_3d[:, :, 2] = keypoints_cam_0[[0]][0][:,1]
  #tmp_kpt_3d = tmp_kpt_3d - tmp_kpt_3d[:, 14:15, :] 

  return tmp_kpt_3d

def mupots15_to_smpl24(pose):
  smpl_pose = np.zeros((24, 3))

  # hips (leftUpLeg for SMPL)
  smpl_pose[1] = pose[11] # left
  smpl_pose[2] = pose[8]

  # knees (leftLeg)
  smpl_pose[4] = pose[12]
  smpl_pose[5] = pose[9]

  # ankles (leftFoot)
  smpl_pose[7] = pose[13]
  smpl_pose[8] = pose[10]

  # shoulders (leftArm)
  smpl_pose[16] = pose[5]
  smpl_pose[17] = pose[2]

  # elbows (leftForeArm)
  smpl_pose[18] = pose[6]
  smpl_pose[19] = pose[3]

  # wrists (leftHand)
  smpl_pose[20] = pose[7]
  smpl_pose[21] = pose[4]

  # near_neck (leftShoulder)
  smpl_pose[13] = pose[1] + (pose[5] - pose[1]) / 2.608
  smpl_pose[14] = pose[1] + (pose[2] - pose[1]) / 2.608

  # neck
  smpl_pose[12] = pose[1] + (pose[0] - pose[1]) / 1.917

  # pelvis (hips)
  smpl_pose[0] = pose[14] + (smpl_pose[12] - pose[14]) / 6.69

  # spine
  smpl_pose[3] = pose[14] + (smpl_pose[12] - pose[14]) / 2.774

  # spine1
  smpl_pose[6] = pose[14] + (smpl_pose[12] - pose[14]) / 1.779

  # spine2
  smpl_pose[9] = pose[14] + (smpl_pose[12] - pose[14]) / 1.539

  # hand_index
  smpl_pose[22] = pose[7] + (pose[7] - pose[6]) / 3.03
  smpl_pose[23] = pose[4] + (pose[4] - pose[3]) / 3.03

  # head
  smpl_pose[15] = pose[0]

  # toes
  #     normal = np.cross(pose[10]-pose[14], pose[13]-pose[14])
  normal = np.cross(pose[1]-pose[14], pose[8]-pose[14])

  unit_normal = normal / np.linalg.norm(normal)

  smpl_pose[10] = smpl_pose[7] + unit_normal * 0.145
  smpl_pose[11] = smpl_pose[8] + unit_normal * 0.145

  smpl_pose[:,1] = -smpl_pose[:, 1] 
  smpl_pose[:,2] = -smpl_pose[:, 2] 

  return smpl_pose