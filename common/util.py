import pickle
import cv2
import numpy as np
import tensorflow as tf
from main import config as cfg
from main import plot_3d as p3d_3dpw
from tqdm import tqdm

## https://github.com/mkocabas/VIBE/blob/731c27382978e242fc1492c4bf53ddd72fd345de/lib/utils/eval_utils.py
def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S2.shape[0] != 3:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_error(pred_3d_all, gt_3d_all):
    pred_3d_all_flat = pred_3d_all.copy()
    pred_3d_all_flat = pred_3d_all_flat - pred_3d_all_flat[:, 0:1,:]
    gt_3d_all_flat = gt_3d_all.copy()
    gt_3d_all_flat = gt_3d_all_flat - gt_3d_all_flat[:, 0:1,:]
    
    joint_wise_error = []
    error = []
    pa_joint_wise_error = []
    pa_error = []
    
    for i in range(len(pred_3d_all_flat)):
        each_pred_3d = pred_3d_all_flat[i]
        each_gt_3d  = gt_3d_all_flat[i]

        tmp_err = np.linalg.norm(each_pred_3d-each_gt_3d, axis=1)
        joint_wise_error.append(tmp_err)
        error.append(np.mean(tmp_err))

        pred3d_sym = compute_similarity_transform(each_pred_3d.copy(), each_gt_3d.copy())
        tmp_pa_err = np.linalg.norm(pred3d_sym-each_gt_3d, axis=1)
        pa_joint_wise_error.append(tmp_pa_err)
        pa_error.append(np.mean(tmp_pa_err))

    joint_wise_error = np.array(joint_wise_error)
    # print(pred_3d_all_flat.shape, gt_3d_all_flat.shape)
    # print(joint_wise_error.shape)
    # print('MPJPE : {:.2f}'.format(np.mean(joint_wise_error)*1000))
    # print('MPJPE : {:0.2f}'.format(np.mean(error)*1000))
    # print('PA_MPJPE : {:0.2f}'.format(np.mean(pa_joint_wise_error)*1000))
    # print('PA_MPJPE : {:0.2f}'.format(np.mean(pa_error)*1000))

    mpjpe = np.mean(error)*1000
    pampjpe = np.mean(pa_error)*1000

    return mpjpe, pampjpe


def inverse_graph(graph_dict):
    inverse_graph_dict = {joint_name: [] for joint_name in joint_names}
    for u_node, edges in graph_dict.items():
        for i, v_node in enumerate(edges):
            inverse_graph_dict.setdefault(v_node, [])
            inverse_graph_dict[v_node].append((i, u_node))
    return inverse_graph_dict


def ft_name(name, ft_id=None):
    return name + '_ft' if ft_id is None else "{}_ft_{}".format(name, ft_id)


def ft_less_name(name_ft):
    return name_ft[:-2]

joint_names = {
                   #'0':'hips', 
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
parent_id = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21]
encoder_joints_dict = {
    'r_arm': ['neck', 'rightShoulder', 'rightForeArm', 'rightHand', 'rightHandIndex1', 'rightArm'],
    'l_arm': ['neck', 'leftShoulder', 'leftForeArm', 'leftHand', 'leftHandIndex1', 'leftArm'],
    ##
    'r_leg': ['rightUpLeg', 'rightLeg', 'rightFoot', 'rightToeBase'],
    'l_leg': ['leftUpLeg', 'leftLeg', 'leftFoot', 'leftToeBase'],
    ##
    'trunk': ['neck', 'spine', 'spine1', 'spine2', 'head'],
    ##
    'trunk_r_arm': ['trunk_ft', 'r_arm_ft'],
    'trunk_l_arm': ['trunk_ft', 'l_arm_ft'],
    'trunk_r_leg': ['trunk_ft', 'r_leg_ft'],
    'trunk_l_leg': ['trunk_ft', 'l_leg_ft'],
    ##
    'upper_body': ['trunk_r_arm_ft', 'trunk_l_arm_ft'],
    'lower_body': ['trunk_r_leg_ft', 'trunk_l_leg_ft'],
    ##
    'full_body': ['upper_body_ft', 'lower_body_ft'],
}
decoder_joints_dict = inverse_graph(encoder_joints_dict)

def get_smpl_j3d_helper(model, session, theta_in):
    
    #if cfg.J3D_3x1:
    root = np.array([[-2.95038585, -0.01918368,  0.30016215]])
    theta_smpl_in = np.reshape(theta_in[0:256], [-1, 23, 3])
    root = np.tile(root, [256, 1, 1])
    #else:
        #root_3x3 = np.array([[[[ 0.97958732, -0.00494194, -0.20095837],
         #      [ 0.03048416, -0.98448372,  0.17280775],
          #     [-0.19869425, -0.17540633, -0.96423712]]]])
        
        #theta_smpl_in = np.reshape(theta_in[0:256], [-1, 23, 3, 3])
        #root = np.tile(root_3x3, [256, 1, 1, 1])

    mean_betas = np.array([[ -0.07105808,  0.5095577 , -0.2100164 , -0.43625274,  0.1097799, -0.00590819,  0.05962404,  0.02799411,  0.23164646, -0.14605673]], dtype=np.float32)    
    theta_smpl_in = np.concatenate([root, theta_smpl_in], axis=1)

    j3d = []
    for i in range(theta_in.shape[0] // 4):

        feed_dict = {}
        feed_dict[model.smpl_theta_ph] = theta_smpl_in[i*4:(i+1)*4]
        feed_dict[model.smpl_beta_ph] = np.tile(mean_betas, [4, 1])

        [cam_smpl_out_] = session.run([model.j3d], feed_dict=feed_dict)

        j3d.append(cam_smpl_out_)

    j3d = np.concatenate(j3d)

    return j3d

def get_smpl_j3d(model, session, theta_in):
    
    j3d = get_smpl_j3d_helper(model, session, theta_in)

    p3d_3dpw.pose_grid(j3d[::16])
    
def grid_interpolate(model, session, theta_emb):
    
    lam = [(i+1)/256. for i in range(256)]
    new_emb = np.zeros((256, 32))

    idx = np.random.randint(0, 255, 1)[0]
    new_emb[0] = theta_emb[idx]
    idx = np.random.randint(0, 255, 1)[0]
    new_emb[255] = theta_emb[idx]

    for i in range(1, 254):
        new_emb[i] = lam[i]*new_emb[255] + (1. - lam[i])*new_emb[0]


    feed_dict = {}
    feed_dict[model.theta_emb_ph] = new_emb.astype(np.float32)
    [theta_inter] = session.run([model.decoder_net['full_body_x']], feed_dict=feed_dict)

    get_smpl_j3d(model, session, theta_inter)
    
def sample_random(model, session):
    
    # if uniform:
    #     new_emb = np.random.uniform(-1, 1, size=[256, 32])
    # else:
    #     new_emb = np.random.normal(size=[256, 32])

    #new_emb = np.random.uniform(-1, 1, size=[256, 32])
    new_emb = np.random.randn(256, 32)
    feed_dict = {}
    feed_dict[model.theta_emb_ph] = new_emb.astype(np.float32)
    [theta_inter] = session.run([model.decoder_net['full_body_x']], feed_dict=feed_dict)

    get_smpl_j3d(model, session, theta_inter)

def vis_recon(model, session, theta_gt, theta_recon):

    j3d_gt = get_smpl_j3d_helper(model, session, theta_gt)
    j3d_recon = get_smpl_j3d_helper(model, session, theta_recon)

    p3d_3dpw.pose_grid_recon(j3d_gt[::16], j3d_recon[::16])


def rodrigues_repr(poses):
    poses_r = np.zeros((poses.shape[0], poses.shape[1], 9))
    for i in tqdm(range(poses.shape[0])):
        for j in range(poses.shape[1]):
            poses_r[i][j] = np.reshape(cv2.Rodrigues(poses[i][j])[0], 9)

    return poses_r
