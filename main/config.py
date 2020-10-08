"""
Overall project configuration module
"""

# export PYTHONPATH=$PYTHONPATH:/home/cds/govind/prj/aae_theta/

import os
import sys
import numpy as np
import tensorflow as tf
PY_VERSION = 3
if sys.version_info[0] < 3:
    PY_VERSION = 2


def add_pypath(path):
    """insert path to system path"""
    if path not in sys.path:
        sys.path.insert(0, path)


def make_folder(path):
    """make folders"""
    if not os.path.exists(path):
        os.makedirs(path)


ROOT_DIR = '/home/cds/nishant/'

# if ROOT_DIR not in sys.path:
#     sys.path.append(ROOT_DIR)

# add_pypath(os.path.join(ROOT_DIR, 'conf'))
# add_pypath(os.path.join(ROOT_DIR, 'main'))
# add_pypath(os.path.join(ROOT_DIR, 'common'))

D3PW_DATASET_NPY_DIR = os.path.join(ROOT_DIR, 'data')
MOSH_DATASET_DIR = os.path.join(D3PW_DATASET_NPY_DIR, 'Moshpose.npy')
MOSH_DATASET_DIR_ROTMAT = os.path.join(D3PW_DATASET_NPY_DIR, 'rotmat_moshpose.npy')
#   MOSH_DATASET_DIR = os.path.join(ROOT_DIR, 'rotmat_moshpose.npy')

# GPU ID
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Train setting
TRAIN_BATCH_SIZE = 256
NUM_EPOCH = 500
ENCODER_LR = 1e-5
DECODER_LR = 1e-5
Disc_LR = 1e-6
MAX_TO_KEEP=100
THETA_EMB_SIZE = 32
OVERFIT=False
Model_start = 500
Model_start_encoder = 10
INIT_EPOCH = 0
dtype = tf.float32
# Log
LOG_EPOCH = 1
SAVE_MODEL_EPOCH = 5

LOG_PATH = '/home/cds/nishant/tensorboard'
EXP_NAME = 'exp_9'
#tf_log_path = os.path.join(LOG_PATH, EXP_NAME, 'tf_logs')
tf_log_path = LOG_PATH
model_save_path = os.path.join(ROOT_DIR, 'model_aae')
Z_DIM = 32
INIT_EPOCH = 0
model_load_path = os.path.join(ROOT_DIR, 'model_aae')
model_save_path_1 = os.path.join(ROOT_DIR, 'model_aae_theta_20_5')
model_save_path_2 = os.path.join(ROOT_DIR, 'model_Loss_ratio_100_5')
model_save_path_3 = os.path.join(ROOT_DIR, 'model_40_5')
model_save_path_4 = os.path.join(ROOT_DIR, 'model_wgan_loss_20_5')
model_save_path_5 = os.path.join(ROOT_DIR, 'group_joints_autoencoder')
model_save_path_6 = os.path.join(ROOT_DIR, 'group_joints_50_5')
model_save_path_7 = os.path.join(ROOT_DIR, 'group_joints_10_5')
model_save_path_8 = os.path.join(ROOT_DIR, 'group_joints_25_5')
model_save_path_9 = os.path.join(ROOT_DIR, 'group_joints_35_5')
model_save_path_10 = os.path.join(ROOT_DIR, 'gan_only')
model_save_path_11 = os.path.join(ROOT_DIR, 'decoder_only')
model_save_path_12 = os.path.join(ROOT_DIR, 'group_wgan_loss')
model_save_path_13 = os.path.join(ROOT_DIR, 'group_rotmat_gan')
model_save_path_14 = os.path.join(ROOT_DIR, 'new_architecture_gan_only')
model_save_path_15 = os.path.join(ROOT_DIR, 'vae')
model_save_path_16 = os.path.join(ROOT_DIR, 'aae_new')
model_save_path_17 = os.path.join(ROOT_DIR, 'aae_final')
model_save_path_18 = os.path.join(ROOT_DIR, 'aae_final_2')
model_save_path_19 = os.path.join(ROOT_DIR, 'rotmat_final_aae')
model_save_path_20 = os.path.join(ROOT_DIR, 'rotmat_final_aae_cyclic')

#model_save_path_2 = os.path.join(ROOT_DIR, 'model_s')

# Debug
debug_save_path = os.path.join(ROOT_DIR, 'debug')
make_folder(debug_save_path)

# SMPL Model Path
SMPL_NEUTRAL = os.path.join(ROOT_DIR, 'assets', 'neutral_smpl.pkl')
#SMPL_MALE = os.path.join(ROOT_DIR, 'smpl', 'models', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
SMPL_FEMALE = os.path.join(ROOT_DIR, 'smpl', 'models', 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
SMPL_BATCH_SIZE = 8
lambda1 = 100
lambda2 = 5
