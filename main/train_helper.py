import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from main import config as cfg

from main.model_groups import Model
from common.logger_util import logger
from common import util


def read_npy(rotmat = False):
    if rotmat==True:
        mosh_theta = np.load(cfg.MOSH_DATASET_DIR_ROTMAT)[:,1:,:]
        d3pw_train_theta = np.load(os.path.join(cfg.D3PW_DATASET_NPY_DIR, 'rotmat_train_poses.npy'))
        d3pw_validation_theta = np.load(os.path.join(cfg.D3PW_DATASET_NPY_DIR, 'rotmat_val_poses.npy'))
        d3pw_test_theta = np.load(os.path.join(cfg.D3PW_DATASET_NPY_DIR, 'rotmat_test_poses.npy'))

     
        d3pw_train_root = d3pw_train_theta[:, 0:1, :]
        d3pw_train_root = np.reshape(d3pw_train_root,[-1,1,3,3])
        d3pw_validation_root = d3pw_validation_theta[:, 0:1, :]
        d3pw_validation_root = np.reshape(d3pw_validation_root,[-1,1,3,3])
        d3pw_test_root = d3pw_test_theta[:, 0:1, :]
        d3pw_test_root = np.reshape(d3pw_test_root,[-1,1,3,3])
   
        d3pw_train_theta = d3pw_train_theta[:, 1:, :]
        d3pw_validation_theta = d3pw_validation_theta[:, 1:, :]
        d3pw_test_theta = d3pw_test_theta[:, 1:, :]
        train_theta = np.concatenate((mosh_theta, d3pw_train_theta), axis=0)  
    
    
    
    
    
    else:
     mosh_theta = np.load(cfg.MOSH_DATASET_DIR)[:, 3:]
     #mosh_theta1 = np.load(cfg.MOSH_DATASET_DIR)
     mosh_theta = np.reshape(mosh_theta, [-1,23,3])
     d3pw_train_theta = np.load(os.path.join(cfg.D3PW_DATASET_NPY_DIR, 'train_poses.npy'))
     d3pw_validation_theta = np.load(os.path.join(cfg.D3PW_DATASET_NPY_DIR, 'validation_poses.npy'))
     d3pw_test_theta = np.load(os.path.join(cfg.D3PW_DATASET_NPY_DIR, 'test_poses.npy'))

     
     
     d3pw_train_root = d3pw_train_theta[:, 0:1, :]
     d3pw_train_root = np.reshape(d3pw_train_root,[-1,1,3])
     d3pw_validation_root = d3pw_validation_theta[:, 0:1, :]
     d3pw_validation_root = np.reshape(d3pw_validation_root,[-1,1,3])
     d3pw_test_root = d3pw_test_theta[:, 0:1, :]
     d3pw_test_root = np.reshape(d3pw_test_root,[-1,1,3])
     #mosh_theta = mosh_theta1[:,1:,:]
     #d3pw_train_theta = np.reshape(d3pw_train_theta[:, 1:, :], [-1, 69])
     d3pw_train_theta = d3pw_train_theta[:, 1:, :]
     #d3pw_validation_theta = np.reshape(d3pw_validation_theta[:, 1:, :], [-1, 69])
     d3pw_validation_theta = d3pw_validation_theta[:, 1:, :]
     #d3pw_test_theta = np.reshape(d3pw_test_theta[:, 1:, :], [-1, 69])
     d3pw_test_theta = d3pw_test_theta[:, 1:, :]
     train_theta = np.concatenate((mosh_theta, d3pw_train_theta), axis=0)
    d3pw_train_beta = np.load(os.path.join(cfg.D3PW_DATASET_NPY_DIR, 'train_betas.npy'))
    d3pw_validation_beta = np.load(os.path.join(cfg.D3PW_DATASET_NPY_DIR, 'validation_betas.npy'))
    d3pw_test_beta = np.load(os.path.join(cfg.D3PW_DATASET_NPY_DIR, 'test_betas.npy'))
    print('mosh_theta', mosh_theta.shape)
    print('d3pw_train_theta', d3pw_train_theta.shape)
    print('train_theta', train_theta.shape)
    print('d3pw_validation_theta', d3pw_validation_theta.shape)
    print('d3pw_test_theta', d3pw_test_theta.shape)
    print('d3pw_train_root', d3pw_train_root.shape)
    print('d3pw_validation_root', d3pw_validation_root.shape)
    print('d3pw_test_root', d3pw_test_root.shape)
    print('d3pw_train_beta', d3pw_train_beta.shape)
    print('d3pw_validation_beta', d3pw_validation_beta.shape)
    print('d3pw_test_beta', d3pw_test_beta.shape)

    npy_data = {}
    npy_data['train_theta'] = train_theta
    npy_data['d3pw_train_theta'] = d3pw_train_theta
    npy_data['d3pw_validation_theta'] = d3pw_validation_theta
    npy_data['d3pw_test_theta'] = d3pw_test_theta

    npy_data['d3pw_train_root'] = d3pw_train_root
    npy_data['d3pw_validation_root'] = d3pw_validation_root
    npy_data['d3pw_test_root'] = d3pw_test_root

    npy_data['d3pw_train_beta'] = d3pw_train_beta
    npy_data['d3pw_validation_beta'] = d3pw_validation_beta
    npy_data['d3pw_test_beta'] = d3pw_test_beta

    return npy_data

def parse_npy_data(npy_data):
    train_theta = npy_data['train_theta']
    validation_theta = npy_data['d3pw_validation_theta']
    test_theta = npy_data['d3pw_test_theta']

    d3pw_train_root = npy_data['d3pw_train_root']  
    d3pw_validation_root = npy_data['d3pw_validation_root']  
    d3pw_test_root = npy_data['d3pw_test_root']  

    d3pw_train_beta = npy_data['d3pw_train_beta']  
    d3pw_validation_beta = npy_data['d3pw_validation_beta']  
    d3pw_test_beta = npy_data['d3pw_test_beta'] 

    return train_theta, validation_theta, test_theta, d3pw_train_root, d3pw_validation_root, d3pw_test_root, d3pw_train_beta, d3pw_validation_beta, d3pw_test_beta 


def init_model():
    model = Model()
    model.debug_dict = {}
    model.loss_values = []
    return model
def init_old_model():
    model = Model_old()
    model.debug_dict = {}
    model.loss_values = []
    return model    
    

def init_tf_session():
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    session = tf.InteractiveSession(config=config)
    session.run(tf.global_variables_initializer())
    return session



def new_init_tf_summary(model):
    emb_histograms = [tf.summary.histogram('emb_%02d' % i, model.encoder_net['z_joints'][:, i]) for i in range(cfg.THETA_EMB_SIZE)]
    emb_histograms1 = [tf.summary.histogram('data_distribution' , model.theta_ph)]
    emb_histograms2 = [tf.summary.histogram('decoder_distribution' , model.decoder_net['full_body_x'])]
    emb_histograms3 = [tf.summary.histogram('emb' , model.encoder_net['z_joints'])] 
    train_summary = [
                        tf.summary.scalar('lr', model.encoder_lr_ph),
                        tf.summary.scalar('BATCH_THETA_RECON_MAE', model.loss_theta_recon),
                        
                        tf.summary.scalar('gen_loss', model.loss_gen_adv),
                        tf.summary.scalar('disc_loss', model.disc_loss),
                        
                        
                    ]

    train_summary_op = tf.summary.merge(train_summary+emb_histograms+emb_histograms1+emb_histograms2+emb_histograms3)
    return train_summary_op

def init_metric_summary(model):
    metric_summary_1 = [
                        tf.summary.scalar('PAMPJPE', model.pampjpe_ph),
                        tf.summary.scalar('MPJPE', model.mpjpe_ph)
                    ]

    metric_summary_op_1 = tf.summary.merge(metric_summary_1)

    metric_summary_2 = [
                        tf.summary.scalar('THETA_RECON_MAE', model.theta_recon_ph)
                    ]

    metric_summary_op_2 = tf.summary.merge(metric_summary_2)

    return [metric_summary_op_1, metric_summary_op_2]

def get_optimizer_aae(model):
    
    var_to_train1 = model.encoder_params + model.decoder_params
    opt_1 = tf.train.AdamOptimizer(model.encoder_lr_ph, epsilon=1e-8, use_locking=True).minimize(model.loss_theta_recon  , var_list=var_to_train1)
    var_to_train2 = model.encoder_params
    opt_2 = tf.train.AdamOptimizer(model.encoder_lr_ph, epsilon=1e-8, use_locking=True).minimize(model.loss_gen_adv, var_list=var_to_train2)
    var_to_train3 = model.discriminator_params
    opt_3 = tf.train.AdamOptimizer(model.encoder_lr_ph, epsilon=1e-8, use_locking=True).minimize(model.disc_loss, var_list=var_to_train3)
    
    return opt_1,opt_2,opt_3



def get_feed_dict(step, model, npy_split):
    feed_dict = {}
    feed_dict[model.theta_ph] = npy_split[step*cfg.TRAIN_BATCH_SIZE:(step+1)*cfg.TRAIN_BATCH_SIZE]
    feed_dict[model.embed_prior] = np.random.uniform(-1, 1, size=[cfg.TRAIN_BATCH_SIZE, cfg.Z_DIM])
    #feed_dict[model.embed_prior] = np.random.randn(cfg.TRAIN_BATCH_SIZE, cfg.Z_DIM)
    weight_joint, weight_joint_group, weight_full = 1. / (3* 23), 1. / (3 * 5), 1. / (3 * 1)
    feed_dict[model.weight_vec] = np.array([weight_joint] * 23 + [weight_joint_group] * 5 + [weight_full])
    
    feed_dict[model.encoder_lr_ph] = cfg.ENCODER_LR    
    feed_dict[model.discriminator_lr_ph] = cfg.Disc_LR
    feed_dict[model.decoder_lr_ph] = cfg.DECODER_LR
    
    return feed_dict








def loss_disc_gen(model,real=True,included=1):
    if real==True:
        
        model.loss_disc_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(model.disc_pred), logits=model.disc_pred)) 
        model.disc_acc_real = tf.reduce_mean(tf.cast(model.disc_pred >= 0, cfg.dtype)) * 100.0
    else:    
        
        model.loss_disc_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(model.disc_pred), logits=model.disc_pred)) 
        model.disc_acc_fake = tf.reduce_mean(tf.cast(model.disc_pred < 0, cfg.dtype)) * 100.0
        model.gen_acc = 100 - model.disc_acc_fake
        model.loss_gen_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(model.disc_pred), logits=model.disc_pred))
       
        model.loss_values.append(model.loss_gen_adv)
    if included==2:
        model.disc_loss = (model.loss_disc_real + model.loss_disc_fake)/2
        model.disc_acc = (model.disc_acc_real + model.disc_acc_fake) / 2
        model.loss_values.append(model.disc_loss)
        model.loss_values.append(model.disc_acc)
        model.loss_values.append(model.gen_acc)
        model.debug_dict['disc_loss'] = model.disc_loss
        model.debug_dict['disc_acc'] = model.disc_acc 




def group_loss_autoencoder(model):

    model.loss_theta_recon = tf.reduce_mean(tf.abs(model.theta_ph - model.decoder_net['full_body_x']))
    model.loss_val_1 = model.loss_theta_recon
    model.loss_values.append(model.loss_theta_recon)
    
    model.debug_dict['loss_val_1'] = model.loss_val_1
    model.debug_dict['theta_gt'] = model.theta_ph
    model.debug_dict['theta_pred'] = model.decoder_net['full_body_x']
    
    

        
def add_tf_log(split, npy_data, model, session, metric_summary_ops, epoch, log_writer,rotmat = False):
    theta_recon_error, mpjpe, pampjpe = test_routine(split, npy_data, model, session, rotmat)
    logger.info('{} {}: Theta_Rec: {:.6f}, MPJPE: {:.2f}, PAMPJPE: {:.2f}'.format(epoch, split, theta_recon_error, mpjpe, pampjpe))
    metric_feed_dict = {}
    metric_feed_dict[model.mpjpe_ph] = mpjpe
    metric_feed_dict[model.pampjpe_ph] = pampjpe
    metric_feed_dict[model.theta_recon_ph] = theta_recon_error

    [summ_rec, summ_mpjpe] = session.run([metric_summary_ops[0], metric_summary_ops[1]], feed_dict=metric_feed_dict)
    log_writer.add_summary(summ_rec, epoch)
    log_writer.add_summary(summ_mpjpe, epoch)

def test_routine(split, npy_data, model, session,rotmat = False):

    npy_theta = npy_data['d3pw_{}_theta'.format(split)]
    npy_root = npy_data['d3pw_{}_root'.format(split)]
    npy_beta = npy_data['d3pw_{}_beta'.format(split)]

    theta_pred = []
    theta_gt   = []

    for step in tqdm(range(npy_theta.shape[0] // cfg.TRAIN_BATCH_SIZE)):
        feed_dict = get_feed_dict(step, model, npy_theta)
        [debug_dict] = session.run([model.debug_dict], feed_dict=feed_dict)
        theta_pred.append(debug_dict['theta_pred'])
        theta_gt.append(debug_dict['theta_gt'])
    

    if rotmat==True:
        theta_pred = np.reshape(np.concatenate(theta_pred), [-1, 23, 3, 3])
        theta_gt   = np.reshape(np.concatenate(theta_gt), [-1, 23, 3, 3])
    else:    
        theta_pred = np.reshape(np.concatenate(theta_pred), [-1, 23, 3])
        theta_gt   = np.reshape(np.concatenate(theta_gt), [-1, 23, 3])
    npy_root   = npy_root[0:theta_pred.shape[0]]
    npy_beta   = npy_beta[0:theta_pred.shape[0]]
    theta_pred = np.concatenate([npy_root, theta_pred], axis=1).astype(np.float32)
    theta_gt   = np.concatenate([npy_root, theta_gt], axis=1).astype(np.float32)
    npy_beta   = npy_beta.astype(np.float32)
    
    theta_recon_error = np.mean(np.abs(theta_pred-theta_gt))


    j3d_gt = []

    for step in range(theta_gt.shape[0] // cfg.SMPL_BATCH_SIZE):
        theta = theta_gt[step*cfg.SMPL_BATCH_SIZE:(step+1)*cfg.SMPL_BATCH_SIZE]
        beta = npy_beta[step*cfg.SMPL_BATCH_SIZE:(step+1)*cfg.SMPL_BATCH_SIZE]

        smpl_feed_dict = {}
        smpl_feed_dict[model.smpl_theta_ph] = theta
        smpl_feed_dict[model.smpl_beta_ph] = beta
        [j3d] = session.run([model.j3d], feed_dict=smpl_feed_dict)
        j3d_gt.append(j3d)
    
    j3d_pred = []

    for step in range(theta_pred.shape[0] // cfg.SMPL_BATCH_SIZE):
        theta = theta_pred[step*cfg.SMPL_BATCH_SIZE:(step+1)*cfg.SMPL_BATCH_SIZE]
        beta = npy_beta[step*cfg.SMPL_BATCH_SIZE:(step+1)*cfg.SMPL_BATCH_SIZE]

        smpl_feed_dict = {}
        smpl_feed_dict[model.smpl_theta_ph] = theta
        smpl_feed_dict[model.smpl_beta_ph] = beta
        [j3d] = session.run([model.j3d], feed_dict=smpl_feed_dict)
        j3d_pred.append(j3d)
    
    j3d_gt = np.concatenate(j3d_gt)
    j3d_pred = np.concatenate(j3d_pred)

    mpjpe, pampjpe = util.compute_error(j3d_pred, j3d_gt)

    return theta_recon_error, mpjpe, pampjpe
    
