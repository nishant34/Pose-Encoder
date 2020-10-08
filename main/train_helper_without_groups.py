import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from main import config as cfg
#from main.model import Model
#from main.model_groups import Model
from main.model import Model
from common.logger_util import logger
from common import util


def read_npy():
    mosh_theta = np.load(cfg.MOSH_DATASET_DIR)[:, 3:]
    #mosh_theta1 = np.load(cfg.MOSH_DATASET_DIR)
    #mosh_theta = np.reshape(mosh_theta, [-1,23,3])
    d3pw_train_theta = np.load(os.path.join(cfg.D3PW_DATASET_NPY_DIR, 'train_poses.npy'))
    d3pw_validation_theta = np.load(os.path.join(cfg.D3PW_DATASET_NPY_DIR, 'validation_poses.npy'))
    d3pw_test_theta = np.load(os.path.join(cfg.D3PW_DATASET_NPY_DIR, 'test_poses.npy'))

    d3pw_train_beta = np.load(os.path.join(cfg.D3PW_DATASET_NPY_DIR, 'train_betas.npy'))
    d3pw_validation_beta = np.load(os.path.join(cfg.D3PW_DATASET_NPY_DIR, 'validation_betas.npy'))
    d3pw_test_beta = np.load(os.path.join(cfg.D3PW_DATASET_NPY_DIR, 'test_betas.npy'))
     
    d3pw_train_root = d3pw_train_theta[:, 0:1, :]
    #d3pw_train_root = np.reshape(d3pw_train_root,[-1,1,3,3])
    d3pw_validation_root = d3pw_validation_theta[:, 0:1, :]
    #d3pw_validation_root = np.reshape(d3pw_validation_root,[-1,1,3,3])
    d3pw_test_root = d3pw_test_theta[:, 0:1, :]
    #d3pw_test_root = np.reshape(d3pw_test_root,[-1,1,3,3])
    #mosh_theta = mosh_theta1[:,1:,:]
    #mosh_theta = np.reshape(mosh_theta,[-1,69])
    d3pw_train_theta = np.reshape(d3pw_train_theta[:, 1:, :], [-1, 69])
    #d3pw_train_theta = d3pw_train_theta[:, 1:, :]
    d3pw_validation_theta = np.reshape(d3pw_validation_theta[:, 1:, :], [-1, 69])
    #d3pw_validation_theta = d3pw_validation_theta[:, 1:, :]
    d3pw_test_theta = np.reshape(d3pw_test_theta[:, 1:, :], [-1, 69])
    #d3pw_test_theta = d3pw_test_theta[:, 1:, :]
    train_theta = np.concatenate((mosh_theta, d3pw_train_theta), axis=0)

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

def init_tf_summary(model):
    emb_histograms = [tf.summary.histogram('emb_%02d' % i, model.theta_emb_pred[:, i]) for i in range(cfg.THETA_EMB_SIZE)]
    #emb_histograms = [tf.summary.histogram('emb_%02d' % i, model.theta_ph[:, i]) for i in range(cfg.THETA_EMB_SIZE)]
    emb_histograms1 = [tf.summary.histogram('data_distribution' , model.theta_ph)]
    emb_histograms2 = [tf.summary.histogram('decoder_distribution' , model.theta_pred)]
    emb_histograms3 = [tf.summary.histogram('embedding_distribution' , model.theta_emb_pred)]
    train_summary = [
                        tf.summary.scalar('lr', model.encoder_lr_ph),
                        tf.summary.scalar('BATCH_THETA_RECON_MAE', model.loss_theta_recon),
                        tf.summary.scalar('z_recon', model.loss_embed_recon),
                        tf.summary.scalar('loss_cyclic_mae', model.loss_cyclic),
                        tf.summary.scalar('encoder_loss', model.encoder_loss),
                        tf.summary.scalar('decoder_loss', model.decoder_loss),
                        tf.summary.scalar('gen_adv_loss', model.loss_gen_adv),
                        tf.summary.scalar('disc_loss', model.disc_loss),
                        #tf.summary.scalar('disc_acc', model.disc_acc),
                        #tf.summary.scalar('gen_acc', model.gen_acc)
                        
                        
                    ]

    train_summary_op = tf.summary.merge(train_summary+emb_histograms+emb_histograms1+emb_histograms2+emb_histograms3)
    return train_summary_op

def init_vae_summary(model):
    emb_histograms = [tf.summary.histogram('emb_%02d' % i, model.theta_emb_pred[:, i]) for i in range(cfg.THETA_EMB_SIZE)]
    emb_histograms1 = [tf.summary.histogram('data_distribution' , model.theta_ph)]
    emb_histograms2 = [tf.summary.histogram('decoder_distribution' , model.theta_pred)]
    emb_histograms3 = [tf.summary.histogram('embedding_distribution' , model.z_vae)]
    train_summary = [
                        tf.summary.scalar('lr', model.encoder_lr_ph),
                        tf.summary.scalar('BATCH_THETA_RECON_MAE', model.loss_theta_recon),
                        tf.summary.scalar('BATCH_vae_loss', model.vae_loss),
                        #tf.summary.scalar('BATCH_kl_div_loss', model.loss_kl_div),
                        #tf.summary.scalar('disc_acc', model.disc_acc),
                        #tf.summary.scalar('gen_acc', model.gen_acc)
                        
                        
                    ]
    train_summary_op = tf.summary.merge(train_summary+emb_histograms+emb_histograms1+emb_histograms2+emb_histograms3)
    return train_summary_op                



def group_init_tf_summary(model):
    emb_histograms = [tf.summary.histogram('emb_%02d' % i, model.encoder_net['z_joints'][:, i]) for i in range(cfg.THETA_EMB_SIZE)]
    #emb_histograms = [tf.summary.histogram('emb_%02d' % i, model.theta_ph[:, i]) for i in range(cfg.THETA_EMB_SIZE)]
    emb_histograms1 = [tf.summary.histogram('data_distribution' , model.theta_ph)]
    emb_histograms2 = [tf.summary.histogram('decoder_distribution' , model.decoder_net['full_body_x'])]
    emb_histograms3 = [tf.summary.histogram('embedding_distribution' , model.encoder_net['z_joints'])]
    train_summary = [
                        tf.summary.scalar('lr', model.encoder_lr_ph),
                        tf.summary.scalar('BATCH_THETA_RECON_MAE', model.loss_theta_recon),
                        tf.summary.scalar('z_recon', model.loss_embed_recon),
                        tf.summary.scalar('loss_cyclic_mae', model.loss_cyclic),
                        tf.summary.scalar('encoder_loss', model.encoder_loss),
                        tf.summary.scalar('decoder_loss', model.decoder_loss),
                        tf.summary.scalar('gen_adv_loss', model.loss_gen_adv),
                        tf.summary.scalar('disc_loss', model.disc_loss),
                        #tf.summary.scalar('disc_acc', model.disc_acc),
                        #tf.summary.scalar('gen_acc', model.gen_acc)
                        
                        
                    ]

    train_summary_op = tf.summary.merge(train_summary+emb_histograms+emb_histograms1+emb_histograms2+emb_histograms3)
    return train_summary_op

def new_init_tf_summary(model):
    emb_histograms = [tf.summary.histogram('emb_%02d' % i, model.theta_emb_pred[:, i]) for i in range(cfg.THETA_EMB_SIZE)]
    emb_histograms1 = [tf.summary.histogram('data_distribution' , model.theta_ph)]
    emb_histograms2 = [tf.summary.histogram('decoder_distribution' , model.theta_pred)]
    emb_histograms3 = [tf.summary.histogram('emb' , model.theta_emb_pred)] 
    train_summary = [
                        tf.summary.scalar('lr', model.encoder_lr_ph),
                        tf.summary.scalar('BATCH_THETA_RECON_MAE', model.loss_theta_recon),
                        #tf.summary.scalar('z_recon', model.loss_embed_recon),
                        #tf.summary.scalar('loss_cyclic_mae', model.loss_cyclic),
                        #tf.summary.scalar('encoder_loss', model.encoder_loss),
                        #tf.summary.scalar('decoder_loss', model.decoder_loss),
                        tf.summary.scalar('gen_loss', model.gen_loss),
                        tf.summary.scalar('disc_loss', model.disc_loss),
                        #tf.summary.scalar('disc_acc', model.disc_acc),
                        #tf.summary.scalar('gen_acc', model.gen_acc)
                        
                        
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
    opt_1 = tf.train.AdamOptimizer(model.encoder_lr_ph, epsilon=1e-8, use_locking=True).minimize(model.loss_cyclic, var_list=var_to_train1)
    var_to_train2 = model.decoder_params
    opt_2 = tf.train.AdamOptimizer(model.encoder_lr_ph, epsilon=1e-8, use_locking=True).minimize(model.loss_gen_adv, var_list=var_to_train2)
    var_to_train3 = model.discriminator_params
    opt_3 = tf.train.AdamOptimizer(model.encoder_lr_ph, epsilon=1e-8, use_locking=True).minimize(model.disc_loss, var_list=var_to_train3)
    #opt_3 = tf.train.AdamOptimizer(model.encoder_lr_ph, epsilon=1e-8, use_locking=True)
    #gradients = opt_3.compute_gradients(model.disc_loss,var_list=var_to_train3)
    #capped_gvs = [(tf.clip_by_value(grad, -0.001, 0.001), var) for grad, var in gradients]
    #opt_3_clip = opt_3.apply_gradients(capped_gvs)
  
    return opt_1,opt_2,opt_3
def get_optimizer_vae(model):
    
    var_to_train1 = model.encoder_params + model.decoder_params
    opt_1 = tf.train.AdamOptimizer(model.encoder_lr_ph, epsilon=1e-8, use_locking=True).minimize(model.vae_loss, var_list=var_to_train1)
     
    return opt_1    

def new_get_optimizer_aae(model):
    
    var_to_train1 = model.encoder_params+model.decoder_params
    opt_1 = tf.train.AdamOptimizer(model.encoder_lr_ph, epsilon=1e-8, use_locking=True).minimize(model.encoder_loss, var_list=var_to_train1)
    var_to_train2 = model.encoder_params
    opt_2 = tf.train.AdamOptimizer(model.encoder_lr_ph, epsilon=1e-8, use_locking=True).minimize(model.gen_loss, var_list=var_to_train2)
    var_to_train3 = model.discriminator_params
    opt_3 = tf.train.AdamOptimizer(model.encoder_lr_ph, epsilon=1e-8, use_locking=True).minimize(model.disc_loss, var_list=var_to_train3)
    
  
    return opt_1,opt_2,opt_3    
    
def get_feed_dict(step, model, npy_split):
    feed_dict = {}
    feed_dict[model.theta_ph] = npy_split[step*cfg.TRAIN_BATCH_SIZE:(step+1)*cfg.TRAIN_BATCH_SIZE]
    feed_dict[model.embed_prior] = np.random.uniform(-1, 1, size=[cfg.TRAIN_BATCH_SIZE, cfg.Z_DIM])
    #feed_dict[model.embed_prior] = np.random.randn(cfg.TRAIN_BATCH_SIZE, cfg.Z_DIM)
    weight_joint, weight_joint_group, weight_full = 1. / (3* 24), 1. / (3 * 5), 1. / (3  * 1)
    feed_dict[model.weight_vec] = np.array([weight_joint] * 24 + [weight_joint_group] * 5 + [weight_full])
    #feed_dict[model.epsilon] = tf.random_normal(tf.shape(32), dtype=tf.float32, mean=0., stddev=1.0,
                       #name='epsilon')
    #feed_dict[model.epsilon] =   np.random.randn(cfg.TRAIN_BATCH_SIZE, cfg.Z_DIM)          
    #feed_dict[model.embed_prior_mean] = np.mean(np.random.uniform(-1, 1, size=[cfg.TRAIN_BATCH_SIZE, cfg.Z_DIM]),axis=0)
    #feed_dict[model.theta_ph_tanh] = tf.nn.tanh(npy_split[step*cfg.TRAIN_BATCH_SIZE:(step+1)*cfg.TRAIN_BATCH_SIZE])
    #feed_dict[model.embed_prior] = np.random.randn(cfg.TRAIN_BATCH_SIZE, cfg.Z_DIM)
    feed_dict[model.encoder_lr_ph] = cfg.ENCODER_LR    
    feed_dict[model.discriminator_lr_ph] = cfg.Disc_LR
    feed_dict[model.decoder_lr_ph] = cfg.DECODER_LR
    
    return feed_dict



def loss_autoencoder(model):

    model.loss_theta_recon = tf.reduce_mean(tf.abs(model.theta_ph - model.theta_pred))
    model.loss_val_1 = model.loss_theta_recon
    #model.loss_z_recon = tf.reduce_mean(tf.abs(model.theta_embed - model.theta_prior))
    
    #model.loss_embed_recon = tf.reduce_mean(tf.abs(model.embed_prior - model.embed_pred))
    #model.disc_real_loss = 
    #model.debug_dict = {}
    model.debug_dict['loss_val_1'] = model.loss_val_1
    model.debug_dict['theta_gt'] = model.theta_ph
    model.debug_dict['theta_pred'] = model.theta_pred
    
    #model.loss_values = [model.loss_theta_recon]
    model.loss_values.append(model.loss_theta_recon)

def loss_vae(model):
    #model.theta_ph_1 = (model.theta_ph + 6)/8
   # model.loss_recon1 = model.theta_ph_1*tf.log(1e-10 + model.new) + (1-model.theta_ph_1)*tf.log(1e-10+1-model.new)
    #model.loss_recon = tf.reduce_sum(model.loss_recon1,1)
    model.loss_recon = tf.reduce_sum(tf.abs(model.theta_ph - model.theta_pred),1)
    #model.loss_recon = tf.reduce_mean(tf.pow(model.theta_ph - model.theta_pred,2))
    model.loss_kl_div1 =  1 + model.z_std - tf.square(model.z_mean) - tf.exp(model.z_std)
    #model.loss_kl_div1 = tf.square(model.z_mean) + tf.square(model.z_std) - tf.log(1e-8 + tf.square(model.z_std)) - 1 
    #model.loss_kl_div = 0.5*tf.reduce_sum(model.loss_kl_div1,1)
    model.loss_kl_div = -0.5*tf.reduce_sum(model.loss_kl_div1,1)
    #model.vae_loss = model.loss_recon + model.loss_kl_div
    model.vae_loss = tf.reduce_mean(model.loss_recon + model.loss_kl_div) 
    #model.loss_values.append(model.loss_kl_div)
    model.loss_values.append(model.vae_loss)

def loss_z_recon(model,cyclic=False):
    model.loss_embed_recon = tf.reduce_mean(tf.abs(model.embed_prior - model.theta_emb_pred))
    model.debug_dict['loss_embed_recon'] = model.loss_embed_recon
    model.debug_dict['embed_distribution'] = model.embed_prior
    if cyclic:
       model.loss_cyclic = model.loss_theta_recon + model.loss_embed_recon 
       model.encoder_loss = 1*model.loss_cyclic
       #model.loss_values.append(model.encoder_loss)
       model.loss_values.append(model.loss_embed_recon)
       #model.loss_values.append(model.loss_theta_recon)
       model.debug_dict['encoder_loss'] = model.encoder_loss       

def loss_disc_gen(model,real=True,included=1):
    if real==True:
        #model.loss_disc_real = tf.reduce_mean(tf.abs(model.disc_pred-tf.ones_like(model.disc_pred)))
        model.loss_disc_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(model.disc_pred), logits=model.disc_pred)) 
        model.disc_acc_real = tf.reduce_mean(tf.cast(model.disc_pred >= 0, cfg.dtype)) * 100.0
    else:    
        #model.loss_disc_fake = tf.reduce_mean(tf.abs(model.disc_pred+tf.ones_like(model.disc_pred)))
        model.loss_disc_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(model.disc_pred), logits=model.disc_pred)) 
        model.disc_acc_fake = tf.reduce_mean(tf.cast(model.disc_pred < 0, cfg.dtype)) * 100.0
        model.gen_acc = 100 - model.disc_acc_fake
        model.loss_gen_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(model.disc_pred), logits=model.disc_pred))
        model.decoder_loss = 100*model.loss_cyclic + 20*model.loss_gen_adv
        model.debug_dict['decoder_loss'] = model.decoder_loss
        model.debug_dict['gen_acc'] = model.gen_acc   
        model.loss_values.append(model.loss_gen_adv)
    if included==2:
        model.disc_loss = (model.loss_disc_real + model.loss_disc_fake)/2
        model.disc_acc = (model.disc_acc_real + model.disc_acc_fake) / 2
        model.loss_values.append(model.disc_loss)
        model.loss_values.append(model.disc_acc)
        model.loss_values.append(model.gen_acc)
        model.debug_dict['disc_loss'] = model.disc_loss
        model.debug_dict['disc_acc'] = model.disc_acc 



def modified_loss_disc_gen(model,real=True,included=1):
    if real==True:
        model.loss_disc_real = tf.reduce_mean(tf.abs(model.disc_pred_kintree_type-tf.ones_like(model.disc_pred_kintree_type)),axis=0)
        #model.loss_disc_real = -tf.reduce_mean(model.logit_kintree,axis=0)
        model.disc_acc_real = tf.reduce_mean(tf.cast(model.disc_pred_kintree_type >= 0, cfg.dtype)) * 100.0
    else:
        model.loss_disc_fake = tf.reduce_mean(tf.abs(model.disc_pred_kintree_type+tf.ones_like(model.disc_pred_kintree_type)),axis=0)
        #model.loss_disc_fake = tf.reduce_mean(model.logit_kintree,axis=0)
        model.disc_acc_fake = tf.reduce_mean(tf.cast(model.disc_pred_kintree_type < 0, cfg.dtype)) * 100.0
        model.gen_acc = 100 - model.disc_acc_fake
        model.loss_gen_adv1 = tf.reduce_mean(tf.abs(model.disc_pred_kintree_type-tf.ones_like(model.disc_pred_kintree_type)),axis=0)
        #model.loss_gen_adv1 = -tf.reduce_mean((model.logit_kintree),axis=0)
        model.loss_gen_adv = 0.02*tf.reduce_sum(model.weight_vec * model.loss_gen_adv1)
        model.decoder_loss = 100*model.loss_cyclic + 5*model.loss_gen_adv    
        model.loss_values.append(model.loss_gen_adv)
    if included==2:
        model.disc_loss1 = model.loss_disc_real + model.loss_disc_fake  
        model.disc_loss = tf.reduce_sum(model.weight_vec * model.disc_loss1)        
        model.disc_acc = (model.disc_acc_real + model.disc_acc_fake) / 2
        model.loss_values.append(model.disc_loss)
        model.loss_values.append(model.disc_acc)
        model.loss_values.append(model.gen_acc)
        model.debug_dict['disc_loss'] = model.disc_loss
        model.debug_dict['disc_acc'] = model.disc_acc   

def group_loss_autoencoder(model):

    model.loss_theta_recon = tf.reduce_mean(tf.abs(model.theta_ph - model.decoder_net['full_body_x']))
    model.loss_val_1 = model.loss_theta_recon
    #model.loss_z_recon = tf.reduce_mean(tf.abs(model.theta_embed - model.theta_prior))
    
    #model.loss_embed_recon = tf.reduce_mean(tf.abs(model.embed_prior - model.embed_pred))
    #model.disc_real_loss = 
    #model.debug_dict = {}
    model.debug_dict['loss_val_1'] = model.loss_val_1
    model.debug_dict['theta_gt'] = model.theta_ph
    model.debug_dict['theta_pred'] = model.decoder_net['full_body_x']
    
    #model.loss_values = [model.loss_theta_recon]
    #model.loss_values.append(model.loss_theta_recon)
def group_z_recon_loss(model,cyclic=False,lambda1=0):
    model.loss_embed_recon = tf.reduce_mean(tf.abs(model.embed_prior - model.encoder_net['z_joints']))
    #model.debug_dict['loss_embed_recon'] = model.loss_embed_recon
    #model.debug_dict['embed_distribution'] = model.embed_prior
    if cyclic:
       model.loss_cyclic = model.loss_theta_recon + model.loss_embed_recon 
       model.encoder_loss = 10*model.loss_cyclic
       #model.encoder_loss = model.loss_theta_recon
       #model.loss_values.append(model.encoder_loss)
       model.loss_values.append(model.loss_embed_recon)
       model.loss_values.append(model.loss_theta_recon)
       #model.debug_dict['encoder_loss'] = model.encoder_loss

def group_modified_loss_disc_gen(model,real=True,included=1,lambda1=0,lambda2=0):
    if real==True:
        model.loss_disc_real = tf.reduce_mean(tf.abs(model.disc_net['fcc_logits']-tf.ones_like(model.disc_net['fcc_logits'])),axis=0)
        #model.loss_disc_real = -tf.reduce_mean(model.disc_net['wgan_logits'],axis=0)
        model.disc_acc_real = tf.reduce_mean(tf.cast(model.disc_net['fcc_logits'] >= 0, cfg.dtype)) * 100.0
    else:
        model.loss_disc_fake = tf.reduce_mean(tf.abs(model.disc_net['fcc_logits']+tf.ones_like(model.disc_net['fcc_logits'])),axis=0)
        #model.loss_disc_fake = tf.reduce_mean(model.disc_net['wgan_logits'],axis=0)
        model.disc_acc_fake = tf.reduce_mean(tf.cast(model.disc_net['fcc_logits'] < 0, cfg.dtype)) * 100.0
        model.gen_acc = 100 - model.disc_acc_fake
        model.loss_gen_adv1 = tf.reduce_mean(tf.abs(model.disc_net['fcc_logits']-tf.ones_like(model.disc_net['fcc_logits'])),axis=0)
        #model.loss_gen_adv1 = -tf.reduce_mean(model.disc_net['wgan_logits'],axis=0)
        model.loss_gen_adv = tf.reduce_sum(model.weight_vec * model.loss_gen_adv1)
        model.decoder_loss = lambda1*model.loss_cyclic + lambda2*model.loss_gen_adv    
        #model.decoder_loss = model.loss_theta_recon    
        model.loss_values.append(model.loss_gen_adv)
    if included==2:
        model.disc_loss1 = model.loss_disc_real + model.loss_disc_fake  
        model.disc_loss = tf.reduce_sum(model.weight_vec * model.disc_loss1)        
        model.disc_acc = (model.disc_acc_real + model.disc_acc_fake) / 2
        model.loss_values.append(model.disc_loss)
        model.loss_values.append(model.disc_acc)
        model.loss_values.append(model.gen_acc)
        #model.debug_dict['disc_loss'] = model.disc_loss
        #model.debug_dict['disc_acc'] = model.disc_acc   

         

def new_aae_loss(model,real = True):
    #model.loss_values.append(model.loss_theta_recon)
    if real:
      model.disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(model.disc_pred), logits=model.disc_pred)) 
    else :
       model.disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(model.disc_pred), logits=model.disc_pred))
       model.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(model.disc_pred), logits=model.disc_pred))
       model.encoder_loss = model.loss_theta_recon + 0.02*model.gen_loss
       model.disc_loss = (model.disc_loss_real + model.disc_loss_fake)/2   
       model.loss_values.append(model.disc_loss)
       model.loss_values.append(model.gen_loss)         
        
def add_tf_log(split, npy_data, model, session, metric_summary_ops, epoch, log_writer):
    theta_recon_error, mpjpe, pampjpe = test_routine(split, npy_data, model, session)
    logger.info('{} {}: Theta_Rec: {:.6f}, MPJPE: {:.2f}, PAMPJPE: {:.2f}'.format(epoch, split, theta_recon_error, mpjpe, pampjpe))
    metric_feed_dict = {}
    metric_feed_dict[model.mpjpe_ph] = mpjpe
    metric_feed_dict[model.pampjpe_ph] = pampjpe
    metric_feed_dict[model.theta_recon_ph] = theta_recon_error

    [summ_rec, summ_mpjpe] = session.run([metric_summary_ops[0], metric_summary_ops[1]], feed_dict=metric_feed_dict)
    log_writer.add_summary(summ_rec, epoch)
    log_writer.add_summary(summ_mpjpe, epoch)

def test_routine(split, npy_data, model, session):

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
    
def batch_skew(vec, batch_size=None):
    """
    vec is N x 3, batch_size is int
    returns N x 3 x 3. Skew_sym version of each matrix.
    """
    with tf.name_scope("batch_skew", [vec]):
        if batch_size is None:
            batch_size = vec.shape.as_list()[0]
        col_inds = tf.constant([1, 2, 3, 5, 6, 7])
        indices = tf.reshape(
            tf.reshape(tf.range(0, batch_size) * 9, [-1, 1]) + col_inds,
            [-1, 1])
        updates = tf.reshape(
            tf.stack(
                [
                    -vec[:, 2], vec[:, 1], vec[:, 2], -vec[:, 0], -vec[:, 1],
                    vec[:, 0]
                ],
                axis=1), [-1])
        out_shape = [batch_size * 9]
        res = tf.scatter_nd(indices, updates, out_shape)
        res = tf.reshape(res, [batch_size, 3, 3])

        return res
    
    
def batch_rodrigues(theta, name=None):
    """
    Theta is N x 3
    """
   
    with tf.name_scope(name, "batch_rodrigues", [theta]):
        batch_size = theta.shape.as_list()[0]

        # angle = tf.norm(theta, axis=1)
        # r = tf.expand_dims(tf.div(theta, tf.expand_dims(angle + 1e-8, -1)), -1)
        # angle = tf.expand_dims(tf.norm(theta, axis=1) + 1e-8, -1)
        angle = tf.expand_dims(tf.norm(theta + 1e-8, axis=1), -1)
        r = tf.expand_dims(tf.div(theta, angle), -1)

        angle = tf.expand_dims(angle, -1)
        cos = tf.cos(angle)
        sin = tf.sin(angle)

        outer = tf.matmul(r, r, transpose_b=True, name="outer")

        eyes = tf.tile(tf.expand_dims(tf.eye(3), 0), [batch_size, 1, 1])
        R = cos * eyes + (1 - cos) * outer + sin * batch_skew(
            r, batch_size=batch_size)
        return R


def batch_lrotmin(theta, name=None):
    """ NOTE: not used bc I want to reuse R and this is simple.
    Output of this is used to compute joint-to-pose blend shape mapping.
    Equation 9 in SMPL paper.
    Args:
      pose: `Tensor`, N x 72 vector holding the axis-angle rep of K joints.
            This includes the global rotation so K=24
    Returns
      diff_vec : `Tensor`: N x 207 rotation matrix of 23=(K-1) joints with identity subtracted.,
    """
    with tf.name_scope(name, "batch_lrotmin", [theta]):
        with tf.name_scope("ignore_global"):
            
         #theta1 = theta[:,3:72]  
        # N*23 x 3 x 3
         Rs = batch_rodrigues(tf.reshape(theta, [-1, 3]))
         
         #print(Rs.shape)
         lrotmin = tf.reshape(Rs, [-1, 207])
         lrotmin = tf.Session().run(lrotmin)  
        
        return lrotmin
    