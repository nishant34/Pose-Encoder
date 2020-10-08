"""
Training Module
"""

import os
import time
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from main import config as cfg
from main.model_groups import Model
from main import train_helper
from common.logger_util import logger


def main():
    flag_disc_type=True
    model = train_helper.init_model()
    model.encoder(model.theta_ph)
    model.decoder(model.encoder_net['z_joints'])
    #model.decoder(model.embed_prior)
    
    train_helper.group_loss_autoencoder(model)
    
    model.discriminator(model.encoder_net['z_joints'])
    train_helper.loss_disc_gen(model,real=False)
    model.discriminator(model.embed_prior)
    train_helper.loss_disc_gen(model,real=True,included=2)
    model.encoder_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='AAE_Encoder')
    model.decoder_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='AAE_Decoder')
    model.discriminator_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='AAE_Discriminator')
    print(np.shape(model.decoder_params))
    
    model.tanh1(model.theta_ph)
    opt_encoder,opt_decoder,opt_disc = train_helper.get_optimizer_aae(model)
    #opt_autoencoder,opt_generator,opt_disc = train_helper.new_get_optimizer_aae(model)
    session = train_helper.init_tf_session()

    npy_data = train_helper.read_npy(rotmat =True)
    train_theta, validation_theta, test_theta, _, _, _, _, _, _ = train_helper.parse_npy_data(npy_data)
    #engan
    #train_summary_op = train_helper.group_init_tf_summary(model)
    train_summary_op = train_helper.new_init_tf_summary(model)
    
    metric_summary_ops = train_helper.init_metric_summary(model)

    encoder_saver = tf.train.Saver(model.encoder_params, max_to_keep=cfg.MAX_TO_KEEP)
    decoder_saver = tf.train.Saver(model.decoder_params, max_to_keep=cfg.MAX_TO_KEEP)
    discriminator_saver = tf.train.Saver(model.discriminator_params, max_to_keep=cfg.MAX_TO_KEEP)
    if cfg.INIT_EPOCH != 0:
        logger.info('LOADING MODEL {} FROM {}'.format(cfg.Model_start, cfg.model_load_path))
        #encoder_saver.restore(session, os.path.join(cfg.model_save_path_5, 'encoder-' + str(cfg.Model_start)))
        decoder_saver.restore(session, os.path.join(cfg.model_save_path_12, 'decoder-' + str(cfg.INIT_EPOCH)))
        discriminator_saver.restore(session, os.path.join(cfg.model_save_path_12, 'discriminator-' + str(cfg.INIT_EPOCH)))

    logger.info('EXP NAME: {}'.format(cfg.EXP_NAME))
    logger.info('START EPOCH: {}'.format(cfg.INIT_EPOCH))
    logger.info('END_EPOCH: {}'.format(cfg.INIT_EPOCH + cfg.NUM_EPOCH))
    logger.info('OVERFIT: {}'.format(cfg.OVERFIT))
    logger.info('ENCODER_LR: {}'.format(cfg.ENCODER_LR))
    logger.info('TRAIN_BATCH_SIZE: {}'.format(cfg.TRAIN_BATCH_SIZE))
    logger.info('MAX_TO_KEEP: {}'.format(cfg.MAX_TO_KEEP))

    train_batch_writer = tf.summary.FileWriter(os.path.join(cfg.tf_log_path, 'train_batch'), session.graph)
    train_writer = tf.summary.FileWriter(os.path.join(cfg.tf_log_path, 'train'), session.graph)
    val_writer = tf.summary.FileWriter(os.path.join(cfg.tf_log_path, 'val'))
    test_writer = tf.summary.FileWriter(os.path.join(cfg.tf_log_path, 'test'))
    
    for epoch in range(cfg.INIT_EPOCH+1, cfg.INIT_EPOCH+cfg.NUM_EPOCH+1):
        train_theta = train_theta[np.random.permutation(train_theta.shape[0])]
        n_batches=train_theta.shape[0] // cfg.TRAIN_BATCH_SIZE
        count_gen = 0
        count_disc = 0
        for step in tqdm(range(train_theta.shape[0] // cfg.TRAIN_BATCH_SIZE)):
            
            feed_dict = train_helper.get_feed_dict(step, model, train_theta)
            
            if count_disc<=20:
             session.run([opt_disc], feed_dict=feed_dict)
             count_disc+=1
             count_gen = 0
            else:
                switch+=1
            
                session.run([opt_encoder], feed_dict=feed_dict)
                session.run([opt_decoder], feed_dict=feed_dict)
                count_gen +=1
                
                if count_gen > 20:
                    count_disc = 0
            
        
            

        if epoch % cfg.LOG_EPOCH == 0:
           
            [loss_values_, lr_, debug_dict, train_summary] = session.run([model.loss_values, model.encoder_lr_ph, model.debug_dict, train_summary_op], feed_dict=feed_dict)
            #print('{0:>4} '.format(epoch), '{0:>22} '.format(lr_), cfg.TRAIN_BATCH_SIZE, 'loss_disc:{:.6f}'.format(loss_values_[3]), 'loss_gen_adv:{:.6f}'.format(loss_values_[2]), 'disc_acc:{:.6f}'.format(loss_values_[4]), 'gen_acc:{:.6f}'.format(loss_values_[5]))
            print('{0:>4} '.format(epoch), '{0:>22} '.format(lr_), cfg.TRAIN_BATCH_SIZE, 'autoencoder_loss:{:.6f}'.format(loss_values_[0]), 'loss_disc:{:.6f}'.format(loss_values_[2]), 'gen_loss:{:.6f}'.format(loss_values_[1]))
            #   disc_acc.append(loss_values_[3]) 
            # print(debug_dict['theta_gt'].shape, debug_dict['theta_pred'].shape)
            train_batch_writer.add_summary(train_summary, epoch)
            
        if epoch % cfg.SAVE_MODEL_EPOCH == 0:
        #if epoch % 1 ==0:
            encoder_saver.save(session, os.path.join(cfg.model_save_path_19, 'encoder'), global_step=epoch) 
            decoder_saver.save(session, os.path.join(cfg.model_save_path_19, 'decoder'), global_step=epoch) 
            discriminator_saver.save(session, os.path.join(cfg.model_save_path_19, 'discriminator'), global_step=epoch) 


            train_helper.add_tf_log('train', npy_data, model, session, metric_summary_ops, epoch, train_writer, rotmat=True)
            train_helper.add_tf_log('validation', npy_data, model, session, metric_summary_ops, epoch, val_writer, rotmat=True)
            train_helper.add_tf_log('test', npy_data, model, session, metric_summary_ops, epoch, test_writer, rotmat=True)


if __name__ == '__main__':
    main()