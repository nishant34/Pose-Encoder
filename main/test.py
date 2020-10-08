"""
Training Module
"""

import os
import time
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from main import config as cfg
from main.model import Model
from main import train_helper
from common.logger_util import logger


def main():
    model = train_helper.init_model()
    model.encoder(model.theta_ph)
    model.decoder(model.theta_emb_pred)
    train_helper.loss(model)

    model.encoder_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='AAE_Encoder')
    model.decoder_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='AAE_Decoder')

    opt_1 = train_helper.get_optimizer_ae(model)
    session = train_helper.init_tf_session()

    npy_data = train_helper.read_npy()
    train_theta, validation_theta, test_theta, _, _, _, _, _, _ = train_helper.parse_npy_data(npy_data)

    train_summary_op = train_helper.init_tf_summary(model)
    metric_summary_ops = train_helper.init_metric_summary(model)

    encoder_saver = tf.train.Saver(model.encoder_params, max_to_keep=cfg.MAX_TO_KEEP)
    decoder_saver = tf.train.Saver(model.decoder_params, max_to_keep=cfg.MAX_TO_KEEP)

    if cfg.INIT_EPOCH != 0:
        logger.info('LOADING MODEL {} FROM {}'.format(cfg.INIT_EPOCH, cfg.model_load_path))
        encoder_saver.restore(session, os.path.join(cfg.model_load_path, 'encoder-' + str(cfg.INIT_EPOCH)))
        decoder_saver.restore(session, os.path.join(cfg.model_load_path, 'decoder-' + str(cfg.INIT_EPOCH)))

    logger.info('EXP NAME: {}'.format(cfg.EXP_NAME))
    logger.info('START EPOCH: {}'.format(cfg.INIT_EPOCH))
    logger.info('END_EPOCH: {}'.format(cfg.INIT_EPOCH + cfg.NUM_EPOCH))
    logger.info('OVERFIT: {}'.format(cfg.OVERFIT))
    logger.info('ENCODER_LR: {}'.format(cfg.ENCODER_LR))
    logger.info('TRAIN_BATCH_SIZE: {}'.format(cfg.TRAIN_BATCH_SIZE))

    train_batch_writer = tf.summary.FileWriter(os.path.join(cfg.tf_log_path, 'train_batch'), session.graph)
    train_writer = tf.summary.FileWriter(os.path.join(cfg.tf_log_path, 'train'), session.graph)
    val_writer = tf.summary.FileWriter(os.path.join(cfg.tf_log_path, 'val'))
    test_writer = tf.summary.FileWriter(os.path.join(cfg.tf_log_path, 'test'))

    # train_helper.add_tf_log('train', npy_data, model, session, metric_summary_ops, epoch, train_writer)
    # train_helper.add_tf_log('validation', npy_data, model, session, metric_summary_ops, epoch, val_writer)
    # train_helper.add_tf_log('test', npy_data, model, session, metric_summary_ops, epoch, test_writer)

    split = 'train'
    theta_recon_error, mpjpe, pampjpe = train_helper.test_routine(split, npy_data, model, session)
    logger.info('{}: Theta_Rec: {:.6f}, MPJPE: {:.2f}, PAMPJPE: {:.2f}'.format(split, theta_recon_error, mpjpe, pampjpe))

    split = 'validation'
    theta_recon_error, mpjpe, pampjpe = train_helper.test_routine(split, npy_data, model, session)
    logger.info('{}: Theta_Rec: {:.6f}, MPJPE: {:.2f}, PAMPJPE: {:.2f}'.format(split, theta_recon_error, mpjpe, pampjpe))

    split = 'test'
    theta_recon_error, mpjpe, pampjpe = train_helper.test_routine(split, npy_data, model, session)
    logger.info('{}: Theta_Rec: {:.6f}, MPJPE: {:.2f}, PAMPJPE: {:.2f}'.format(split, theta_recon_error, mpjpe, pampjpe))

    # for epoch in range(cfg.NUM_EPOCH):
    #     train_theta = train_theta[np.random.permutation(train_theta.shape[0])]

    #     for step in tqdm(range(train_theta.shape[0] // cfg.TRAIN_BATCH_SIZE)):

    #         feed_dict = train_helper.get_feed_dict(step, model, train_theta)
    #         session.run([opt_1], feed_dict=feed_dict)

    #     if epoch % cfg.LOG_EPOCH == 0:
    #         [loss_values_, lr_, debug_dict, train_summary] = session.run([model.loss_values, model.encoder_lr_ph, model.debug_dict, train_summary_op], feed_dict=feed_dict)
    #         print('{0:>4} '.format(epoch), '{0:>22} '.format(lr_), cfg.TRAIN_BATCH_SIZE, 'loss:{:.6f}'.format(loss_values_[0]))

    #         # print(debug_dict['theta_gt'].shape, debug_dict['theta_pred'].shape)
    #         train_batch_writer.add_summary(train_summary, epoch)
            
    #     if epoch % cfg.SAVE_MODEL_EPOCH == 0:
    #         encoder_saver.save(session, os.path.join(cfg.model_save_path, 'encoder'), global_step=epoch) 
    #         decoder_saver.save(session, os.path.join(cfg.model_save_path, 'decoder'), global_step=epoch) 

    #         train_helper.add_tf_log('train', npy_data, model, session, metric_summary_ops, epoch, train_writer)
    #         train_helper.add_tf_log('validation', npy_data, model, session, metric_summary_ops, epoch, val_writer)
    #         train_helper.add_tf_log('test', npy_data, model, session, metric_summary_ops, epoch, test_writer)


if __name__ == '__main__':
    main()