import tensorflow as tf

from common.logger_util import logger
from common import util
from smpl.smpl_layer import SmplTPoseLayer
import tensorflow.contrib.layers as tf_layer
from collections import OrderedDict


class Model:

    def __init__(self):
    	self.init_placeholder()
    	self.init_smpl()
        self.joint_names = util.joint_names
        self.joint_idx = util.joint_idx
        self.parent_id = util.parent_id
        self.encoder_joints_dict = util.encoder_joints_dict
        self.decoder_joints_dict = util.decoder_joints_dict

    	# self.encoder(self.theta_ph)
    	# self.decoder(self.theta_emb_ph)

    def init_placeholder(self):
    	self.theta_ph = tf.placeholder(tf.float32, shape=[None, 23, 9])
        self.embed_prior = tf.placeholder(tf.float32, shape=[None,32])
        #self.embed_prior_mean = tf.placeholder(tf.float32, shape=[32])
    	#self.theta_ph_tanh = tf.placeholder(tf.float32, shape=[None, 69])
    	self.theta_emb_ph = tf.placeholder(tf.float32, shape=[None, 32])
        self.encoder_lr_ph = tf.placeholder(tf.float32)
    	self.decoder_lr_ph = tf.placeholder(tf.float32)
        self.discriminator_lr_ph = tf.placeholder(tf.float32)
        self.weight_vec = tf.placeholder(tf.float32, shape=[29,], name='weight_vec_ph')    
    	# Metric
        self.mpjpe_ph = tf.placeholder(tf.float32, shape=(), name="MPJPE")
        self.pampjpe_ph = tf.placeholder(tf.float32, shape=(), name="PAMPJPE") 
        self.theta_recon_ph = tf.placeholder(tf.float32, shape=(), name="THETA_RECON_MAE") 

        # SMPL
        #self.smpl_theta_ph = tf.placeholder(tf.float32, shape=[None, 24, 3])
        self.smpl_theta_ph = tf.placeholder(tf.float32, shape=[None, 24, 3, 3])
        self.smpl_beta_ph = tf.placeholder(tf.float32, shape=[None, 10])

    def init_smpl(self):
    	self.smpl = SmplTPoseLayer(theta_in_rodrigues=False, theta_is_perfect_rotmtx=True)
    	cam_smpl_out = self.smpl([self.smpl_theta_ph, self.smpl_beta_ph, None, None])
        j3d = cam_smpl_out[1]
        self.j3d = j3d - j3d[:,0:1]
        

    def tanh1(self, theta):
		self.theta_ph_tanh = tf.nn.tanh(theta) 

    

    def encoder(self, input_encoder_x):

        with tf.variable_scope('AAE_Encoder', reuse=tf.AUTO_REUSE):
            encoder_net = OrderedDict()


            ############### Indiviual Joints ###############
            for joint_name in self.joint_idx:
                encoder_net[joint_name] = input_encoder_x[:, self.joint_idx[joint_name]-1, :]

            ##################### Level 1 Joint Group ####################
            ###  joint -> joint_group_1 -> joint_group_1_ft ###
            for joint_group in ['l_arm', 'r_arm', 'r_leg', 'l_leg', 'trunk']:
                joint_group_ft = joint_group + '_ft'
                encoder_net[joint_group] = tf.concat([encoder_net[sub_part] for sub_part in self.encoder_joints_dict[joint_group]], axis=1)
                encoder_net[joint_group_ft] = tf_layer.fully_connected(encoder_net[joint_group],
                                                                   num_outputs=32,
                                                                   activation_fn=tf.nn.relu,
                                                                   scope=joint_group_ft)
            ##################### Level 2 Joint Group ####################
            ###  joint_group_1_ft -> joint_group_2 -> joint_group_2_ft ###
            for joint_group in ['trunk_l_arm', 'trunk_r_arm', 'trunk_r_leg', 'trunk_l_leg']:
                joint_group_ft = joint_group + '_ft'
                encoder_net[joint_group] = tf.concat([encoder_net[sub_part] for sub_part in self.encoder_joints_dict[joint_group]], axis=1)
                encoder_net[joint_group_ft] = tf_layer.fully_connected(encoder_net[joint_group],
                                                                   num_outputs=64,
                                                                   activation_fn=tf.nn.relu,
                                                                   scope=joint_group_ft)

            ##################### Level 3 Joint Group ####################
        ###  joint_group_2_ft -> joint_group_3 -> joint_group_3_ft ###
            for joint_group in ['upper_body', 'lower_body']:
                joint_group_ft = joint_group + '_ft'
                encoder_net[joint_group] = tf.concat([encoder_net[sub_part] for sub_part in self.encoder_joints_dict[joint_group]], axis=1)
                encoder_net[joint_group_ft] = tf_layer.fully_connected(encoder_net[joint_group],
                                                                   num_outputs=128,
                                                                   activation_fn=tf.nn.relu,
                                                                   scope=joint_group_ft)

        ##################### Level 4 Joint Group ####################
        ###  joint_group_3_ft -> joint_group_4 -> joint_group_4_ft ###
            for joint_group in ['full_body']:
                joint_group_ft = joint_group + '_ft'
                encoder_net[joint_group] = tf.concat([encoder_net[sub_part] for sub_part in self.encoder_joints_dict[joint_group]], axis=1)
                encoder_net[joint_group_ft] = tf_layer.fully_connected(encoder_net[joint_group],
                                                                   num_outputs=512,
                                                                   activation_fn=tf.nn.relu,
                                                                   scope=joint_group_ft)

        ##################### Final Layer of FCC ####################
            encoder_net['full_body_ft2'] = tf_layer.fully_connected(encoder_net['full_body_ft'],
                                                                num_outputs=512,
                                                                activation_fn=tf.nn.relu,
                                                                scope='full_body_ft2')

            encoder_net['z_joints'] = tf_layer.fully_connected(encoder_net['full_body_ft'],
                                                           num_outputs=32,
                                                           activation_fn=tf.nn.tanh,
                                                           scope='z_joints')
            #return encoder_net
            self.encoder_net = encoder_net                                                     
                                                       


    def decoder(self, input_decoder_z_joints):

        with tf.variable_scope('AAE_Decoder', reuse=tf.AUTO_REUSE):

            decoder_net = OrderedDict()

            decoder_net['z_joints'] = input_decoder_z_joints

            decoder_net['full_body_ft2'] = tf_layer.fully_connected(decoder_net['z_joints'],
                                                                num_outputs=512,
                                                                activation_fn=tf.nn.relu,
                                                                scope='full_body_ft2')

            decoder_net['full_body_ft'] = tf_layer.fully_connected(decoder_net['full_body_ft2'],
                                                               num_outputs=512,
                                                               activation_fn=tf.nn.relu,
                                                               scope='full_body_ft')

        ###
            decoder_net['full_body'] = (
                tf_layer.fully_connected(decoder_net['full_body_ft'],
                                     num_outputs=512,
                                     activation_fn=tf.nn.relu,
                                     scope='full_body')

            )

            for joint_group in ['upper_body', 'lower_body']:
                n_units = 128
                relu_fn = tf.nn.relu
                joint_group_ft = util.ft_name(joint_group)
                super_group_layers = [decoder_net[super_part][:, i * n_units: (i + 1) * n_units] for i, super_part in
                                  self.decoder_joints_dict[joint_group_ft]]
                decoder_net[joint_group_ft] = relu_fn(tf.add_n(super_group_layers))

                decoder_net[joint_group] = tf_layer.fully_connected(decoder_net[joint_group_ft],
                                                                num_outputs=n_units,
                                                                activation_fn=tf.nn.relu,
                                                                scope=joint_group)

            for joint_group in ['trunk_l_arm', 'trunk_r_arm', 'trunk_r_leg', 'trunk_l_leg']:
               n_units = 64
               relu_fn = tf.nn.relu
               joint_group_ft = util.ft_name(joint_group)
               super_group_layers = [decoder_net[super_part][:, i * n_units: (i + 1) * n_units] for i, super_part in
                                  self.decoder_joints_dict[joint_group_ft]]
               decoder_net[joint_group_ft] = relu_fn(tf.add_n(super_group_layers))

               decoder_net[joint_group] = tf_layer.fully_connected(decoder_net[joint_group_ft],
                                                                num_outputs=n_units,
                                                                activation_fn=tf.nn.relu,
                                                                scope=joint_group)

            for joint_group in ['l_arm', 'r_arm', 'r_leg', 'l_leg', 'trunk']:
                n_units = 32
                relu_fn = tf.nn.relu
                joint_group_ft = util.ft_name(joint_group)
                super_group_layers = [decoder_net[super_part][:, i * n_units: (i + 1) * n_units] for i, super_part in
                                  self.decoder_joints_dict[joint_group_ft]]
                decoder_net[joint_group_ft] = relu_fn(tf.add_n(super_group_layers))
                #no. of params per joint = 3
                joint_group_ft_units = 9 * len(self .encoder_joints_dict[joint_group])
                decoder_net[joint_group] = tf_layer.fully_connected(decoder_net[joint_group_ft],
                                                                num_outputs=joint_group_ft_units,
                                                                activation_fn=None,
                                                                scope=joint_group)
            for joint in self.joint_idx:
                n_units = 9
                super_group_layers = [decoder_net[super_part][:, i * n_units: (i + 1) * n_units] for i, super_part in
                                  self.decoder_joints_dict[joint]]
                decoder_net[joint] = tf.add_n(super_group_layers)

        ############### concat Indiviual Joints ###############
            full_body_x = tf.concat([tf.expand_dims(decoder_net[joint_name], axis=1) for joint_name in self.joint_idx], axis=1)
            #norms = tf.norm(full_body_x, axis=2, keep_dims=True)
            #decoder_net['full_body_x'] = full_body_x / (norms + 1e-8)  # For Handing 0 norm cases
            decoder_net['full_body_x'] = full_body_x 
            #return decoder_net
            #print("shape of decoder output is " )
            #print(decoder_net['full_body_x'].shape)
            self.decoder_net = decoder_net
    def discriminator(self, theta):
        with tf.variable_scope('AAE_Discriminator', reuse=tf.AUTO_REUSE):
             
            fc_out = tf.layers.dense(theta, 512, activation=tf.nn.relu)
            print('DISCRIMINATOR_FC_1', fc_out.shape)

            fc_out = tf.layers.dense(fc_out, 512, activation=tf.nn.relu)
		    #print('DISCRIMINATOR_FC_2', fc_out.shape)
            
            self.per_joint = tf.layers.dense(fc_out, 24, activation=None)

            fc_out = tf.layers.dense(fc_out, 1000, activation=tf.nn.relu)

            self.per_joint_group = tf.layers.dense(fc_out, 5, activation=None)

            fc_out = tf.layers.dense(fc_out, 1024, activation=tf.nn.relu)
			#print('DISCRIMINATOR_FC_3', fc_out.shape)
           
            fc_out = tf.layers.dense(fc_out, 1024, activation=tf.nn.relu)
			#print('DISCRIMINATOR_FC_4', fc_out.shape)
            										 
            self.disc_pred = tf.layers.dense(fc_out, 1, activation=None)
			#print('DISCRIMINATOR_OUT', self.real.shape)
            self.disc_pred_kintree_type = tf.concat([self.per_joint,self.per_joint_group,self.disc_pred],axis=-1)
            self.logit_kintree = tf.nn.sigmoid(self.disc_pred_kintree_type)

    def discriminator1(self,input_disc_x):
        with tf.variable_scope('AAE_Discriminator', reuse=tf.AUTO_REUSE):
             disc_net = {'final_fc_names': []}

        ############### Indiviual Joints ###############
             for i, joint_number in enumerate(self.joint_names):
              joint_name = self.joint_names[joint_number]
              disc_net[self.joint_names[joint_number]] = input_disc_x[:, i-1, :]

              joint_ft_1 = util.ft_name(joint_name, 1)
              joint_ft_2 = util.ft_name(joint_name, 2)
              print(joint_ft_2)
              joint_fc = joint_name + '_fc'
              reuse = False if i == 0 else True
              disc_net[joint_ft_1] = tf_layer.fully_connected(disc_net[joint_name],
                                                            num_outputs=32,
                                                            activation_fn=tf.nn.relu,
                                                            scope='layer_1_shared',
                                                            reuse=reuse)

              disc_net[joint_ft_2] = tf_layer.fully_connected(disc_net[joint_ft_1],
                                                            num_outputs=32,
                                                            activation_fn=tf.nn.relu,
                                                            scope='layer_2_shared',
                                                            reuse=reuse)

              disc_net[joint_fc] = tf_layer.fully_connected(disc_net[joint_ft_2],
                                                          num_outputs=1,
                                                          activation_fn=None,
                                                          scope=joint_fc)
              disc_net['final_fc_names'].append(joint_fc)

            ############### Level 1 Joint Group ###############
            ###  joint -> joint_group_1 -> joint_group_1_ft ###
              joint_groups = ['l_arm', 'r_arm', 'r_leg', 'l_leg', 'trunk']

             for joint_group in joint_groups:
              joint_group_ft = joint_group + '_ft'
              joint_group_fc = joint_group + '_fc'

              disc_net[joint_group] = tf.concat([disc_net[util.ft_name(sub_part, 2)] for sub_part in self.encoder_joints_dict[joint_group]],
                                              axis=1)

              disc_net[joint_group_ft] = tf_layer.fully_connected(disc_net[joint_group],
                                                                num_outputs=200,
                                                                activation_fn=tf.nn.relu,
                                                                scope=joint_group_ft)

              disc_net[joint_group_fc] = tf_layer.fully_connected(disc_net[joint_group_ft],
                                                                num_outputs=1,
                                                                activation_fn=None,
                                                                scope=joint_group_fc)
              disc_net['final_fc_names'].append(joint_group_fc)

             disc_net['joint_groups_concat'] = tf.concat([disc_net[util.ft_name(joint_group)] for joint_group in joint_groups], axis=1)

             disc_net['joint_groups_fcc_1'] = tf_layer.fully_connected(disc_net['joint_groups_concat'],
                                                                  num_outputs=1024,
                                                                  activation_fn=tf.nn.relu,
                                                                  scope='joint_groups_fcc_1')

             disc_net['joint_groups_fcc_2'] = tf_layer.fully_connected(disc_net['joint_groups_fcc_1'],
                                                                  num_outputs=1024,
                                                                  activation_fn=tf.nn.relu,
                                                                  scope='joint_groups_fcc_2')

             disc_net['joint_groups_final_fc'] = tf_layer.fully_connected(disc_net['joint_groups_fcc_2'],
                                                                     num_outputs=1,
                                                                     activation_fn=None,
                                                                     scope='joint_groups_final_fc')
             disc_net['final_fc_names'].append('joint_groups_final_fc')
             disc_net['fcc_logits'] = tf.concat([disc_net[fc_name] for fc_name in disc_net['final_fc_names']], axis=1)
             disc_net['wgan_logits'] = tf.nn.sigmoid(disc_net['fcc_logits'])
             self.disc_net = disc_net   
           


            
 


def main():
	model = Model()


if __name__ == '__main__':
    main()
