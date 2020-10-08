import tensorflow as tf

from common.logger_util import logger
from smpl.smpl_layer import SmplTPoseLayer


class Model:

    def __init__(self):
    	self.init_placeholder()		
    	self.init_smpl()
    	# self.encoder(self.theta_ph)
    	# self.decoder(self.theta_emb_ph)

    def init_placeholder(self):
    	self.theta_ph = tf.placeholder(tf.float32, shape=[None, 69])
        self.embed_prior = tf.placeholder(tf.float32, shape=[None,32])
        #self.embed_prior_mean = tf.placeholder(tf.float32, shape=[32])
    	#self.theta_ph_tanh = tf.placeholder(tf.float32, shape=[None, 69])
    	self.theta_emb_ph = tf.placeholder(tf.float32, shape=[None, 32])
        self.encoder_lr_ph = tf.placeholder(tf.float32)
    	self.decoder_lr_ph = tf.placeholder(tf.float32)
        self.discriminator_lr_ph = tf.placeholder(tf.float32)
        self.weight_vec = tf.placeholder(tf.float32, shape=[30,], name='weight_vec_ph')    
    	# Metric
        self.mpjpe_ph = tf.placeholder(tf.float32, shape=(), name="MPJPE")
        self.pampjpe_ph = tf.placeholder(tf.float32, shape=(), name="PAMPJPE") 
        self.theta_recon_ph = tf.placeholder(tf.float32, shape=(), name="THETA_RECON_MAE") 

        # SMPL
        self.smpl_theta_ph = tf.placeholder(tf.float32, shape=[None, 24, 3])
        self.smpl_beta_ph = tf.placeholder(tf.float32, shape=[None, 10])

    def init_smpl(self):
    	self.smpl = SmplTPoseLayer(theta_in_rodrigues=True, theta_is_perfect_rotmtx=False)
    	cam_smpl_out = self.smpl([self.smpl_theta_ph, self.smpl_beta_ph, None, None])
        j3d = cam_smpl_out[1]
        self.j3d = j3d - j3d[:,0:1]
        

    def tanh1(self, theta):
		self.theta_ph_tanh = tf.nn.tanh(theta) 
		

    def encoder(self, theta):

		with tf.variable_scope('AAE_Encoder', reuse=tf.AUTO_REUSE):
			
			fc_out = tf.layers.dense(theta, 128, activation=tf.nn.relu)
			print('ENCODER_FC_1', fc_out.shape)

			fc_out = tf.layers.dense(fc_out, 256, activation=tf.nn.relu)
			print('ENCODER_FC_2', fc_out.shape)

			fc_out = tf.layers.dense(fc_out, 256, activation=tf.nn.relu)
			print('ENCODER_FC_3', fc_out.shape)

			fc_out = tf.layers.dense(fc_out, 512, activation=tf.nn.relu)
			print('ENCODER_FC_4', fc_out.shape)

			fc_out = tf.layers.dense(fc_out, 512, activation=tf.nn.relu)
			print('ENCODER_FC_5', fc_out.shape)

			fc_out = tf.layers.dense(fc_out, 1024, activation=tf.nn.relu)
			print('ENCODER_FC_5', fc_out.shape)

			fc_out = tf.layers.dense(fc_out, 1024, activation=tf.nn.relu)
			print('ENCODER_FC_5', fc_out.shape)

			self.theta_emb_pred = tf.layers.dense(fc_out, 32, activation=tf.nn.tanh)
			print('ENCODER_Theta_Emb_Layer', self.theta_emb_pred.shape)
            
			#fc_out_1 = tf.layers.dense(theta, 128, activation=tf.nn.relu)
			#print('ENCODER_FC_1', fc_out_1.shape)

			#fc_out_1 = tf.layers.dense(fc_out_1, 256, activation=tf.nn.relu)
			#print('ENCODER_FC_2', fc_out.shape)

			#fc_out_1 = tf.layers.dense(fc_out_1, 256, activation=tf.nn.relu)
			#print('ENCODER_FC_3', fc_out.shape)

			#fc_out_1 = tf.layers.dense(fc_out_1, 512, activation=tf.nn.relu)
			#print('ENCODER_FC_4', fc_out.shape)

			#fc_out_1 = tf.layers.dense(fc_out_1, 512, activation=tf.nn.relu)
			#print('ENCODER_FC_5', fc_out.shape)

			#fc_out_1 = tf.layers.dense(fc_out_1, 1024, activation=tf.nn.relu)
			#print('ENCODER_FC_5', fc_out.shape)

			#fc_out_1 = tf.layers.dense(fc_out_1, 1024, activation=tf.nn.relu)
			#print('ENCODER_FC_5', fc_out.shape)

			#self.theta_emb_pred = tf.layers.dense(fc_out_1, 16, activation=tf.nn.tanh)
			#print('ENCODER_Theta_Emb_Layer', self.theta_emb_pred.shape)
            
			#self.theta_emb_pred =  tf.concat([self.theta_emb_pred_1,self.theta_emb_pred],axis=-1)
			#VAE
			
			#self.z_mean = tf.layers.dense(fc_out, 32, activation=None)

			#self.z_std = tf.layers.dense(fc_out, 32, activation=None)
			#self.z_std = 1e-6 + tf.nn.softplus(self.z_std)
			#epsilon = tf.random_normal(tf.shape(self.z_std), dtype=tf.float32, mean=0., stddev=1.0,
                       #name='epsilon')

			
            
			#self.z_vae = self.z_mean + tf.exp(self.z_std/2)*epsilon		   
			#self.z_vae = self.z_mean + self.z_std*(self.epsilon)		   

    def decoder(self, theta_emb_pred):

    	with tf.variable_scope('AAE_Decoder', reuse=tf.AUTO_REUSE):

			fc_out = tf.layers.dense(theta_emb_pred, 512, activation=tf.nn.relu)
			print('DECODER_FC_1', fc_out.shape)
             
			fc_out_1 = tf.layers.dense(theta_emb_pred, 512, activation=tf.nn.relu)
			print('DECODER_FC_2', fc_out_1.shape)

			fc_out_1 = tf.layers.dense(fc_out_1, 512, activation=tf.nn.relu)
			print('DECODER_FC_3', fc_out.shape)

			fc_out_1 = tf.layers.dense(fc_out_1, 256, activation=tf.nn.relu)
			print('DECODER_FC_2', fc_out_1.shape)

			fc_out_1 = tf.layers.dense(fc_out_1, 256, activation=tf.nn.relu)
			print('DECODER_FC_3', fc_out.shape)

			fc_out_1 = tf.layers.dense(fc_out_1, 128, activation=tf.nn.relu)
			print('DECODER_FC_2', fc_out_1.shape)

			#fc_out_1 = tf.layers.dense(fc_out_1, 24, activation=tf.nn.relu)
			#print('DECODER_FC_3', fc_out.shape)			

			fc_out = tf.layers.dense(fc_out, 512, activation=tf.nn.relu)
			#print('DECODER_FC_2', fc_out.shape)
            
			fc_out = tf.layers.dense(fc_out, 512, activation=tf.nn.relu)
			#print('DECODER_FC_2', fc_out.shape)

			fc_out = tf.layers.dense(fc_out, 256, activation=tf.nn.relu)
			print('DECODER_FC_3', fc_out.shape)

			fc_out = tf.layers.dense(fc_out, 256, activation=tf.nn.relu)
			print('DECODER_FC_4', fc_out.shape)

			fc_out = tf.layers.dense(fc_out, 128, activation=tf.nn.relu)
			print('DECODER_FC_5', fc_out.shape)

			self.theta_pred = tf.layers.dense(fc_out, 24)
			self.theta_pred_1 = tf.layers.dense(fc_out_1, 24)
            
			fc_out_2 = tf.layers.dense(theta_emb_pred, 512, activation=tf.nn.relu)
			print('DECODER_FC_2', fc_out_1.shape)

			fc_out_2 = tf.layers.dense(fc_out_2, 512, activation=tf.nn.relu)
			print('DECODER_FC_3', fc_out.shape)

			fc_out_2 = tf.layers.dense(fc_out_2, 256, activation=tf.nn.relu)
			print('DECODER_FC_2', fc_out_1.shape)

			fc_out_2 = tf.layers.dense(fc_out_2, 256, activation=tf.nn.relu)
			print('DECODER_FC_3', fc_out.shape)

			fc_out_2 = tf.layers.dense(fc_out_2, 128, activation=tf.nn.relu)
			print('DECODER_FC_2', fc_out_1.shape)

			#fc_out_2 = tf.layers.dense(fc_out_2, 30, activation=tf.nn.relu)
			#print('DECODER_FC_3', fc_out.shape)			
			self.theta_pred_2 = tf.layers.dense(fc_out_2, 21)
			self.theta_pred = tf.concat([self.theta_pred,self.theta_pred_1,self.theta_pred_2],axis=-1)
			print('DECODER_Theta_Layer', self.theta_pred.shape)
			#self.new = tf.layers.dense(fc_out, 69, activation=tf.nn.sigmoid)
			#self.theta_pred = 8*self.new - 6
			

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
           


            
 


def main():
	model = Model()


if __name__ == '__main__':
    main()
