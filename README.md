# Pose-Encoder
Encoding pose parameters for SMPL body model from 72 to 32 dimensions and capable of pose interpolations in latent space and generating new poses on random noise.
vis_pose.ipynb contains code to generate the visualisation of  results. Pose_interpolation.png contains grid interpolation between 2 poses in the latent space and random_sampling.png shows results of random sampling on the decoder. All this is done using a cyclic gan loss and this can be used as an embedding for SMPL pose paramters useful in 3D ose estimation. It contains encoder and decoder both in hierarchial and non-hierarchial fashion along with analysing both VAEs and GANS in latent space. 



