"""
SISSI: Simultaneous Image Synthesis and SegmentatIon

Employ a 3D UNet backbone for joint MS lesion segmentation and DIR synthesis from input T1w and FLAIR images

Ideas:
> Use attention loss for synthesis? => Does the segmentation mask come from the gt or network?
> How to weigh the loss terms (Dice/BCE vs. SSIM)?
> Include dropout?
> Have the discriminator also look at the segmentation? => Potentially very interesting (gives segmentation supervision also for cases w/o gt seg); how to handle cases w/ missing gt?
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow_addons as tfa
from glob import glob
import numpy as np
import random
import nibabel as nib
from skimage.exposure import adjust_gamma

"""
HYPERPARAMETERS
"""
input_shape = (192,192,192)
gf = 32 #Number of filters in the generator (3D nnUNet)
df = 32 #Number of filters in the discriminator
dropout_rate = 0.3 #dropout rate
gan_lambda = 1. # Loss: (gan_lambda * Disc_Loss) + (syn_lambda * SSIM_Loss) + (seg_lambda * DiceBCE_Loss)
syn_lambda = 1. # Loss: (gan_lambda * Disc_Loss) + (syn_lambda * SSIM_Loss) + (seg_lambda * DiceBCE_Loss)
seg_lambda = 1. # Loss: (gan_lambda * Disc_Loss) + (syn_lambda * SSIM_Loss) + (seg_lambda * DiceBCE_Loss)
batch_size = 1
epochs = 101
lr = 1e-3

"""
MODELS
"""
def nnunet_3d(n_filters=gf):
	"""
	Returns a 3D nnUNet w/ two outputs (seg and syn)
	"""
	def conv_3d(in_layer,n_filters):
		"""
		Two conv steps per stage with leaky ReLU, instance norm (and dropout)
		Arguments:
			in_layer_: input tensor.
			n_filters: Number of filters
		"""
		layer_ = tf.keras.layers.Conv3D(n_filters, kernel_size=3, strides=1, kernel_initializer='he_uniform', padding='same', use_bias=False) (in_layer)
		layer_ = tfa.layers.InstanceNormalization() (layer_)
		layer_ = tf.keras.layers.LeakyReLU() (layer_)

		layer_ = tf.keras.layers.Conv3D(n_filters, kernel_size=3, strides=1, kernel_initializer='he_uniform', padding='same', use_bias=False) (layer_)
		layer_ = tfa.layers.InstanceNormalization() (layer_)
		layer_ = tf.keras.layers.LeakyReLU() (layer_)

		return layer_

	def upconv_3d(in_layer,n_filters):
		"""
		Performs a stride 2 transposed convolution
		"""
		layer_ = tf.keras.layers.Conv3DTranspose(n_filters, kernel_size=4, strides=2, kernel_initializer='he_uniform', padding='same', use_bias=False) (in_layer)
		layer_ = tfa.layers.InstanceNormalization() (layer_)
		layer_ = tf.keras.layers.LeakyReLU() (layer_)

		return layer_

	def downconv_3d(in_layer,n_filters):
		"""
		Performs a stride 2 downconvolution
		"""
		layer_ = tf.keras.layers.Conv3D(n_filters, kernel_size=4, strides=2, kernel_initializer='he_uniform', padding='same', use_bias=False) (in_layer)
		layer_ = tfa.layers.InstanceNormalization() (layer_)
		layer_ = tf.keras.layers.LeakyReLU() (layer_)

		return layer_

	input_layer = tf.keras.layers.Input((192,192,192,2), name = "input_layer")

	#Encoder block
	conv_1 = conv_3d(input_layer,n_filters)

	conv_2 = downconv_3d(conv_1,n_filters*2)
	conv_2 = conv_3d(conv_2,n_filters*2)

	conv_3 = downconv_3d(conv_2,n_filters*3)
	conv_3 = conv_3d(conv_3,n_filters*3)

	conv_4 = downconv_3d(conv_3,n_filters*4)
	conv_4 = conv_3d(conv_4,n_filters*4)

	conv_5 = downconv_3d(conv_4,n_filters*5)
	conv_5 = conv_3d(conv_5,n_filters*5)

	#Bottle neck
	bn_conv = downconv_3d(conv_5,n_filters*5)
	bn_conv = conv_3d(bn_conv,n_filters*5)

	#Decoder block
	deconv_5 = upconv_3d(bn_conv,n_filters*5)
	deconv_5 = tf.keras.layers.concatenate([conv_5, deconv_5])
	deconv_5 = conv_3d(deconv_5,n_filters*5)

	deconv_4 = upconv_3d(deconv_5,n_filters*4)
	deconv_4 = tf.keras.layers.concatenate([conv_4, deconv_4])
	deconv_4 = conv_3d(deconv_4,n_filters*4)

	deconv_3 = upconv_3d(deconv_4,n_filters*3)
	deconv_3 = tf.keras.layers.concatenate([conv_3, deconv_3])
	deconv_3 = conv_3d(deconv_3,n_filters*3)

	deconv_2 = upconv_3d(deconv_3,n_filters*2)
	deconv_2 = tf.keras.layers.concatenate([conv_2, deconv_2])
	deconv_2 = conv_3d(deconv_2,n_filters*2)

	deconv_1 = upconv_3d(deconv_2,n_filters)
	deconv_1 = tf.keras.layers.concatenate([conv_1, deconv_1])
	deconv_1 = conv_3d(deconv_1,n_filters)

	out_seg = tf.keras.layers.Conv3D(1, 1, 1, activation='sigmoid', kernel_initializer='glorot_uniform', padding='same', name = "out_seg") (deconv_1)
	out_syn = tf.keras.layers.Conv3D(1, 1, 1, activation='relu', kernel_initializer='he_normal', padding='same', name = "out_syn") (deconv_1)

	return tf.keras.Model(input_layer,{"out_seg": out_seg, "out_syn": out_syn})

def discriminator_3d(df=32):
	"""The discriminator (ImageGAN)"""

	def d_layer(layer_input, filters):
		"""Discriminator layer"""

		d_1 = tf.keras.layers.Conv3D(filters, kernel_size=1, strides=1, kernel_initializer='he_uniform', padding='same', use_bias=False)(layer_input)
		d_3 = tf.keras.layers.Conv3D(filters, kernel_size=3, strides=1, kernel_initializer='he_uniform', padding='same', use_bias=False)(layer_input)

		d = tf.keras.layers.Concatenate()([d_1, d_3])
		d = tfa.layers.InstanceNormalization() (d)
		d = tf.keras.layers.LeakyReLU()(d)

		d = tf.keras.layers.Conv3D(filters, kernel_size=4, strides=2, kernel_initializer='he_uniform', padding='same', use_bias=False)(d)
		d = tfa.layers.InstanceNormalization() (d)
		d = tf.keras.layers.LeakyReLU()(d)

		return d

	img_real_input = tf.keras.layers.Input(shape=(192,192,192,2)) #This is the input image to the generator
	img_output = tf.keras.layers.Input(shape=(192,192,192,1)) #This is either real or fake DIR
	#img_output = tf.keras.layers.Input(shape=(192,192,192,2)) #This is either real or fake Seg+DIR
	# Concatenate image and conditioning image by channels to produce input
	combined_imgs = tf.keras.layers.Concatenate(axis=-1)([img_real_input, img_output])

	d1 = d_layer(combined_imgs, df)
	d2 = d_layer(d1, df*2)
	d3 = d_layer(d2, df*3)
	d4 = d_layer(d3, df*4)
	d5 = d_layer(d4, df*5)

	validity = tf.keras.layers.Conv3D(1, kernel_size=1, strides=1, padding='same', activation="linear")(d5)

	return tf.keras.Model([img_real_input, img_output], validity)

"""
LOSSES

> L2 / MSE for the discriminator (LS-GAN)
> SSIM (plus attention loss) for the generator
> Dice-BCE for the segmentation
"""
loss_object = tf.keras.losses.MeanSquaredError()

def segmentation_loss(gt, seg_output):
	gt_f = K.flatten(gt)
	seg_output_f = K.flatten(seg_output)
	intersection_ = K.sum(gt_f * seg_output_f)
	sum_ = K.sum(gt_f) + K.sum(seg_output_f)
	dice_ = (2. * intersection_ + K.epsilon()) / (sum_ + K.epsilon())
	dice_loss_ = 1. - dice_

	return tf.reduce_mean(tf.keras.losses.binary_crossentropy(gt,seg_output)) + dice_loss_, dice_

def generator_loss(disc_generated_output, gen_syn, gen_seg, target, gt):
	gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

	# mean absolute error, w/ attention loss
	syn_loss = 1. - tf.reduce_mean(tf.image.ssim(target,gen_syn,max_val=1))

	if tf.reduce_mean(gt) > 0.: #Cases with gt segmentation receive supervision (Dice+BCE) from the segmentation aswell
		seg_loss, dice = segmentation_loss(gt, gen_seg)
	else:
		seg_loss, dice = 0., 0.

	total_gen_loss = (gan_lambda * gan_loss) + (syn_lambda * syn_loss) + (seg_lambda * seg_loss) 

	return total_gen_loss, gan_loss, syn_loss, seg_loss, dice

def discriminator_loss(disc_real_output, disc_generated_output):
	real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

	generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

	total_disc_loss = real_loss + generated_loss

	return total_disc_loss 

"""
DATA LOADER
"""
def adapt_shape(img_arr):
	"""Crops input image array to target self.shape"""
	difference_0 = img_arr.shape[0] - input_shape[0]
	difference_1 = img_arr.shape[1] - input_shape[1]
	difference_2 = img_arr.shape[2] - input_shape[2]
	img_arr_cropped = img_arr[(difference_0 // 2)+(difference_0 % 2):img_arr.shape[0] - (difference_0 // 2),(difference_1 // 2)+(difference_1 % 2):img_arr.shape[1] - (difference_1 // 2),(difference_2 // 2)+(difference_2 % 2):img_arr.shape[2] - (difference_2 // 2)]
	return img_arr_cropped.astype(np.float32)

def preprocess_image(img_arr):
	"""Clip to [0.1;0.99] and norm to [0;1] (inside brain!)"""
	temp_bm = np.zeros(img_arr.shape)
	temp_bm[img_arr != 0] = 1
	img_arr = np.clip(img_arr, np.percentile(img_arr[temp_bm != 0],1.),np.percentile(img_arr[temp_bm != 0],99.) )
	img_arr -= img_arr[temp_bm == 1].min()
	img_arr = img_arr / img_arr[temp_bm == 1].max()
	img_arr *= temp_bm
	return img_arr.astype(np.float32)

def yield_batch(ids,training=True):
	"""ids is a list of len(batch_size) with file names (FLAIR)"""
	imgs_input = []
	imgs_dir = []
	imgs_gt = []
	for img_path in ids:

		f2 = adapt_shape(nib.load(img_path).get_fdata())
		t1 = adapt_shape(nib.load(img_path.replace("_f2","_t1")).get_fdata())
		d2 = adapt_shape(nib.load(img_path.replace("_f2","_d2")).get_fdata())
		if os.path.exists(img_path.replace("_f2","_seg")):
			gt = adapt_shape(nib.load(img_path.replace("_f2","_seg")).get_fdata())
			gt[gt > 0] = 1 #Make sure segmentation is binary
		else:
			gt = np.zeros(d2.shape,dtype=np.float32) #For cases w/o segmentation, we only use the loss from the translation task

		f2 = preprocess_image(f2)
		t1 = preprocess_image(t1)
		d2 = preprocess_image(d2)

		if training:
			#Morphology augmentations ... applied to all images
			if random.random() > 0.33:
				axis = random.sample((0,1,2),1)
				t1 = np.flip(t1,axis=axis)
				f2 = np.flip(f2,axis=axis)
				d2 = np.flip(d2,axis=axis)
				gt = np.flip(gt,axis=axis) 

			if random.random() > 0.33:
				axis = random.sample((0,1,2),1)
				t1 = np.flip(t1,axis=axis)
				f2 = np.flip(f2,axis=axis)
				d2 = np.flip(d2,axis=axis)
				gt = np.flip(gt,axis=axis)

			#Intensity augmentations ... only for images, not segmentation!
			if random.random() > 0.33:
				gamma_ = random.uniform(0.5,1.5)
				t1 = adjust_gamma(t1,gamma=gamma_)
				gamma_ = random.uniform(0.5,1.5)
				f2 = adjust_gamma(f2,gamma=gamma_)
				gamma_ = random.uniform(0.5,1.5)
				d2 = adjust_gamma(d2,gamma=gamma_)
			
		if random.random() > 0.8: #Channel swaps
			imgs_input.append(np.stack([t1,f2],axis=-1))
		else:
			imgs_input.append(np.stack([f2,t1],axis=-1))
		imgs_dir.append(np.expand_dims(d2,axis=-1))
		imgs_gt.append(np.expand_dims(gt,axis=-1))

	return np.stack(imgs_input,axis=0), np.stack(imgs_dir,axis=0), np.stack(imgs_gt,axis=0)

"""
MAIN
"""
train_samples = glob("/mnt/Drive4/bene/sissi/**/*f2.nii.gz",recursive=True)
imgs_src_test,imgs_target_test,_ = yield_batch([train_samples[-1]],training=False)
train_samples.remove(train_samples[-1])
nib.save(nib.Nifti1Image(imgs_src_test[0,:,:,:,0].astype(np.float32),np.eye(4)),"/home/home/bene/sissi_realFLAIR.nii.gz")
nib.save(nib.Nifti1Image(imgs_target_test[0,:,:,:,0].astype(np.float32),np.eye(4)),"/home/home/bene/sissi_realDIR.nii.gz")

for condition in ["both","SegOnly","SynOnly"]:
	filename = "/home/home/bene/sissi_v2_" + condition

	if condition == "both":
		gan_lambda = 1.
		syn_lambda = 1.
		seg_lambda = 1.
	
	elif condition == "SynOnly":
		gan_lambda = 1.
		syn_lambda = 1.
		seg_lambda = 0.

	elif condition == "SegOnly":
		gan_lambda = 0.
		syn_lambda = 0.
		seg_lambda = 1.
		train_samples = [item for item in train_samples if os.path.exists(item.replace("_f2","_seg"))] #Eliminate all cases w/o segmentation

	"""
	TRAINING FUNCTION (INSIDE OF MAIN TO MAKE LAMBDAS ADAPTABLE)
	"""
	@tf.function
	def train_step(src_image, target, gt):
		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			gen_output = generator(src_image, training=True)

			#Only look at DIR in the discriminator
			disc_real_output = discriminator([src_image, target], training=True)
			disc_generated_output = discriminator([src_image, gen_output["out_syn"]], training=True)

			#Look at both seg + DIR in the discriminator
			#disc_real_output = discriminator([src_image, tf.concat([gt,target],axis=-1)], training=True)
			#disc_generated_output = discriminator([src_image, tf.concat([gen_output["out_seg"],gen_output["out_syn"]],axis=-1)], training=True)

			gen_total_loss, gan_loss, syn_loss, seg_loss, dice = generator_loss(disc_generated_output, gen_output["out_syn"], gen_output["out_seg"], target, gt)
			disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

		generator_gradients = gen_tape.gradient(gen_total_loss,generator.trainable_variables)
		discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

		generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
		discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))
		return gen_total_loss, disc_loss, gan_loss, syn_loss, seg_loss, dice

	discriminator = discriminator_3d(df)
	generator = nnunet_3d(gf)
	generator_optimizer = tf.keras.optimizers.SGD(lr,momentum=0.9,nesterov=True)
	discriminator_optimizer = tf.keras.optimizers.SGD(lr,momentum=0.9,nesterov=True)

	bat_per_epo = int(len(train_samples) / batch_size)
	for epoch in range(epochs):
		random.shuffle(train_samples)
		for batch in range(bat_per_epo):
			ids = train_samples[(batch*batch_size):((batch+1)*batch_size)]
			# select a batch of real samples
			imgs_src,imgs_target,imgs_gt = yield_batch(ids,training=True)
			gen_total_loss, disc_loss, gan_loss, syn_loss, seg_loss, dice = train_step(imgs_src, imgs_target, imgs_gt)
			print("Batch %d/%d / Epoch %d/%d: Gen_Total_loss [%.3f], Disc_Loss [%.3f], Gan_Loss [%.3f], Syn_Loss [%.3f], Seg_Loss [%.3f], Dice [%.3f]" % (batch+1, bat_per_epo, epoch+1, epochs, gen_total_loss, disc_loss, gan_loss, syn_loss, seg_loss, dice))
		#At epoch end, randomly save one example
		tmp_ = generator(imgs_src_test)
		syn_ = np.squeeze(tmp_["out_syn"])
		seg_ = np.squeeze(tmp_["out_seg"])
		nib.save(nib.Nifti1Image(syn_.astype(np.float32),np.eye(4)),filename+"_synDIR.nii.gz")
		nib.save(nib.Nifti1Image(seg_.astype(np.float32),np.eye(4)),filename+"_synSEG.nii.gz")
	generator.save(filename+"_final.h5")