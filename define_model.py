# coding: utf-8
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import concatenate, multiply
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse
from utils import *
import PIL
import tensorflow as tf
import numpy as np
import os
from keras.optimizers import Adam
import cv2
import h5py
from keras import backend as K
import time

def reflection_padding(x, padding):
    reflected = Lambda(lambda x: x[:, :, ::-1, :])(x)
    reflected = Lambda(lambda x: x[:, :, :padding[1], :])(reflected)
    upper_row = concatenate([x, reflected], axis=2)
    lower_row = Lambda(lambda x: x[:, ::-1, :, :])(upper_row)
    lower_row = Lambda(lambda x: x[:, :padding[0], :, :])(lower_row)
    padded = concatenate([upper_row, lower_row], axis=1)
    return padded

def conv_bn_relu(x, size, filters, kernel_size, strides):
    padding = [0, 0]
    padding[0] =  (int(size[0]/strides[0]) - 1) * strides[0] + kernel_size - size[0]
    padding[1] =  (int(size[1]/strides[1]) - 1) * strides[1] + kernel_size - size[1]
    x = reflection_padding(x, padding)

    x = Conv2D(filters, kernel_size, strides=strides)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    new_size = [int(size[0]/strides[0]), int(size[1]/strides[1])]
    return x, new_size

def down_sampling(x, size, filters, kernel_size):
    new_size = [size[0], size[1]]
    if size[0] % 2 != 0:
#         x = reflection_padding(x, [np.int32(np.floor(kernel_size/2)), 0])
        x = reflection_padding(x, [1, 0])
        new_size[0] = size[0] + 1
    if size[1] % 2 != 0:
#         x = reflection_padding(x, [0, np.int32(np.floor(kernel_size/2))])
        x = reflection_padding(x, [0, 1])
        new_size[1] = size[1] + 1
    size = new_size
    x, size = conv_bn_relu(x, size, filters, kernel_size, (2, 2))
    x, size = conv_bn_relu(x, size, filters, kernel_size, (1, 1))
    return x, size

def upsample(x, size, inter, filters):
#     x = reflection_padding(x, (1,1))
    x = Conv2D(filters, kernel_size=5, strides=1, padding='same')(x)
#     x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
#     x = Lambda(lambda x: tf.image.resize_images(x, [size[0]*2, size[1]*2], method=tf.image.ResizeMethod.BICUBIC, align_corners=False))(x)
    if inter == "bilinear":
        x_padded = reflection_padding(x, (1, 1))
        x = Lambda(lambda x: (x[:, :-1, 1:, :] + x[:, 1:, :-1, :] + x[:, :-1, :-1, :] + x[:, :-1, :-1, :]) / 4.0)(x_padded)
    return x, [size[0]*2, size[1]*2]

def up_sampling(x, size, filters, kernel_size, inter):
    x, size = upsample(x, size, inter, filters)
    x, size = conv_bn_relu(x, size, filters, kernel_size, (1, 1))
    x, size = conv_bn_relu(x, size, filters, 1, (1, 1))
    return x, size

def skip(x, size, filters, kernel_size):
    x, size = conv_bn_relu(x, size, filters, kernel_size, (1, 1))
    return x, size

def define_model(num_u, num_d, kernel_u, kernel_d, num_s, kernel_s, height, width, inter, 
                 lr, input_channel=32):
    depth = len(num_u)
    size = [height, width]

    inputs = Input(shape=(height, width, input_channel))

    x = inputs
    down_sampled = []
    sizes = [size]
    for i in range(depth):
        x, size = down_sampling(x, size, num_d[i], kernel_d[i])
        down_sampled.append(x)
        sizes.append(size)

    for i in range(depth-1, -1, -1):
        if num_s[i] != 0:
            skipped, size = skip(down_sampled[i], size, num_s[i], kernel_s[i])
            x = concatenate([x, skipped], axis=3)
        x, size = up_sampling(x, size, num_u[i], kernel_u[i], inter)

        if sizes[i] != size:
            x = Lambda(lambda x: x[:, :sizes[i][0], :sizes[i][1], :])(x)
            size = sizes[i]

    x = Conv2D(1, 1)(x)
    x = LeakyReLU(alpha=0.2)(x)
    model = Model(inputs, x)

    return model

def ssim_loss(y_true, y_pred):
#     print(y_true.shape)
    return 255*tf.reduce_mean(tf.image.ssim(y_pred, y_true, max_val=255))

def mse_tv(model_output, lambda_tv=0.0000005):
    """ lambda_tv follows the suggested value in DeepRED paper """
    
    def loss(y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred) + lambda_tv*tf.reduce_mean(tf.image.total_variation(model_output))
    return loss

def get_model(height, width, height_lr, width_lr, h_factor, l_factor, kernel_size, 
              input_depth=32):
    img_shape = (height, width, input_depth)
    num_u = [64, 64, 64, 64, 64]
    num_d = [64, 64, 64, 64, 64]
    kernel_u = [11, 11, 11, 11, 11]
    kernel_d = [11, 11, 11, 11, 11]
    num_s = [4, 4, 4, 4, 4]
    kernel_s = [1, 1, 1, 1, 1]
    lr = 0.1
    inter = None 
    
    """ WARNING: transposed convolution for upsampling appears to cause severe checkerboard artifact: 
        https://distill.pub/2016/deconv-checkerboard/ - avoid using conv2dtranspose in nn """

    base_model = define_model(num_u, num_d, kernel_u, kernel_d, num_s, kernel_s, height, 
                              width, inter, lr)
    mask = Input(shape=(height, width,1))

    down_sampled = Lambda(apply_mask)([base_model.output, mask])
    model = Model([base_model.input, mask], down_sampled)

    # Original
    model.compile(loss='mse', 
                  optimizer=Adam(lr=lr, amsgrad=True, clipvalue=10), 
                  metrics=['mse'])
    return model, base_model

def apply_mask(fn_input):
    model_out = fn_input[0]
    mask = fn_input[1]
    fft_out = FFT_mod(model_out[...,0])
    down = multiply([tf.cast(mask[...,0], tf.complex64), fft_out])
    down_sampled = FFT_inv_mod(down)
    down_sampled = down_sampled[...,None]
    return down_sampled
def MSE_inv_fft(y_true, y_pred):
    y_true_inv = FFT_inv(y_true)
    y_pred_inv = FFT_inv(y_pred)
    if tf.executing_eagerly():
        output = tf.keras.losses.MeanSquaredError()(y_true_inv, y_pred_inv).numpy()
    else:
        output = tf.keras.losses.MeanSquaredError()(y_true_inv, y_pred_inv)
    return output
def train_dp(image, full_sampled, mask, iter=5000, noise_reg = 0.05, show_output=False, im_down=None):
    input_depth = 32 # check out the paper
    height_lr, width_lr = image.shape[:2]
    height, width = full_sampled.shape[:2]
    h_factor, l_factor = round(height/height_lr), round(width/width_lr)
#     kernel_size = (h_factor, l_factor)
    kernel_size = 3
    model, base_model = get_model(height, width, height_lr, width_lr, h_factor, l_factor, 
                                  kernel_size, input_depth)
    if im_down is None:
        input_noise = np.random.uniform(0, 0.1, (1, height, width, input_depth))
    else:
        im_down = im_down.reshape(np.append(np.asarray(im_down.shape), 1))
        input_noise = np.tile(im_down, (1, 1, input_depth))/255
        input_noise = input_noise[None, :, :, :]
    #image = image.reshape(np.append(np.asarray(image.shape), 1))
    #mask = mask.reshape(np.append(np.asarray(mask.shape), 1))
    mask = mask/np.max(mask)
    l = []
    
#     print(mask.shape)
    initialTime = time.time()
#     ori_ssim = ssim(np.squeeze(image), np.squeeze(full_sampled))
    for i in range(iter):
        '''
        real_gauss_noise = np.random.normal(0, noise_reg, (height, width, input_depth)).astype(np.float32)
        imag_gauss_noise = (np.random.normal(0, noise_reg, (height, width, input_depth))*1j).astype(np.complex64)
        complex_gauss_noise = (real_gauss_noise + imag_gauss_noise).astype(np.complex64)
        func_input = (input_noise + complex_gauss_noise).astype(np.complex64)
        '''
        func_input = np.random.normal(0, noise_reg, (height, width, input_depth)).astype(np.float32)
        loss = model.train_on_batch([func_input, 
                                     mask[None, :, :, None]], 
                                     image[None, :, :, None])
        l.append(loss)
        if i % 500 == 0 and show_output:
            test_im = np.squeeze(base_model.predict(input_noise))
            pred = FFT_inv(test_im)
            plt.imshow(pred, cmap='gray')
#             plt.colorbar()
            plt.show()
#             plt.imshow(np.squeeze(full_sampled))
# #             plt.colorbar()
#             plt.show()
            test_ssim = ssim(norm_uint8(pred), norm_uint8(im_down))
            print(f'Epoch {str(i)}')
            print(test_ssim)
#             print(ori_ssim)
    
    sr_image = base_model.predict(input_noise)
    totalTrainingTimeHr = (time.time() - initialTime) / 60
    return sr_image, l, model, totalTrainingTimeHr, input_noise