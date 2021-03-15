# coding: utf-8
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import concatenate, multiply
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim
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
from build_unet import *

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

def upsample(x, size, inter, filters, transconvo=False):
#     x = reflection_padding(x, (1,1))
    if transconvo:
        x = Conv2DTranspose(filters, kernel_size=5, strides=(2,2), padding='same')(x)
    else:
        x = Conv2D(filters, kernel_size=5, strides=1, padding='same')(x)
    #     x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    
    if inter == "bilinear":
        x_padded = reflection_padding(x, (1, 1))
        x = Lambda(lambda x: (x[:, :-1, 1:, :] + x[:, 1:, :-1, :] + x[:, :-1, :-1, :] + x[:, :-1, :-1, :]) / 4.0)(x_padded)
    return x, [size[0]*2, size[1]*2]

def up_sampling(x, size, filters, kernel_size, inter, transconvo=False):
    x, size = upsample(x, size, inter, filters, transconvo)
    x, size = conv_bn_relu(x, size, filters, kernel_size, (1, 1))
    x, size = conv_bn_relu(x, size, filters, 1, (1, 1))
    return x, size

def skip(x, size, filters, kernel_size):
    x, size = conv_bn_relu(x, size, filters, kernel_size, (1, 1))
    return x, size

def define_model(num_u, num_d, kernel_u, kernel_d, num_s, kernel_s, height, width, inter, 
                 lr, input_channel=32, transconvo=False):
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
#         print(x.shape)
        if num_s[i] != 0:
            skipped, size = skip(down_sampled[i], size, num_s[i], kernel_s[i])
            x = concatenate([x, skipped], axis=3)
        x, size = up_sampling(x, size, num_u[i], kernel_u[i], inter, transconvo)

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
              input_depth=32, transconvo=False):
    img_shape = (height, width, input_depth)
    num_u = [64, 64, 64, 64, 64]
    num_d = [64, 64, 64, 64, 64]
#     num_u = [4, 4, 4, 4, 4]
#     num_d = [4, 4, 4, 4, 4]
    kernel_u = [kernel_size, kernel_size, kernel_size, kernel_size, kernel_size]
    kernel_d = [kernel_size, kernel_size, kernel_size, kernel_size, kernel_size]
    num_s = [4, 4, 4, 4, 4]
    kernel_s = [1, 1, 1, 1, 1]
    lr = 0.1
    inter = None 
    
    """ WARNING: transposed convolution for upsampling appears to cause severe checkerboard artifact: 
        https://distill.pub/2016/deconv-checkerboard/ - avoid using conv2dtranspose in nn """

    base_model = define_model(num_u, num_d, kernel_u, kernel_d, num_s, kernel_s, height, 
                              width, inter, lr, input_channel=input_depth, transconvo=transconvo)
#     base_model = build_unet(img_shape, kernel_size)
    
    mask = Input(shape=(height, width,1))

#     lanczos_kernel = np.zeros((h_factor,l_factor))
#     for i in range(h_factor):
#         for j in range(l_factor):
#             x_d = np.abs(j-1.5)
#             y_d = np.abs(i-1.5)
#             lanczos_kernel[i,j] = np.sinc(x_d) * np.sinc(x_d/2.0) + np.sinc(y_d) * np.sinc(y_d/2.0)
#     lanczos_kernel = lanczos_kernel / lanczos_kernel.sum()

#     x = base_model.output
#     down_sampled = Lambda(lambda x: K.zeros_like(x[:, ::h_factor, 
#                                                    ::l_factor, :]))(x)

# #     for i in range(h_factor):
# #         for j in range(h_factor):
# #             down_sampled = Lambda(lambda x: tf.image.resize_images(x[0][:, i::h_factor, j::l_factor, :] * 
# #                                   lanczos_kernel[i, j], [height_lr, width_lr], method=tf.image.ResizeMethod.BICUBIC) + 
# #                                   x[1])([x, down_sampled])
#     for i in range(h_factor):
#         for j in range(h_factor):
#             down_sampled = Lambda(lambda x: x[0][:, i::h_factor, j::l_factor, :] * lanczos_kernel[i, j] + 
#                                   x[1])([x, down_sampled])

# #     down_sampled = Conv2D(filters=1, kernel_size=9, 
# #                           strides=(h_factor, l_factor), padding='same')(base_model.output)
# #     down_sampled = BatchNormalization()(down_sampled)
# #     down_sampled = Conv2D(filters=1, kernel_size=9, 
# #                           strides=1, padding='same', activation='relu')(down_sampled)

#     down_sampled = Lambda(lambda x: tf.image.resize_images(x, [round(height/h_factor), round(width/l_factor)],
#                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))(base_model.output)
    
    down_sampled = multiply([mask, base_model.output])
#     down_sampled_2 = multiply([mask, base_model.output])

#     print(down_sampled.shape, base_model.output.shape)
    model = Model([base_model.input, mask], down_sampled)
#     model = Model(base_model.input, base_model.output)

# Original
    model.compile(loss='mse', 
                  optimizer=Adam(lr=lr, amsgrad=True, clipvalue=10), 
                  metrics=['mse'])
#     model.summary()
    
#     model.compile(loss='mse', 
#                   optimizer=Adam(lr=lr), 
#                   metrics=['mse'])

    return model, base_model


def train_dp(image, full_sampled, mask, iter=5000, noise_reg = 0.05, show_output=False, im_down=None, transconvo=False, kernel_size=11,
            save_imglog=False, img_path=None):
    input_depth = 32 # check out the paper
    height_lr, width_lr = image.shape[:2]
    height, width = full_sampled.shape[:2]
    h_factor, l_factor = round(height/height_lr), round(width/width_lr)
#     kernel_size = (h_factor, l_factor)
    model, base_model = get_model(height, width, height_lr, width_lr, h_factor, l_factor, 
                                                      kernel_size, input_depth, transconvo)
    if im_down is None:
        input_noise = np.random.uniform(0, 0.1, (1, height, width, input_depth))
    else:
        im_down = im_down.reshape(np.append(np.asarray(im_down.shape), 1))
        input_noise = np.tile(im_down, (1, 1, input_depth))/2550
        input_noise = input_noise[None, :, :, :]
    image = image.reshape(np.append(np.asarray(image.shape), 1))
    mask = mask.reshape(np.append(np.asarray(mask.shape), 1))
    mask = mask/255
    l = []
    ssim_out = []
    psnr_out = []
    
#     print(mask.shape)
    initialTime = time.time()
#     ori_ssim = ssim(np.squeeze(image), np.squeeze(full_sampled))
    for i in range(iter):
        loss = model.train_on_batch([input_noise + 
                                     np.random.normal(0, noise_reg, (height, width,
                                                                     input_depth)), 
                                     mask[None, :, :, :]], 
                                    image[None, :, :, :])
        l.append(loss)
#         if save_imglog and i in np.concatenate((np.arange(0,1000,100), np.arange(1000,5001,500))):
#         if save_imglog and i in np.arange(0,5001,250):
#             test_im = base_model.predict(input_noise)
#             cv2.imwrite(img_path + '_' + str(i) + '_iteration.png', norm_uint8(np.squeeze(test_im)))
        if save_imglog:
            test_im = base_model.predict(input_noise)
            test_ssim = ssim(norm_uint8(np.squeeze(test_im)), norm_uint8(np.squeeze(full_sampled)))
            ssim_out.append(test_ssim)      
#             print(ssim_out)      
            test_psnr = psnr(norm_uint8(np.squeeze(test_im)), norm_uint8(np.squeeze(full_sampled)))
            psnr_out.append(test_psnr)
            
        if i % 500 == 0 and show_output:
            test_im = base_model.predict(input_noise)
            plt.imshow(np.squeeze(test_im))
#             plt.colorbar()
            plt.show()
#             plt.imshow(np.squeeze(full_sampled))
# #             plt.colorbar()
#             plt.show()
            test_ssim = ssim(norm_uint8(np.squeeze(test_im)), norm_uint8(np.squeeze(full_sampled)))
            print(str(i))
            print(test_ssim)
#             print(ori_ssim)
    
    if save_imglog:        
        np.savetxt(img_path + '_SSIMIter.txt', np.asarray([ssim_out]))
        np.savetxt(img_path + '_PSNRIter.txt', np.asarray([psnr_out]))
    sr_image = base_model.predict(input_noise)
    totalTrainingTimeHr = (time.time() - initialTime) / 60
    return sr_image, l, model, totalTrainingTimeHr, input_noise, base_model