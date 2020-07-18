import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf 
import utils

def readImg(im, full_img=False, fft=False, axis1_range=[200, 600], axis2_range=[300, 1400]):
    im = im.astype(np.float32)
#     im = im[axis1_range[0]:axis1_range[1], axis2_range[0]:axis2_range[1], :]
    if not full_img:
        im = im[round(im.shape[0]/2)-150:round(im.shape[0]/2)+150, round(im.shape[1]/2)-150:round(im.shape[1]/2)+150, :]
#     im = im[round(im.shape[0]/2)-50:round(im.shape[0]/2)+250, round(im.shape[1]/2)-50:round(im.shape[1]/2)+250, :]
#     print(im.shape)
    
    # Assign each channel to the designated var
    if fft:
        im_mask = np.copy(im[:, :, 0])  # Binary image of the mask
        im_masked = FFT(np.copy(im[:, :, 1]))  # 
        im_gt = FFT(np.copy(im[:, :, 2]))
    else:
        im_mask = np.copy(im[:, :, 0]).astype(np.float32)  # Binary image of the mask
        im_masked = np.copy(im[:, :, 1]).astype(np.float32)  # 
        im_gt = np.copy(im[:, :, 2]).astype(np.float32)
    # Get the down-sampled image
    im_down = im_masked#*im_mask;
    return im, im_gt, im_masked, im_mask, im_down
    
def constructPlot(data, title, fsize=(10,6), cmax=None, log_plot = True):
    """ Construct subplots based on the stacked matrices and its corresponded label 
        NOTE: title[0] is the main title. title[i] is the """
    fig = plt.figure(figsize=fsize)
    for i in range(data.shape[2]):
        ax = fig.add_subplot(1,data.shape[2],i+1)
        plot_data = data[:,:,i]
        #print(type(plot_data))
#         if i > 0 and log_plot:
#             plot_data = np.log10(data[:,:,i])
        plt.imshow(plot_data, cmap='gray')
        plt.axis('off')
        if cmax is not None:
            plt.clim(plot_data.min(), cmax)
#         plt.ylim(ax_length_mm[-1],5)
#         if i == 0:
#             plt.ylabel('Axial Direction (mm)', fontsize=15)
#         plt.xlabel('Lateral Direction (mm)', fontsize=15)
        ax.set_title(title[i+1], fontsize=15)
    plt.show()
    
    fig = plt.figure(figsize=fsize)
    for i in range(data.shape[2]):
        ax = fig.add_subplot(1,data.shape[2],i+1)
        plot_data = data[:,:,i]
        if i > 0 and log_plot:
            plot_data = np.log10(FFT_mag(data[:,:,i]))
        plt.imshow(plot_data, cmap='gray')
        plt.axis('off')
        if cmax is not None:
            plt.clim(plot_data.min(), cmax)
#         plt.ylim(ax_length_mm[-1],5)
#         if i == 0:
#             plt.ylabel('Axial Direction (mm)', fontsize=15)
#         plt.xlabel('Lateral Direction (mm)', fontsize=15)
        ax.set_title(title[i+1], fontsize=15)
    plt.show()

def norm_uint8(sr_image):
    temp = np.squeeze(sr_image)  # Normalize corrected image before converting to uint8 to 
                                 # avoid blackdots generation
    temp = temp - np.min(temp)
    temp = temp/np.max(temp)*255
    return temp.astype(np.uint8)

def FFT_mag(input):
    # FFT Function to be performed for each instance in batch
    real = tf.cast(input, tf.float32)
    imag = tf.zeros_like(tf.cast(input, tf.float32))
    out = tf.abs(tf.signal.fftshift(tf.signal.fft2d(tf.complex(real, imag)[:, :]))).numpy()
    '''
    real = np.cast(input, np.float32)
    imag = np.zeros_like(np.cast(input, np.float32))
    out = np.abs(np.fft.fftshift(np.fft.fft2(np.complex(real, imag)[:, :])))
    '''
    return out
def FFT(input):
    # FFT Function to be performed for each instance in batch
    real = tf.cast(input, tf.float32)
    imag = tf.zeros_like(tf.cast(input, tf.float32))
    '''
    if tf.executing_eagerly():
        out = tf.signal.fftshift(tf.signal.fft2d(tf.complex(real, imag)[:, :])).numpy().astype(np.complex64)
    else:
        out = tf.cast(tf.signal.fftshift(tf.signal.fft2d(tf.complex(real, imag)[:, :])), tf.complex64)
    '''
    out = tf.cast(tf.signal.fftshift(tf.signal.fft2d(tf.complex(real, imag)[:, :])), tf.complex64)
    '''
    real = np.cast(input, np.float32)
    imag = np.zeros_like(np.cast(input, np.float32))
    out = np.abs(np.fft.fftshift(np.fft.fft2(np.complex(real, imag)[:, :])))
    '''
    return out
def FFT_inv(input):
    # FFT Function to be performed for each instance in batch
    #out = tf.signal.ifft2d(tf.signal.ifftshift(tf.cast(input, tf.complex64))).numpy()

    data_in = tf.cast(input, tf.complex64)
    '''
    if tf.executing_eagerly():
        out = tf.abs(tf.signal.ifft2d(tf.signal.ifftshift(data_in))).numpy().astype(np.float32)
    else:
        out = tf.cast(tf.abs(tf.signal.ifft2d(tf.signal.ifftshift(data_in))),tf.float32)
    '''
    out = tf.cast(tf.abs(tf.signal.ifft2d(tf.signal.ifftshift(data_in))),tf.float32)
    return out
def FFT_mod(input):
    return tf.map_fn(FFT, input)
def FFT_inv_mod(input):
    return tf.map_fn(FFT_inv, input)