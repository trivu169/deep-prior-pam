import cv2
import numpy as np
from matplotlib import pyplot as plt

def readImg(im, axis1_range=[200, 600], axis2_range=[300, 1400]):
#     im = im[axis1_range[0]:axis1_range[1], axis2_range[0]:axis2_range[1], :]
#     im = im[round(im.shape[0]/2)-150:round(im.shape[0]/2)+150, round(im.shape[1]/2)-150:round(im.shape[1]/2)+150, :]
#     im = im[150:450, 350:650, :]  # For '20190423_thinnedskull_Epi_   2_Image1_index0_pad_3-7'
#     im = im[1000:1300, 1000:1300, :]
#     print(im.shape)
    
    # Assign each channel to the designated var
    im_mask = np.copy(im[:, :, 0])  # Binary image of the mask
    im_masked = np.copy(im[:, :, 1])  # 
    im_gt = np.copy(im[:, :, 2])
    
#     # # Zero-pad fully-sampled image to make sure that it is integer-factor bigger than down sampled image (this is only for 
#     # # original SR method by Satoshi et al.; comment out for our SR method)
#     # h_factor, l_factor = round(im_gt.shape[0]/im_down.shape[0]), round(im_gt.shape[1]/im_down.shape[1])
#     # im_gt_temp = np.zeros((im_down.shape[0]*h_factor, im_down.shape[1]*l_factor))
#     # im_gt_temp[:im_gt.shape[0], :im_gt.shape[1]] = im_gt
#     # im_gt = im_gt_temp

#     # Crop the fully-sampled image with multiplier of 2^4 as the spatial size
#     im_gt = im_gt[:im_gt.shape[0]-(im_gt.shape[0] % (16*mask[0])), 
#                   :im_gt.shape[1]-(im_gt.shape[1] % (16*mask[1]))] # enforce size that contains 2^4 in multiplication 
#                                                         # to prevent trouble with max pooling in the Unet  
#     im_mask = im_mask[:im_gt.shape[0], :im_gt.shape[1]]
#     im_masked = im_masked[:im_gt.shape[0], :im_gt.shape[1]]

#     plt.imshow(im_mask)
#     plt.show()
#     plt.imshow(im_masked)
#     plt.show()
#     plt.imshow(im_gt)
#     plt.show()

    # Get the down-sampled image
    im_down = im_masked[im_mask!=0];
    [x_down, y_down] = np.where(im_mask!=0);
    x_down = np.unique(x_down);
    y_down = np.unique(y_down);
#     print(x_down, y_down)
    im_down = im_down.reshape(len(x_down), len(y_down));
    im_bicubic = cv2.resize(im_down, (im_gt.shape[1], im_gt.shape[0]),
                        interpolation=cv2.INTER_CUBIC)
    im_bilinear = cv2.resize(im_down, (im_gt.shape[1], im_gt.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    im_lanczos = cv2.resize(im_down, (im_gt.shape[1], im_gt.shape[0]),
                        interpolation=cv2.INTER_LANCZOS4)
    
    # Get interpolated factor
    factor = np.divide(im_gt.shape, (len(x_down), len(y_down)))
    factor = np.round(factor)
    factor = factor.astype(int)
    
    return im, im_gt, im_masked, im_mask, im_bicubic, factor, im_bilinear, im_lanczos
    
def constructPlot(data, title, fsize=(10,6), cmax=None):
    """ Construct subplots based on the stacked matrices and its corresponded label 
        NOTE: title[0] is the main title. title[i] is the """
    fig = plt.figure(figsize=fsize)
    for i in range(data.shape[2]):
        ax = fig.add_subplot(1,data.shape[2],i+1)
        plt.imshow(np.rot90(data[:,:,i]), 
                   interpolation='nearest', aspect='auto', cmap='gray')
        plt.axis('off')
        if cmax is not None:
            plt.clim(data[:,:,i].min(), cmax)
#         plt.ylim(ax_length_mm[-1],5)
#         if i == 0:
#             plt.ylabel('Axial Direction (mm)', fontsize=15)
#         plt.xlabel('Lateral Direction (mm)', fontsize=15)
        ax.set_title(title[i+1], fontsize=15)  

def norm_uint8(sr_image):
    temp = np.squeeze(sr_image)  # Normalize corrected image before converting to uint8 to 
                                 # avoid blackdots generation
    temp = temp - np.min(temp)
    temp = temp/np.max(temp)*255
    return temp.astype(np.uint8)