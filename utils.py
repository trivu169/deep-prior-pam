import cv2
import numpy as np
from matplotlib import pyplot as plt

def readImg(im, axis1_range=[200, 600], axis2_range=[300, 1400]):
    try:   
        # Assign each channel to the designated var
        im_mask = np.copy(im[:, :, 0])  # Binary image of the mask
        im_masked = np.copy(im[:, :, 1])  # 
        im_gt = np.copy(im[:, :, 2])
        
        # Get the down-sampled image
        im_down = im_masked[im_mask!=0];
        [x_down, y_down] = np.where(im_mask!=0);
        x_down = np.unique(x_down);
        y_down = np.unique(y_down);
        im_down = im_down.reshape(len(x_down), len(y_down))
    except:
        # There will be data that the mask and the ground truth channel swap
        # Reassign each channel to the designated var
        im_mask = np.copy(im[:, :, 2])  # Binary image of the mask
        im_masked = np.copy(im[:, :, 1])  # 
        im_gt = np.copy(im[:, :, 0])
        im_down = im_masked[im_mask!=0];
        [x_down, y_down] = np.where(im_mask!=0);
        x_down = np.unique(x_down);
        y_down = np.unique(y_down);
        im_down = im_down.reshape(len(x_down), len(y_down))       
        im = np.dstack((im_mask, im_masked, im_gt))
        
    
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