import numpy as np
import imageio
import math
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from matplotlib import cm

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)




def show_hough_linetransform(img, accumulator, thetas, rhos, Xcalibration, Tcalibration, low_slope_threshold,high_slope_threshold, save_path=None, File = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator, cmap=cm.gray,
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')
    ax[2].imshow(img, cmap=cm.gray)
    for _, angle, dist in zip(*hough_line_peaks(accumulator, thetas, rhos)):
     y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
     y1 = (dist - img.shape[1] * np.cos(angle)) / np.sin(angle)
    
        
     slope =  -( np.cos(angle) / np.sin(angle) )* (Xcalibration / Tcalibration)
    #Draw high slopes
     
     if(np.abs(angle * 180 / 3.14) < high_slope_threshold and np.abs(angle * 180 / 3.14) > low_slope_threshold ):
      print("Estimated Wave Velocity : " ,np.abs(slope))   
    
      ax[2].plot((0, img.shape[1]), (y0, y1), '-r')
    
      ax[2].set_xlim((0, img.shape[1]))
      ax[2].set_ylim((img.shape[0], 0))
      ax[2].set_axis_off()
      ax[2].set_title('Detected lines')

    # plt.axis('off')
    if save_path is not None and File is not None:
        plt.savefig(save_path + 'HoughPlot' + File + '.png')
    if save_path is not None and File is None:
         plt.savefig(save_path + 'HoughPlot' + '.png')
    plt.show()
