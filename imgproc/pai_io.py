'''
Created on Aug 1, 2019

@author: jsaavedr
io functions
'''
import skimage.io as skio
import numpy as np

def imread(filename, as_gray = False):
    image = skio.imread(filename, as_gray = as_gray)
    if image.dtype == np.float64 :
        image = to_uint8(image)
    return image

def to_uint8(image) :
    if image.dtype == np.float64 :
        image = image * 255
    image[image<0]=0
    image[image>255]=255
    image = image.astype(np.uint8, copy=False)
    return image