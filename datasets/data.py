"""
@author: jsaavedr
"""
import os
import skimage.io as io
import skimage.color as color
import imgproc.pai_io as pai_io
import numpy as np
import imgproc.orientation_histograms as oh
import random


def read_image(filename, number_of_channels):
    """ It reads an image using skimage. The output is a gray-scale image [H, W, C]
    """        
    if number_of_channels  == 1 :            
        image = io.imread(filename, as_gray = True)
        image = pai_io.to_uint8(image)                                    
    elif number_of_channels == 3 :
        image = io.imread(filename)
        if(len(image.shape) == 2) :
            image = color.gray2rgb(image)
        elif image.shape[2] == 4 :
            image = color.rgba2rgb(image) 
        image = pai_io.to_uint8(image)
            
    else:
        raise ValueError("number_of_channels must be 1 or 3")
    
    assert len(image.shape) == 2, ' image should be in a grayscale format '
    assert os.path.exists(filename), ' {} does not exist'.format(filename)
    return image

def read_data_from_file(datafile, shuf = True):    
    """read data from text files
    and apply shuffle by default 
    """                
    assert os.path.exists(datafile)        
    # reading data from files, line by line
    with open(datafile) as file :        
        lines = [line.rstrip() for line in file]     
        if shuf:
            random.shuffle(lines)
        _lines = [tuple(line.rstrip().split('\t'))  for line in lines ] 
        filenames, labels = zip(*_lines)
    return filenames, labels

def extract_features(data_dir, file_type, vector_size):
    file = os.path.join(data_dir, file_type) + '.txt'
    print(file)    
    x_file = os.path.join(data_dir, file_type) + '_x.npy'
    lbl_file = os.path.join(data_dir, file_type) + '_lbl.npy'    
    
    #first let's go with training data
    filenames, lbl = read_data_from_file(file)
    fvs = np.zeros((len(filenames), vector_size))
    K = 16
    for i, filename in enumerate(filenames) :
        if (i % 100 == 0) :
            print('{} / {}'.format(i, len(filenames)))
        image = read_image(filename, 1)
        A, R = oh.compute_local_orientations(image, K)
        fvs[i] = oh.compute_histogram(A, R, vector_size)    
    x = fvs.astype(np.float32)
    y = np.array(lbl, dtype = np.float32)
    np.save(x_file, x)
    print('data saved at {}'.format(x_file))
    np.save(lbl_file, y)
    print('labels saved at {}'.format(lbl_file))
    
    
if __name__ == '__main__' :
    data_dir = '/home/vision/smb-datasets/SBIR/QuickDraw-Animals'
    ff= '/home/vision/smb-datasets/SBIR/QuickDraw-Animals/train100_x.npy'
    a = np.load(ff)
    print(a.shape)    
    print(np.linalg.norm(a, ord=2, axis = 1).shape)
    #extract_features(data_dir, file_type = 'train100', vector_size = 32)