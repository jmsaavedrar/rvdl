import pathlib
import sys
sys.path.append(str(pathlib.Path().absolute()))
import utils
import imgproc.pai_io as pai_io
import imgproc.orientation_histograms as oh
import matplotlib.pyplot as plt
import numpy as np
 
if __name__ == '__main__' :
    #filename = '/home/jsaavedr/Research/git/courses/CC5508/images/gray/chair_gray.jpg'
    filename = '/home/vision/smb-datasets/QuickDraw-Animals/test_images/bear/023_00133562.jpg'    
    imageA = pai_io.imread(filename, as_gray = True)
    K = 16
    A, R = oh.compute_local_orientations(imageA,K)
    ys = np.arange(K)
    xs = np.arange(K)
    ys = np.floor(( (ys + 0.5) / K ) * imageA.shape[0])
    xs = np.floor(( (xs + 0.5) / K ) * imageA.shape[1])     
    plt.imshow(imageA, cmap = 'gray', vmax = 255, vmin = 0) 
    plt.quiver(xs, ys, np.cos(A)*R, np.sin(A)*R, angles = 'xy', color = 'r')
    plt.axis('off')         
    plt.show()
    
