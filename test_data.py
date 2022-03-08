import numpy as np
import os

if __name__ == '__main__' :
    #_dir = '/home/vision/smb-datasets/MNIST/MNIST-5000'
    _dir = '/home/vision/smb-datasets/PrintedSymbols'
    train_data_file = os.path.join(_dir, 'train_x.npy')
    train_data = np.load(train_data_file)
    n_rows, n_cols = train_data.shape
    print('{} {}'.format(n_rows, n_cols))
    
    for i in np.arange(n_rows) :
        #print(np.linalg.norm(train_data[i]))
        print(train_data[i])
