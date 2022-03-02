import tensorflow as tf
import numpy as np

if __name__ == '__main__' :
    a = np.array([[1,2],[3,4]], dtype =  np.float32)
    b = np.array([0, 1], dtype = np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((a,b))
    
    for item in dataset :
        print(item)
    