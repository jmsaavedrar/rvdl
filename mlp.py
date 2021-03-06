"""
@jsaavedr    
march, 2022
"""
import pathlib
import sys
sys.path.append(str(pathlib.Path().absolute()))
import tensorflow as tf
import argparse
import models.simple_mlp as mlp_model
import os
import numpy as np

def process_input(fv, lbl, n_classes, _mean):
    #fv = fv - _mean
    lbl = tf.one_hot(tf.cast(lbl, tf.int32), n_classes)
    return fv, lbl
    
if __name__ == '__main__' :
    """
    mode: train test    
    """
    #------- read input parameters
    parser = argparse.ArgumentParser(description = "Train a simple mlp model")        
    parser.add_argument("-mode", type=str, choices=['train', 'val'],  help=" train or val", required = False, default = 'train')    
    parser.add_argument("-dir", type=str,  help=" It's the folder where the data is located", required = True)
    parser.add_argument("-n_classes", type=int,  help=" It's the number of classes", required = True)
    parser.add_argument("-ckp", type=str,  help="checkpoint", required = False)
    pargs = parser.parse_args()
    # defining some required parameters. In the future, we will use a configuration file
    n_classes = pargs.n_classes
    batch_size = 64 
    fv_size = 64
    n_epochs = 50    
    #------- load data    
    x_train_file = os.path.join(pargs.dir, 'train_x.npy')
    x_train = np.load(x_train_file)
    _mean = tf.reduce_mean(x_train, axis = 0)
    if pargs.mode == 'train' :        
        lbl_train_file = os.path.join(pargs.dir, 'train_lbl.npy')                
        lbl_train = np.load(lbl_train_file)
        print('train_x: {}'.format(x_train.shape))
        print('train_lbl: {}'.format(lbl_train.shape))
        tr_dataset = tf.data.Dataset.from_tensor_slices((x_train, lbl_train))
        tr_dataset = tr_dataset.map(lambda x,lbl : process_input(x,lbl, n_classes, _mean));    
        tr_dataset = tr_dataset.shuffle(10000)        
        tr_dataset = tr_dataset.batch(batch_size = batch_size)
            
            
    if pargs.mode == 'val' or  pargs.mode == 'train':
        x_val_file = os.path.join(pargs.dir, 'val_x.npy')
        lbl_val_file = os.path.join(pargs.dir, 'val_lbl.npy')        
        x_val = np.load(x_val_file)
        lbl_val = np.load(lbl_val_file)
        print('val_x: {}'.format(x_val.shape))
        print('val_lbl: {}'.format(lbl_val.shape))
        validation_steps = len(lbl_val) / batch_size
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, lbl_val))        
        val_dataset = val_dataset.map(lambda x,lbl : process_input(x,lbl, n_classes, _mean));
        val_dataset = val_dataset.batch(batch_size = batch_size)
      
    #Creating the model
    
    model = mlp_model.SimpleMLP(n_classes)
    input_shape = (fv_size,)
    _input = tf.keras.Input(input_shape, name = 'input')
    model(_input)
    model.summary()
    #if ckp is present, load the learned weights
    if pargs.ckp is not None :
        model.load_weights(pargs.ckp, by_name = True, skip_mismatch = True)        
    #SGD-based optimizer (Adam) 
    opt = tf.optimizers.Adam()
    model.compile(optimizer = opt, 
                  loss = tf.keras.losses.categorical_crossentropy,
                  metrics = ['accuracy'])
             
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                  filepath=os.path.join(pargs.dir, 'models', '{epoch:03d}.h5'),
                                  save_weights_only=True,                                  
                                  monitor='val_acc',
                                  save_freq = 'epoch')
    #train
    if pargs.mode == 'train' :                             
        history = model.fit(tr_dataset, 
                            epochs = n_epochs,
                            validation_data=val_dataset,
                            validation_steps = validation_steps,
                            callbacks=[model_checkpoint_callback])
    #test
    if pargs.mode == 'val' :
        model.evaluate(val_dataset, steps = validation_steps)
