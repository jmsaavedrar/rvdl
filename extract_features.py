"""
@author: Jose M. Saavedra
2022
"""
import pathlib
import sys
sys.path.append(str(pathlib.Path().absolute()))
import argparse
import datasets.data as data

    
if __name__ == '__main__' :
    """
    mode: train test    
    """
    #------- read input parameters
    parser = argparse.ArgumentParser(description = "Extract features from images")        
    parser.add_argument("-mode", type=str, choices=['train', 'val'],  help=" train or val", required = False, default = 'train')    
    parser.add_argument("-dir", type=str,  help=" It's the folder where the data is located", required = True)
    pargs = parser.parse_args()            
    #------- load data
    fv_size = 64
    data.extract_features(pargs.dir, file_type = pargs.mode , vector_size = fv_size)
    print(" extract_features  completed!")
    