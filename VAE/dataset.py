# -*- coding: utf-8 -*-
"""
Created on Sun Dec  12, 2022

@author: talha.kilic
"""

#library imports 

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import h5py


# Macros

# Paths
root_dir = "C:/Users/talha/Desktop/Microstructure-Generation/dataset/dataset_z9.hdf5" 


class MicrostructureDataset(Dataset):
    
    def __init__(self, path):
        
        
        dataset = h5py.File(path, 'r')
        
        self.images = dataset["x"]  # 100x96x96 
        
        self.optical_absorbtion = dataset["y"]  #100x1

        self.latent_vectors = dataset["z"]  # 100x3x3

        
    def __len__(self):
        return len(self.images)  

    def __getitem__(self,idx):
        
        return self.images[idx]
    
    
    def visualize(self, img):
        
        import matplotlib.pyplot as plt
        from skimage import filters

        plt.figure(0)
        gray=img
        val = filters.threshold_otsu(gray)
        plt.imshow(gray>val, cmap='gray')
        print ("====> image!")

        
if __name__ == "__main__":
    
    dataset = MicrostructureDataset(root_dir)
    
    ratio = list(map(int,[0.7*len(dataset), 0.2*len(dataset), 0.1*len(dataset)]))
    train, test ,validation = random_split(dataset, ratio)
    
    trainloader = DataLoader(dataset=dataset,batch_size=4, shuffle=True)
    
 

        
        