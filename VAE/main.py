# -*- coding: utf-8 -*-
"""
Created on Sun Dec  12, 2022

@author: talha.kilic
"""

# Written Classes
from dataset import MicrostructureDataset
from Model import VAEModel
from Train import ModelTrain

#Libraries
import torch.nn as nn
import torch.optim as optim





def main():
    
    
    train_obj = ModelTrain()
    train_obj.GPU_Usage(True)
    print(train_obj.use_gpu)
    
    
    model = VAEModel()
    learning_rate = 3e-7
    n_epochs =  3
    weight_decay =3e-6 
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    criteria = nn.MSELoss()



    loss_values,accuracy_values=my.train(loaded_model, criteria, n_epochs, optimizer)

if __name__ == "__main__":
    main()