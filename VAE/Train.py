# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 20:13:04 2022

@author: talha.kilic
"""
# Written Classes
from dataset import MicrostructureDataset
from Model import VAEModel

#libraries
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split

import torch
import time
import math
import numpy as np
import torch.optim as optim

class ModelTrain:

    # values 
    
    loss_values = {
        'train_every_iteration' : [] ,
        'train_every_epoch' : [] ,
        'validation_every_iteration' : [] ,
        'validation_every_epoch' : []
        }
    
    accuracy_values = {
       'train_every_iteration' : [] ,
       'train_every_epoch' : [] ,
       'validation_every_iteration' : [] ,
       'validation_every_epoch' : []
       }
    
    
    use_gpu = None
    model = None
    criteria = None
    optimizer = None
    batch_size = None
    epoch = None
    dataset = None
    loader = None
    mydataset = None
    
    def __init__(self):
        pass
    
    
    def dataset_load(self,path):
        
        dataset = MicrostructureDataset(path)
        
        #split the dataset 
        ratio = list(map(int,[0.7*len(dataset), 0.2*len(dataset), 0.1*len(dataset)]))
        
        
        train, test, validation = random_split(dataset, ratio)
        
        #Train Loader
        self.train_loader = DataLoader(dataset=train,batch_size=4, shuffle=False)
        self.test_loader = DataLoader(dataset=test,batch_size=4,shuffle=False)
        self.validation_loader = DataLoader(dataset=validation,batch_size=4,shuffle=False)

        
    def GPU_Usage(self,use=False):
        
        self.use_gpu=use
        
    def test(self):
        
        #TEST
        print("---------------------TEST PROCESS------------------------------")
        #Test Values
        test_loss = 0.0
        test_acc = 0.0

        #Model in Evaluatıon mode, no changes in models parameters        
        self.model.eval()
        
        with torch.no_grad():
            for idx, img in enumerate(self.test_loader):
                
                # if self.use_gpu:
                    #     img = img.cuda()

                y_pred = self.model(img)
            
                #Validation Loss calculation part
                loss = self.criteria(y_pred, img)
                
                #On each batch it sum up.
                test_loss += loss.item()* img.size(0)

    def train(self,model_,criteria_,n_epoch_,optimizer_):
        
        
        #TEST
        print("---------------------TRAIN PROCESS------------------------------")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True 

        if self.use_gpu == True and device.type == "cpu":
            print("GPU is not supported --->")
            
        self.model = model_
        self.criteria = criteria_
        self.epoch = n_epoch_
        self.optimizer = optimizer_
        
        #Model to cpu or cuda(GPU)
        self.model.to(device)
        
        start = time.time()

        for epoch in range(self.epoch):
            
            # make it 0 in each epoch
            train_loss = 0.0
            valid_loss = 0.0
            train_acc = 0.0
            valid_acc = 0.0

             #start train gradient calculation active
            self.model.train()

            for idx, img in enumerate(self.loader):
                 
                img = img.to(device)
                 
                #gradient refresh, makes the general faster
                for param in self.model.parameters():
                    param.grad = None
                     
                with torch.cuda.amp.autocast():
                    y_pred = self.model(img)
                    
                    # output is float16 because linear layers autocast to float16.
                    assert y_pred.dtype is torch.float16
                    
                    #Training Loss calculation part
                    loss = self.criteria(y_pred, img)
                    
                    # loss is float32 because mse_loss layers autocast to float32.
                    assert loss.dtype is torch.float32
                    loss.backward()
                    
                self.optimizer.step()

                #On each batch it sum up.
                train_loss += loss.item()* img.size(0)
                    
                                
                        
