# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 20:13:04 2022

@author: talha.kilic
"""
# Written Classes
from dataset import MicrostructureDataset

#libraries
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split

import torch
import time

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
    latent_sample = None

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
        
    
    def test__(self):
        
        #TEST
        print("---------------------TEST PROCESS WITHOUT KL------------------------------")
        #Test Values
        test_loss = 0.0
        test_acc = 0.0

        #Model in Evaluatıon mode, no changes in models parameters        
        self.model.eval()
        
        with torch.no_grad():
            for idx, img in enumerate(self.test_loader):
                
                img = img.cuda()
                img =  img.to(torch.float32)

                y_pred, mean, log_var, latent_sample = self.model(img)
            
                loss = self.criteria(y_pred, img)
                


                
                #On each batch it sum up.
                test_loss += loss.item()* img.size(0)
        
        #Epoch losses and accuracy
        test_loss = test_loss / (len(self.test_loader.sampler))
        
        print("test_loss= "+ str(test_loss))
        
    def test(self,device):
        
        #TEST
        print("---------------------TEST PROCESS------------------------------")
        #Test Values
        test_loss = 0.0
        test_acc = 0.0

        #Model in Evaluatıon mode, no changes in models parameters        
        self.model.eval()
        
        with torch.no_grad():
            for idx, img in enumerate(self.test_loader):
                
                img = img.to(device)
                img =  img.to(torch.float32)

                y_pred, mean, log_var, latent_sample = self.model(img)
            
                loss = self.criteria(y_pred, img)
                
                kl_loss = self.model.kl_loss(latent_sample,mean,log_var)
                
                loss += kl_loss

                
                #On each batch it sum up.
                test_loss += loss.item()* img.size(0)
        
        #Epoch losses and accuracy
        test_loss = test_loss / (len(self.test_loader.sampler))
        
        print("test_loss= "+ str(test_loss))
        
        
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
            
            print(f" ********** { {epoch} } EPOCH")
            
            # make it 0 in each epoch
            train_loss = 0.0
            valid_loss = 0.0
            train_acc = 0.0
            valid_acc = 0.0

             #start train gradient calculation active
            self.model.train()

            for idx, img in enumerate(self.train_loader):
                
                
                img = img.to(device)
                img =  img.to(torch.float32)

 
                #gradient refresh, makes the general faster
                for param in self.model.parameters():
                    param.grad = None
                     
                    
                y_pred, mean, log_var, latent_sample = self.model(img)

                loss = self.criteria(y_pred, img)
                
                kl_loss = self.model.kl_loss(latent_sample,mean,log_var)
                
                loss += kl_loss
                    
                loss.backward()
                    
                self.optimizer.step()

                #On each batch it sum up.
                train_loss += loss.item()* img.size(0)
                    
                self.loss_values['train_every_iteration'].append(loss.item()* img.size(0))
            
            #Epoch losses and accuracy
            train_loss = train_loss / len(self.train_loader.sampler)
            self.loss_values['train_every_epoch'].append(train_loss)
            
            if epoch % 3 == 0:
                
                #start evaluation gradient calculation passive
                self.model.eval()

                # doesn't #turn off Dropout and BatchNorm.                 
                with torch.no_grad():
                    
                    # Measure the performance in validation set.
                    for idx2, img2 in enumerate(self.validation_loader):
                        
                        img2 = img2.to(device)
                        #x = x. flatten operatıon
                        # img2 = torch.flatten(img2, start_dim=1)   
                        img2 = img2.to(torch.float32)
                        

                        y_pred, mu, log_var,latent_sample = self.model(img2)

                        loss = self.criteria(y_pred, img2)
                        
                        kl_loss = self.model.kl_loss(latent_sample,mu,log_var)
                        
                        loss += kl_loss

                        #On each batch it sum up.
                        valid_loss += loss.item()* img2.size(0)
                        
                        self.loss_values['validation_every_iteration'].append(loss.item()* img2.size(0))

                #Epoch losses and accuracy
                valid_loss = valid_loss / (len(self.validation_loader.sampler))
                self.loss_values['validation_every_epoch'].append(valid_loss)
            

        end = time.time()
        
        print('Total Elapsed time is %f seconds.' % (end - start))
        
        #Test Result
        self.test(device)

        return self.loss_values,self.model, latent_sample

                        

                    
                    
                    



                        
