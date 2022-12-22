# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:05:12 2022

@author: talha.kilic
"""

"""
Supportive functions for the AI models will be added in this file 

"""

#library imports 
import torch 
import os 
import torch.nn.functional as F
import torch.nn as nn

def save_pytorch_model(model, model_name, saved_path):
    
    # Check the planned path whether it is exist 
    isExist = os.path.exists(saved_path)
    if not isExist:
        print("Path you wished the model to be saved is not valid...")
        return False
    
    # model.state_dict():
        
    # save it
    path = os.path.join(saved_path, model_name)   
    torch.save(model.state_dict(), path)

def load_pytorch_model(Model_Class, path):   
    """
    take the transform .cuda() and .cpu() into consideration.
    """
    # Check the file whether it is exist
    isExist = os.path.isfile(path)
    if not isExist:
        print("model couldnt found...")
        return False
    
    # create raw model
    raw_model = Model_Class()
    
    # load it
    raw_model.load_state_dict(torch.load(path))
    raw_model.eval()
    return raw_model

def check_model_training(model):
    """
    Check the model if it is in the training or eval model
    
    True: Training Mode
    False: Eval Mode
    """
    return model.training


def subtract_decoder(vae_model):
    
    #Decoder part in VAE model

    extracted = nn.Sequential(
                vae_model.dec1,
                nn.ReLU(),
                vae_model.batch3,
                vae_model.dropout3,
                vae_model.dec2,
                nn.Sigmoid(),
                vae_model.batch4
                )
    # Evaluation mode
    extracted.eval()
    
    return extracted
    
