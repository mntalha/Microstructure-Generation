# -*- coding: utf-8 -*-
"""
Created on Sun Dec  12, 2022

@author: talha.kilic
"""

# Written Classes
from dataset import MicrostructureDataset
from model import VAEModel
from train import ModelTrain
from visualize import visualize_loss, visualize_model_structure
from utils import save_pytorch_model, load_pytorch_model,subtract_decoder
from visualize import visualize_microstructure
#Libraries
import torch.nn as nn
import torch.optim as optim


def main():
    
    
    train_obj = ModelTrain()
    train_obj.GPU_Usage(True)
    print(train_obj.use_gpu)
    train_obj.dataset_load("../dataset/dataset_z9.hdf5")
    
    model = VAEModel()
    learning_rate = 3e-3
    n_epochs =  5
    weight_decay =3e-6 
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    criteria = nn.MSELoss()

    loss_values, model, latent_sample=train_obj.train(model, criteria, n_epochs, optimizer)
    
    return loss_values,model, latent_sample

if __name__ == "__main__":
    
    loss, model, latent_sample = main()
    save_pytorch_model(model, model_name="test", saved_path="../saved_model")
    
    tt = load_pytorch_model(VAEModel, "../saved_model/test")
    
    # Be careful with all tensors to be on the same device
    decoder = subtract_decoder(tt)
    
    img_arr = decoder.cuda()(latent_sample)
    
    visualize_microstructure(img_arr)
    # img_name = "C:/Users/talha/Desktop/Microstructure-Generation/figures/loss_visualize_3.png"
    # loss_visualize(loss["train_every_epoch"], loss["validation_every_epoch"],img_name)
    # model_structure_visualize(model)

