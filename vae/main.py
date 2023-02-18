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
import torch

def main():
    
    torch.manual_seed(42)
    train_obj = ModelTrain()
    train_obj.GPU_Usage(True)
    print(train_obj.use_gpu)
    train_obj.dataset_load("../dataset/dataset_z9.hdf5")
    
    model = VAEModel()
    learning_rate = 3e-4
    n_epochs =  500
    weight_decay =3e-6 
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    criteria = nn.MSELoss()

    loss_values, model, latent_sample=train_obj.train(model, criteria, n_epochs, optimizer)
    
    train_obj.test__()
    
    return loss_values,model, latent_sample

if __name__ == "__main__":
    
    loss, model, latent_sample = main()
    
    decision = [7] # user decision,added which one it is demand.
    
    if 1 in decision: # 1- Save the model into the defined path
        print("PART 1...............")
        save_pytorch_model(model, model_name="test", saved_path="../saved_model")
    
    if 2 in decision: # 2- Load the model 
        print("PART 2...............")
        saved_model = load_pytorch_model(VAEModel, "../saved_model/test")
    
    if 3 in decision: # 3- Subtract decoder part from saved or normal model 
        print("PART 3...............")
        #Be careful with all tensors to be on the same device
        # decoder.cuda() or decoder.cpu()
        decoder = subtract_decoder(saved_model).cuda()
        
    if 4 in decision: #  4- Generate microstructure image from subtracted part
        print("PART 4...............")
        # latent_variable = torch.tensor([-0.4563,  0.22215,  0.5634427,  0.27211975,  0.12787307,
        #        -0.26973462,  0.74367849,  0.455908,  0.78126333]).reshape(1,9).cuda()
        img_arr = decoder(latent_sample)
    
    if 5 in decision: # 5- plot the generated image
        print("PART 5...............")
        visualize_microstructure(img_arr)
    
    if 6 in decision: # 6- Model structure visualize
        print("PART 6...............")
        visualize_model_structure(model)

    if 7 in decision: # 7- Loss plot and saved 
        print("PART 7...............")
        img_name = "../outputs/loss_visualize.png"
        # title = "Loss Graph of VAE Model (Batch + Dropout + KL Divergence)"
        # title = "Loss Graph of VAE Model (Dense Layer + Dense Layer (Added Batch + Dropout))"
        title = "---"
        visualize_loss(loss["train_every_epoch"], loss["validation_every_epoch"],title, img_name)


