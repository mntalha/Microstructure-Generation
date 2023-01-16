# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 20:14:03 2022

@author: talha.kilic
"""

import matplotlib.pyplot as plt
from torchviz import make_dot
import torch
from torchvision import transforms
from skimage import filters

def visualize_loss(train_loss, validation_loss,title,img_name):

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=500)

    plt.title(title)
    # plt.title('Loss Graph of VAE Model (Batch + Dropout + KL Divergence)')

    color = 'tab:purple'
    plt.plot(train_loss, color=color)

    color = 'tab:blue'
    x_axis = list(range(0, 1000, 3))
    plt.plot(x_axis, validation_loss, color=color)

    class_names = ["Train", "Validation"]

    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")

    plt.legend(class_names, loc=1)
    plt.show()

    fig.savefig(img_name, dpi=500)


def visualize_model_structure(model):

    #conda install python-graphviz
    # pip install pip torchviz
    
    
    sample = torch.zeros(1,96*96)
    model = model.to("cpu")
    make_dot(model(sample), params=dict(list(model.named_parameters())))

def visualize_microstructure(img_arr):
    """
    Parameters
    ----------
    img_arr : torch.float32 
    
        will receive (x,9) torch.float32 array
        x is arbitrary, can vary
        randomly 9 point will be chosen from x data point
        Finally, will plot microstructure image 
        
    Returns : 
    -------
    None.

    """
    rand_result = int(torch.randint(img_arr.shape[0],(1,)))
    # randomly chosen
    img = img_arr[rand_result]

    # normalize between 0 and 1 
    # img = img.view(img.size(0), -1)
    img -= img.min()
    img /= img.max()
    
    #plot image
    img = img.reshape(96,96).cpu().detach().numpy()
    plt.figure(0)
    val = filters.threshold_otsu(img)
    plt.imshow(img>val, cmap='gray')
    plt.savefig('../figures/microstructure_example.jpg')
    print ("====> image generated!")