# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 20:14:03 2022

@author: talha.kilic
"""

import matplotlib.pyplot as plt


def loss_visualize(train_loss, validation_loss,img_name):

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=500)

    plt.title('Loss Graph of VAE Model')

    color = 'tab:purple'
    plt.plot(train_loss, color=color)

    color = 'tab:blue'
    x_axis = list(range(0, 1000, 3))
    plt.plot(x_axis, validation_loss, color=color)

    class_names = ["Train", "Validation"]

    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")

    plt.legend(class_names, loc=4)
    plt.show()

    fig.savefig(img_name, dpi=1500)
