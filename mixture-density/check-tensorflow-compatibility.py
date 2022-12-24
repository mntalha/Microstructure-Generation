# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 19:30:25 2022

@author: talha.kilic
"""

# pip install tensorflow-gpu==2.3.0  # GPU
# python version is 3.7
# Cuda 10.1 and cuDNN 7.6

# conda create -n tensorflow_gpu python=3.7
# conda activate tensorflow_gpu
# pip install tensorflow-probability==0.11
# CloudPickle == 1.3



# h5py version error  Headers are 1.10.4, library is 1.10.6
# conda install -c conda-forge hdf5=1.10.6
# pip install matplotlib

   

import tensorflow as tf   # TensorFlow registers PluggableDevices here.
tf.config.list_physical_devices('GPU')

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

print(tf.test.is_gpu_available())
print(tf.test.is_built_with_cuda())

from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices()) 

  

