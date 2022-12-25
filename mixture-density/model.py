"""
Created on Sun Dec 12, 2022
@author: talha.kilic
"""
# Variational AutoEncoder Model by means of PyTorch Support 


#library imports 
import torch.nn as nn
import torch.nn.functional as F
import torch 

# Macros
N_MIXES = 40 # number of mixture gaussion
OUTPUT_DIMS = 4 # output dimension
NUM_SAMPLE = 30 # number of sampled points for each input y
#Paths

class DensityNetwork(nn.Module):

        def __init__(self):
            super(DensityNetwork, self).__init__()
            
            
            self.enc1 = nn.Linear(in_features=1, out_features=16)
            self.batch1=nn.BatchNorm1d(num_features=1) 
            self.relu1 = nn.ReLU()
            
            #concat
            
            self.enc2 = nn.Linear(in_features=17, out_features=16)
            self.batch2=nn.BatchNorm1d(16) 
            self.relu2= nn.ReLU()
            
            
            #concat
            
            self.enc3 = nn.Linear(in_features=33, out_features=16)
            self.batch3=nn.BatchNorm1d(16) 
            self.relu3= nn.ReLU()  
            
            
            #concat
            
            self.enc4 = nn.Linear(in_features=33, out_features=16)
            self.batch4=nn.BatchNorm1d(16) 
            self.relu4= nn.ReLU()
            
            
            
            self.mdn_mus = nn.Linear(in_features = 16, out_features=N_MIXES*OUTPUT_DIMS)
            self.mdn_sigmas = nn.Linear(in_features = 16, out_features=N_MIXES*OUTPUT_DIMS)  #activation=elu_plus_one_plus_epsilon)
            self.mdn_pi = nn.Linear(in_features = 16, out_features=N_MIXES)
            
        def forward(self, x):

            # encoding
            x = x.reshape(-1,1,1)
            y_val = x.clone()
            x = self.enc1(x)
            x = self.batch1(x)            
            x = self.relu1(x)
            
            concat = torch.cat([x,y_val],2)
            x = self.enc2(concat)
            x = self.batch2(x)            
            x = self.relu2(x)
            
            concat = torch.cat(x,concat)
            x = self.enc3(concat)
            x = self.batch3(x)            
            x = self.relu3(x)
            
            
            mdn_mus = self.mdn_mus(x)
            mdn_sigmas = self.mdn_sigmas(x)
            mdn_pi = self.mdn_pi(x)
            
            output = torch.cat([mdn_mus, mdn_sigmas, mdn_pi])
        
            return output


if __name__ == "__main__":
    model = DensityNetwork()
    model.train()
    val = torch.tensor(0.5)
    output = model(val)
    print(model.parameters)