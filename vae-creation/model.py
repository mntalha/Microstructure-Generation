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
dropout_keep_rate= 0.90
features = 9
images_size = 96 * 96 
#Paths

class VAEModel(nn.Module):

        def __init__(self):
            super(VAEModel, self).__init__()
            
            # encoder
            self.enc1 = nn.Linear(in_features=images_size, out_features=512)
            self.enc2 = nn.Linear(in_features=512, out_features=features*2)
            
            # decoder 
            self.dec1 = nn.Linear(in_features=features, out_features=512)
            self.dec2 = nn.Linear(in_features=512, out_features=images_size)
            
            # dropout 
            self.dropout1 = nn.Dropout1d(1-dropout_keep_rate)
            self.dropout2 = nn.Dropout1d(1-dropout_keep_rate)
            self.dropout3 = nn.Dropout1d(1-dropout_keep_rate)
            
            # batch 
            self.batch1=nn.BatchNorm1d(512) 
            self.batch2=nn.BatchNorm1d(features*2) 
            self.batch3=nn.BatchNorm1d(512) 
            self.batch4=nn.BatchNorm1d(images_size) 


        def reparameterize(self, mean, log_var):
            
            epsilon = torch.randn_like(mean) 
            
            # z = mean + e*var
            return epsilon * torch.exp(log_var* 0.5) + mean 
        
        
        def gaussian_loss_fnc(self,log_var,mean):
            
            return (log_var**2 + mean**2 - torch.log(log_var) - 1/2).sum()

            
        def forward(self, x):

            x = torch.flatten(x, start_dim=1)   
            # encoding
            x = F.relu(self.enc1(x))
            x = self.batch1(x)
            x = self.dropout1(x)
            
            x = self.enc2(x)
            x = self.batch2(x)
            x = self.dropout2(x)
            x = x.view(-1, 2, features)
            
           
            mean = x[:, 0, :] # as  mean
            log_var = x[:, 1, :] #  as variance

            # get the latent vector through reparameterization
            z = self.reparameterize(mean, log_var)
            
            
            # decoding
            x = F.relu(self.dec1(z))
            
            x = self.batch3(x)
            x = self.dropout3(x)
            
            
            x = torch.sigmoid(self.dec2(x))
            x = self.batch4(x)
            x = x.view(-1, 96, 96)

            return x, mean, log_var, z


if __name__ == "__main__":
    model = VAEModel()
    print(model.parameters)