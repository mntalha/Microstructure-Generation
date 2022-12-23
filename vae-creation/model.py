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


        def reparameterize(self, mu, log_var):
            
            std = torch.exp(0.5*log_var) 
            eps = torch.randn_like(std) 
            sample = mu + (eps * std)
            
            return sample
            
        def forward(self, x):

            # encoding
            x = F.relu(self.enc1(x))
            x = self.batch1(x)
            x = self.dropout1(x)
            
            x = self.enc2(x)
            x = self.batch2(x)
            x = self.dropout2(x)
            x = x.view(-1, 2, features)
            
            # get `mu` and `log_var`
            mu = x[:, 0, :] # the first feature values as mean
            log_var = x[:, 1, :] # the other feature values as variance

            # get the latent vector through reparameterization
            z = self.reparameterize(mu, log_var)
            
            
            # decoding
            x = F.relu(self.dec1(z))
            
            x = self.batch3(x)
            x = self.dropout3(x)
            
            
            x = torch.sigmoid(self.dec2(x))
            reconstruction = self.batch4(x)

            return reconstruction, mu, log_var, z


if __name__ == "__main__":
    model = VAEModel()
    print(model.parameters)