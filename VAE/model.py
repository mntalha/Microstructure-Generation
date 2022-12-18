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
dropout_keep_rate=0.50
features = 9
images_size = 96 * 96 
#Paths

class VAEModel(nn.Module):

        def __init__(self):
            super(VAEModel, self).__init__()
            
            # encoder
            self.enc1 = nn.Linear(images_size=images_size, out_features=512)
            self.enc2 = nn.Linear(in_features=512, out_features=features*2)
            
            
            # decoder 
            self.dec1 = nn.Linear(in_features=features, out_features=512)
            self.dec2 = nn.Linear(in_features=512, out_features=images_size)

        def reparameterize(self, mu, log_var):
            
            std = torch.exp(0.5*log_var) 
            eps = torch.randn_like(std) 
            sample = mu + (eps * std)
            
            return sample
            
        def forward(self, x):

            # encoding
            #x = x. flatten operatıon
        
            x = F.relu(self.enc1(x))
            x = self.enc2(x).view(-1, 2, features)
            
            # get `mu` and `log_var`
            mu = x[:, 0, :] # the first feature values as mean
            log_var = x[:, 1, :] # the other feature values as variance

            # get the latent vector through reparameterization
            z = self.reparameterize(mu, log_var)
            
            
            # decoding
            x = F.relu(self.dec1(z))
            reconstruction = torch.sigmoid(self.dec2(x))
            return reconstruction, mu, log_var


if __name__ == "__main__":
    model = VAEModel()