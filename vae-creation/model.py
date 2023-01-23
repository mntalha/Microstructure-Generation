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
dropout_keep_rate= 0.66
features = 9
images_size = 96
#Paths
class VAEModel(nn.Module):
    
        #List of Modules
        env = []
        dropout = []
        batch = []
        
        
        dec = []
        dec_drop = []
        dec_batch = []

        def __init__(self):
            super(VAEModel, self).__init__()
            
            self.env1 = nn.Linear(in_features=images_size*images_size, out_features=256)
            self.env.append(self.env1)
            self.env2 = nn.Linear(in_features=256, out_features=64)
            self.env.append(self.env2)
            self.env3 =nn.Linear(in_features=64, out_features=features*2)
            self.env.append(self.env3)        

            self.dropout1 = nn.Dropout2d(1-dropout_keep_rate)
            self.dropout.append(self.dropout1)       
            self.dropout2 = nn.Dropout2d(1-dropout_keep_rate)
            self.dropout.append(self.dropout2)     
            self.dropout3 = nn.Dropout2d(1-dropout_keep_rate)
            self.dropout.append(self.dropout3)
            
            self.batch1=nn.BatchNorm1d(256) 
            self.batch.append(self.batch1)        
            self.batch2=nn.BatchNorm1d(64)
            self.batch.append(self.batch2)
            self.batch3=nn.BatchNorm1d(features*2)
            self.batch.append(self.batch3)            

            # decoder 
            self.dec1 = nn.Linear(in_features=features, out_features=64)
            self.dec.append(self.dec1)
            self.dec2 = nn.Linear(in_features=64, out_features=256)
            self.dec.append(self.dec2)
            self.dec3 = nn.Linear(in_features=256, out_features=images_size*images_size)
            self.dec.append(self.dec3)


            self.dec_drop1 = nn.Dropout2d(1-dropout_keep_rate)
            self.dec_drop.append(self.dec_drop1)
            self.dec_drop2 = nn.Dropout2d(1-dropout_keep_rate)
            self.dec_drop.append(self.dec_drop2)
            self.dec_drop3 = nn.Dropout2d(1-dropout_keep_rate)
            self.dec_drop.append(self.dec_drop3)

            
            self.dec_batch1 =nn.BatchNorm1d(64) 
            self.dec_batch.append(self.dec_batch1)
            self.dec_batch2 =nn.BatchNorm1d(256) 
            self.dec_batch.append(self.dec_batch2)
            self.dec_batch3 =nn.BatchNorm1d(images_size*images_size) 
            self.dec_batch.append(self.dec_batch3)
            
            
            

        def reparameterize(self, mean, log_var):
            
            epsilon = torch.randn_like(mean) 
            
            # z = mean + e*var
            return epsilon * torch.exp(log_var* 0.5) + mean 
        
        def sigmoid_loss(self,label,prediction):
            
            return label - label * prediction + torch.log(1 + torch.exp(-label))
        
        def log_normal_pdf(self,sample,mean,logvar):
            
            log2pi = torch.log(torch.tensor(2*torch.pi))
            sample = torch.tensor(sample).clone().detach()
            mean = torch.tensor(mean).clone().detach()
            logvar = torch.tensor(logvar).clone().detach()
            
            return  torch.sum(-0.5 * ((sample-mean)**2*torch.exp(-logvar) + logvar + log2pi))
            
        def calculate_loss(self,label,prediction,z,mean,logvar):
            
            cross_ent = self.sigmoid_loss(label,prediction)
            logpx_z = torch.sum(cross_ent)
            logpz = self.log_normal_pdf(z,0,0)
            logz_x = self.log_normal_pdf (z,mean,logvar)
            
            return -torch.mean(logpx_z + logpz - logz_x)

        def kl_loss(self,z,mean,logvar):
            
            kl_loss = torch.mean(-0.5 * (1 + logvar - torch.square(mean) - torch.exp(logvar)))
            
            return kl_loss
            
        def forward(self, x):

            x = torch.flatten(x, start_dim=1)   
            
            # x = x.reshape(-1,1,images_size,images_size)
            #Encoder Netwrok
            for i in range(3):
                #cnv
                x = self.env[i](x)
                #print(f"{i}**",x.shape)
                
                 #relu
                x = F.relu(x)
                #print(f"{i}**",x.shape)
                
                 #batch
                x = self.batch[i](x)
                #print(f"{i}**",x.shape)
                
                 #pooling
                # x = self.maxpooling[i](x)
                #print(f"{i}**",x.shape)
                x = self.dropout[i](x)
                
              
                
           
            
            # Latent Space
            x = x.view(-1, 2, features)
            mean = x[:, 0, :] # as  mean
            log_var = x[:, 1, :] #  as variance
            z = self.reparameterize(mean, log_var)
            
            
            #Decoder
            # x = z.reshape(-1,1,1,features)
            x = z
            
            for i in range(3):
                #cnv
                x = self.dec[i](x)
                #print(f"{i}**",x.shape)
                
                 #relu
                x = F.relu(x)
                #print(f"{i}**",x.shape)
                
                 #batch
                x = self.dec_batch[i](x)
                #print(f"{i}**",x.shape)
                                
               
                #dropout
                if i == 2: #Last Layer 
                    # x = torch.flatten(x, start_dim=1) 
                    # x = self.dec[4](x)
                    out =  x.reshape(-1,images_size,images_size)
                else:
                    x = self.dec_drop[i](x)
                # print(f"{i}**",x.shape)
            


            return out, mean, log_var, z


if __name__ == "__main__":
    model = VAEModel()
    print(model.parameters)
    
    
    
    