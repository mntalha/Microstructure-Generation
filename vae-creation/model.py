"""
Created on Sun Dec 12, 2022
@author: talha.kilic
"""
# Variational AutoEncoder Model by means of PyTorch Support 


#library imports 
import torch.nn as nn
import torch.nn.functional as F
import torch 
# from CNN import CNNModel
# Macros
dropout_keep_rate= 0.90
features = 9
images_size = 96
#Paths
keep_rate=0.9
channel_size = 50
class VAEModel(nn.Module):
    
        #List of Modules
        conv = []
        dropout = []
        batch = []
        maxpooling = []
        
        after_cnv_size = channel_size*6*6
        enc_lt = []
        
        dec = []
        dec_drop = []
        dec_batch = []

        def __init__(self):
            super(VAEModel, self).__init__()
            
            self.conv1 = nn.Linear(in_features=images_size*images_size, out_features=128*32)
            self.conv.append(self.conv1)
            self.conv2 = nn.Linear(in_features=128*32, out_features=128*8)
            self.conv.append(self.conv2)
            self.conv3 =nn.Linear(in_features=128*8, out_features=128*4)
            self.conv.append(self.conv3)        
            self.conv4 = nn.Linear(in_features=128*4, out_features=128*2)
            self.conv.append(self.conv4)
            
            
            self.dropout1 = nn.Dropout2d(1-keep_rate)
            self.dropout.append(self.dropout1)       
            self.dropout2 = nn.Dropout2d(1-keep_rate)
            self.dropout.append(self.dropout2)     
            self.dropout3 = nn.Dropout2d(1-keep_rate)
            self.dropout.append(self.dropout3)
            self.dropout4 = nn.Dropout2d(1-keep_rate)
            self.dropout.append(self.dropout4)
            
            
            self.batch1=nn.BatchNorm1d(128*32) 
            self.batch.append(self.batch1)        
            self.batch2=nn.BatchNorm1d(128*8)
            self.batch.append(self.batch2)
            self.batch3=nn.BatchNorm1d(128*4)
            self.batch.append(self.batch3)            
            self.batch4=nn.BatchNorm1d(128*2)
            self.batch.append(self.batch4)
            
            
            # self.maxpooling1= nn.MaxPool2d(2)
            # self.maxpooling.append(self.maxpooling1)
            # self.maxpooling2= nn.MaxPool2d(2)
            # self.maxpooling.append(self.maxpooling2)
            # self.maxpooling3= nn.MaxPool2d(2)
            # self.maxpooling.append(self.maxpooling3)
            # self.maxpooling4= nn.MaxPool2d(2)
            # self.maxpooling.append(self.maxpooling4)
            

            
            #Before Latent Space
            self.enc_lt1 = nn.Linear(in_features=128*2, out_features=features*2)
            self.enc_lt.append(self.enc_lt1)
            self.enc_batch1 =nn.BatchNorm1d(features*2) 
            
            
            # decoder 
            self.dec1 = nn.Linear(in_features=features, out_features=128)
            self.dec.append(self.dec1)
            self.dec2 = nn.Linear(in_features=128, out_features=256)
            self.dec.append(self.dec2)
            self.dec3 = nn.Linear(in_features=256, out_features=512)
            self.dec.append(self.dec3)
            self.dec4 = nn.Linear(in_features=512, out_features=images_size*images_size)
            self.dec.append(self.dec4) 
            

            self.dec_drop1 = nn.Dropout2d(1-dropout_keep_rate)
            self.dec_drop.append(self.dec_drop1)
            self.dec_drop2 = nn.Dropout2d(1-dropout_keep_rate)
            self.dec_drop.append(self.dec_drop2)
            self.dec_drop3 = nn.Dropout2d(1-dropout_keep_rate)
            self.dec_drop.append(self.dec_drop3)
            self.dec_drop4 = nn.Dropout2d(1-dropout_keep_rate)
            self.dec_drop.append(self.dec_drop4)
            
            self.dec_batch1 =nn.BatchNorm1d(128) 
            self.dec_batch.append(self.dec_batch1)
            self.dec_batch2 =nn.BatchNorm1d(256) 
            self.dec_batch.append(self.dec_batch2)
            self.dec_batch3 =nn.BatchNorm1d(512) 
            self.dec_batch.append(self.dec_batch3)
            self.dec_batch4 = nn.BatchNorm1d(images_size*images_size) 
            self.dec_batch.append(self.dec_batch4)
            
            
            

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

            
        def forward(self, x):

            x = torch.flatten(x, start_dim=1)   
            
            # x = x.reshape(-1,1,images_size,images_size)
            #Encoder Netwrok
            for i in range(4):
                #cnv
                x = self.conv[i](x)
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
                
               
                # #dropout
                # x = self.dropout[i](x)
                #print(f"{i}**",x.shape)
               
            #Flattening Operation, Dense Layers
            x = torch.flatten(x, start_dim=1)   
            
            for i in range(1):
                
                x = self.enc_lt[i](x)
                
                #relu
                x = F.relu(x)
                
                # batch
                x = self.enc_batch1(x)
                
           
            
            # Latent Space
            x = x.view(-1, 2, features)
            mean = x[:, 0, :] # as  mean
            log_var = x[:, 1, :] #  as variance
            z = self.reparameterize(mean, log_var)
            
            
            #Decoder
            # x = z.reshape(-1,1,1,features)
            x = z
            
            for i in range(4):
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
                if i == 3: #Last Layer 
                    # x = torch.flatten(x, start_dim=1) 
                    # x = self.dec[4](x)
                    out =  x.reshape(-1,images_size,images_size)
                # else:
                #     x = self.dec_drop[i](x)
                # print(f"{i}**",x.shape)
            


            return out, mean, log_var, z


if __name__ == "__main__":
    model = VAEModel()
    print(model.parameters)
    
    
    
    