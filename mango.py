# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:14:19 2020

@author: User
"""
#因為cfg.weight path 不知為何不能從config內改路徑，因此路徑是從本程式內改，再麻煩助教留意了
#我把一些程式寫在mango.py內
import numpy as np 
import pandas as pd # data processing, CSV file 
import os
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from config.defaults import _C as cfg
import torch
# Neural networks can be constructed using the torch.nn package.
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
############  make train loader and validation loader #######
#label processing:add label in mango pictures
label_train=pd.read_csv('train.csv')
label_val=pd.read_csv('dev.csv')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#A,B,C transfer to 1,2,3
lb = LabelEncoder()
label_train['encoded_labels'] = lb.fit_transform(label_train['label'])
label_val['encoded_labels'] = lb.fit_transform(label_val['label'])
#turn type to long :model requirement
label_train['encoded_labels']=label_train['encoded_labels'].astype('long')
label_train['encoded_labels']=label_train['encoded_labels'].astype('long')
#processing indice
train_indices =list(label_train.index)
val_indices  =list(label_val.index)
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
#normalization
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#set validation path and train path
BASE_PATH="D://stuff//class//DL//cnn_mango//C1-P1_Train//data"
BASE_PATH2="D://stuff//class//DL//cnn_mango//C1-P1_Dev//data"

class Arthopod_Dataset(Dataset):
    def __init__(self, img_data,img_path,transform=None):
        self.img_path = img_path
        self.transform = transform
        self.img_data = img_data
        
    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, index):
        img_name = os.path.join(self.img_path,self.img_data.loc[index, 'image_id'])
        image = Image.open(img_name)
        #image = image.convert('RGB')
        image = image.resize((224,224))
        label = torch.tensor(self.img_data.loc[index, 'encoded_labels'])
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
   
dataset = Arthopod_Dataset(label_train,BASE_PATH,transform)
dataset_val=Arthopod_Dataset(label_val,BASE_PATH2,transform)

#make train loader    
batch_size = 128
shuffle_dataset = True
random_seed= 42
    
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           )
validation_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                                )


dataiter = iter(train_loader)
images, labels = dataiter.next()
arthopod_types = {0: 'A', 1: 'B', 2: 'C'}

        
        
class cnn_model(nn.Module):
    def __init__(self):
        super(cnn_model, self).__init__()
        # Convolution 1 , input_shape=(3,224,224)
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0) #output_shape=(16,220,220) #(224-5+1)/1 #(weigh-kernel+1)/stride 無條件進位
        self.relu1 = nn.ReLU() # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,110,110) #(220/2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) #output_shape=(32,106,106)
        self.relu2 = nn.ReLU() # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,53,53)
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0) #output_shape=(16,51,51)
        self.relu3 = nn.ReLU() # activation
        # Max pool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,25,25)
        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0) #output_shape=(8,23,23)
        self.relu4 = nn.ReLU() # activation
        # Max pool 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) #output_shape=(8,11,11)
        # Fully connected 1 ,#input_shape=(8*12*12)
        self.fc1 = nn.Linear(8 * 11 * 11, 512) 
        self.relu5 = nn.ReLU() # activation
        self.fc2 = nn.Linear(512, 3) 
        self.output = nn.Softmax(dim=1)
        
    
    def forward(self, x):
        out = self.cnn1(x) # Convolution 1
        out = self.relu1(out)
        out = self.maxpool1(out)# Max pool 1
        out = self.cnn2(out) # Convolution 2
        out = self.relu2(out) 
        out = self.maxpool2(out) # Max pool 2
        out = self.cnn3(out) # Convolution 3
        out = self.relu3(out)
        out = self.maxpool3(out) # Max pool 3
        out = self.cnn4(out) # Convolution 4
        out = self.relu4(out)
        out = self.maxpool4(out) # Max pool 4
        out = out.view(out.size(0), -1) # last CNN faltten con. Linear NN
        out = self.fc1(out) # Linear function (readout)
        out = self.fc2(out)
        out = self.output(out)

        return out
    
    
model = cnn_model() 


#rest of the program is the same as train.py
valid_size  = cfg.DATA.VALIDATION_SIZE
epochs      = cfg.MODEL.EPOCH
lr          = cfg.MODEL.LR
weight_path = "D:\stuff\class\DL\cnn_mango"
use_cuda    = cfg.DEVICE.CUDA
gpu_id      = 0


labels=torch.tensor(labels, dtype=torch.long)
data=torch.tensor(data, dtype=torch.long)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range(1, epochs+1):
    model.train()
    train_loss = 0.
    valid_loss = 0.
    train_acc=0.
    val_acc=0.
    count=1
    for data, target in train_loader:
        target = target.type(torch.LongTensor)
        optimizer.zero_grad()
        output = model(data)
        print(output.max(axis=1).indices)
        print((output.max(axis=1).indices==target).float().sum()/batch_size)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        train_acc+=(output.max(axis=1).indices==target).float().sum()
        count=count+1
        print(count)
    model.eval()
    print(train_acc/int(np.floor(len(train_loader.dataset) * (1 - valid_size))))
    for data, target in validation_loader:
        target = target.type(torch.LongTensor)
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        valid_loss += loss.item() * data.size(0)
        val_acc+=(output.max(axis=1).indices==target).float().sum()

    train_loss /= int(np.floor(len(train_loader.dataset) ))
    valid_loss /= int(np.floor(len(validation_loader.dataset)))
    train_acc/=int(np.floor(len(train_loader.dataset)))
    val_acc/=int(np.floor(len(validation_loader.dataset)))
    
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, train_loss, valid_loss))
    print("Training acc: {:.4f},val acc: {:.4f}".format(train_acc,val_acc))

output_dir = "/".join(weight_path.split("/")[:-1])
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

torch.save(model.state_dict(), os.path.join(weight_path,'weight.csv'))