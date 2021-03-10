# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 21:12:57 2020

@author: User
"""

import torch
import torch.nn as nn
from torchvision import datasets ,models,transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from dataloader import make_train_loader , make_test_loader
import os    
    
PATH_train="D:\stuff\class\DL\cnn_mango\C1-P1_Train"
PATH_val="D:\stuff\class\DL\cnn_mango\C1-P1_Dev"
PATH_test=0

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 32
# learning rate
lr = 0.01
epochs = 50 
 #path, image_name,label,pos,pos_old,label_new
#data augumentation
import pandas as pd
from skimage.io import imread, imsave
from tqdm import tqdm
from data_aug import *
train_data=pd.read_csv('train.csv')
val_data=pd.read_csv('dev.csv')
column_names = ["image_id",'label']
df = pd.DataFrame(columns = column_names)
file_dir="D:\stuff\class\DL\cnn_mango\C1-P1_Train\data\\"
output_path="D:\stuff\class\DL\cnn_mango\C1-P1_Train\\aug\\"

file_dir2="D:\stuff\class\DL\cnn_mango\C1-P1_Dev\data\\"
output_path2='D:\stuff\class\DL\cnn_mango\C1-P1_Dev\\aug\\'
dfv = pd.DataFrame(columns = column_names)

pos=0
pos_old=0

for root, _, files in os.walk(file_dir):
    print(root)
    for file in files:
        raw_image = Data_augmentation(root,file,train_data,pos,pos_old,df)
        raw_image.image_augment(output_path)
        pos=pos+4
        pos_old=pos_old+1
        print(df['image_id'][pos-4:pos])
        
pos=0
pos_old=0       
for root, _, files in os.walk(file_dir2):
    print(root)
    for file in files:
        raw_image = Data_augmentation(root,file,val_data,pos,pos_old,dfv)
        raw_image.image_augment(output_path2)
        pos=pos+4
        pos_old=pos_old+1
        print(dfv['image_id'][pos-4:pos])
        
        
df.to_csv('train_aug.csv')
dfv.to_csv('dev_aug.csv')





class cnn_model(nn.Module):
    def __init__(self):
        super(cnn_model, self).__init__()
        # Convolution 1 , input_shape=(3,224,224)
        
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0) #output_shape=(16,220,220) #(224-5+1)/1 #(weigh-kernel+1)/stride 無條件進位
        nn.init.xavier_uniform(self.cnn1.weight)

        self.relu1 = nn.ReLU() # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,110,110) #(220/2)
        
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) #output_shape=(32,106,106)
        nn.init.xavier_uniform(self.cnn2.weight)
        self.relu2 = nn.ReLU()  # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,53,53)
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0) #output_shape=(16,51,51)
        nn.init.xavier_uniform(self.cnn3.weight)
        self.relu3 = nn.ReLU()  # activation
        # Max pool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,25,25)
        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0) #output_shape=(8,23,23)
        nn.init.xavier_uniform(self.cnn4.weight)
        self.relu4 = nn.ReLU()  # activation
        # Max pool 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) #output_shape=(8,11,11)
        # Fully connected 1 ,#input_shape=(8*12*12)
        self.fc1 = nn.Linear(8 * 11 * 11, 512) 
        nn.init.xavier_uniform(self.fc1.weight)
        self.relu5 = nn.ReLU()  # activation
        self.fc2 = nn.Linear(512, 3)
        nn.init.xavier_uniform(self.fc2.weight)
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
    


model_cuda=cnn_model()
#device=torch.device("cuda")
#model_cuda=model_cuda.to(device)
import pandas as pd
train_loader, valid_loader = make_train_loader()
optimizer = torch.optim.Adam(model_cuda.parameters(),lr=0.0001)
for epoch in tqdm(range(1, epochs+1)):
    model_cuda.train()
    #model_cuda.zero_grad()
    train_loss = 0.
    valid_loss = 0.
    train_auc  = 0.
    valid_auc  = 0.


    for data,target in tqdm(train_loader):
        #data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model_cuda(data)
        target = target.long()
        target = torch.reshape(target,(1,-1)).squeeze(0)
        #print(target)
        #print('output')
        #print(output.max(axis=1).indices)
        loss = torch.nn.functional.cross_entropy(output, target)
        print('loss: ',loss)
        loss.backward()
        optimizer.step()
        train_auc  += (output.max(axis=1).indices==target).sum()
        train_loss += loss.item() * data.size(0)
        

    model_cuda.eval()
    for data, target in tqdm(valid_loader):
        #data, target = data.cuda(), target.cuda()
        output = model_cuda(data)
        target = target.long()
        target = torch.reshape(target,(1,-1)).squeeze(0)
        loss = torch.nn.functional.cross_entropy(output, target)
        valid_auc  += (output.max(axis=1).indices==target).sum()
        valid_loss += loss.item() * data.size(0)

    train_loss /= int(np.floor(len(train_loader.dataset)))
    valid_loss /= int(np.floor(len(valid_loader.dataset)))
    train_auc /= int(np.floor(len(train_loader.dataset)))
    valid_auc /= int(np.floor(len(valid_loader.dataset)))
    print('Epoch: {}, Training Loss: {:.4f},Training Auc: {:.4f}, Validation Loss: {:.4f},Validation Auc: {:.4f}'.format(epoch, train_loss,train_auc, valid_loss,valid_auc))

path='D:\stuff\class\DL\cnn_mango'

torch.save(model.state_dict(),os.path.join(path2,'model.pkl'))    






import cv2
from PIL import Image
from torch.utils.data         import DataLoader

def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)
    
    
    
 '''   
transform = transforms.Compose([transforms.Resize([224, 224]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

'''
cnn = cnn_model()
cnn.load_state_dict(torch.load('model.pth', map_location='cpu'))

cam = cv2.VideoCapture(0)#開啟相機
path2='D:\stuff\class\DL\cnn_mango\\test\\data'
img_counter =0
has_picture=0
predict=0
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        if(has_picture==1):
            data=os.listdir(path2)
            for d in data:
                os.remove(os.path.join(path2,d))
                print(os.path.join(path2,d))
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.jpg".format(img_counter)
        cv2.imwrite(path2+'/'+img_name, frame)
        print("{} written!".format(img_name))
        print(path2+'/'+img_name)
        img_counter += 1
        has_picture=1


    elif k==ord('q'):
        print('start training')
        test_loader = make_test_loader()
        for data in test_loader:
            predict=cnn(data[0])
            print(predict.max(axis=1).indices)
            


cam.release()

cv2.destroyAllWindows()


output_dir = "/".join(weight_path.split("/")[:-1])
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

'''
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
        return image,label
    
   
dataset = Arthopod_Dataset(label_train,BASE_PATH,transform)
dataset_val=Arthopod_Dataset(label_val,BASE_PATH2,transform)



train_data = datasets.
valid_data = datasets.
'''