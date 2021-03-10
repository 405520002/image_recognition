# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:51:24 2020

@author: User
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.






import cv2
import random
class Data_augmentation:
    def __init__(self, path, image_name,label,pos,pos_old,label_new):
        '''
        Import image
        :param path: Path to the image
        :param image_name: image name
        '''
        self.path = path
        self.name = image_name
        print(path+image_name)
        self.image = cv2.imread(path+image_name)
        self.label=label
        self.pos=pos
        self.pos_old=pos_old
        self.label_new=label_new
    def rotate(self, image, angle=90, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        #rotate
        image = cv2.warpAffine(image,M,(w,h))
        return image

    def flip(self, image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image 
    def gauss(self,image, noise_sigma=35):
        temp_image = np.float64(np.copy(image))

        h = temp_image.shape[0]
        w = temp_image.shape[1]
        noise = np.random.randn(h, w) * noise_sigma

        noisy_image = np.zeros(temp_image.shape, np.float64)
        if len(temp_image.shape) == 2:
            noisy_image = temp_image + noise
        else:
            noisy_image[:,:,0] = temp_image[:,:,0] + noise
            noisy_image[:,:,1] = temp_image[:,:,1] + noise
            noisy_image[:,:,2] = temp_image[:,:,2] + noise

        return noisy_image
    
    def image_augment(self, save_path): 
        '''
        Create the new image with imge augmentation
        :param path: the path to store the new image
        ''' 
        img = self.image.copy()
        img_flip = self.flip(img, vflip=True, hflip=False)
        img_rot = self.rotate(img)
        img_gaussian = self.gauss(img)
        
        name_int = self.name[:len(self.name)-4]
        cv2.imwrite(save_path+'%s' %str(self.name),img)
        cv2.imwrite(save_path+'%s' %str(name_int)+'_vflip.jpg', img_flip)
        cv2.imwrite(save_path+'%s' %str(name_int)+'_rot.jpg', img_rot)
        cv2.imwrite(save_path+'%s' %str(name_int)+'_GaussianNoise.jpg', img_gaussian)
        self.label_new.at[self.pos,'image_id']=str(self.name)
        self.label_new.at[self.pos+1,'image_id']=str(name_int)+'_vflip.jpg'
        self.label_new.at[self.pos+2,'image_id']=str(name_int)+'_rot.jpg'
        self.label_new.at[self.pos+3,'image_id']=str(name_int)+'_GaussianNoise.jpg'
        #labeling
        self.label_new.at[self.pos,'label']=self.label['label'][self.pos_old]
        self.label_new.at[self.pos+1,'label']=self.label['label'][self.pos_old]
        self.label_new.at[self.pos+2,'label']=self.label['label'][self.pos_old]
        self.label_new.at[self.pos+3,'label']=self.label['label'][self.pos_old]
        print(self.label_new['image_id'][self.pos:self.pos+4])
        print(self.label_new['label'][self.pos:self.pos+4])
        
        
    def main(file_dir,output_path):
        for root, _, files in os.walk(file_dir):
            print(root)
        for file in files:
            raw_image = Data_augmentation(root,file)
            raw_image.image_augment(output_path)
    