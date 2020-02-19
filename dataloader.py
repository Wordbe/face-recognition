#####################################
# Data Set                          #
#####################################
import os
import glob
from datetime import date, datetime
# import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

# import imgaug as ia
# from imgaug import augmenters as iaa

class FaceDataset(Dataset):
  def __init__(self, train_table, train_dir, transform=None, is_train=True):
    self.train_dir = train_dir
    self.train_table = train_table
    
    self.H = 128 # Height
    self.W = 128 # Width
    self.transform = transform
    self.is_train = is_train
      
  def __len__(self):
    return len(self.train_table)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist() # to list
    
    filename = self.train_table['filename'][idx]
    img_file = self.train_dir + filename
    # img = cv2.imread(img_file)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.open(img_file).convert('RGB')
    label = self.train_table['label'][idx]
    
    if self.transform:
      if self.is_train:
        img = self.transform['train'](img)
      else:
        img = self.transform['valid'](img)

    sample = {'img' : img, 'label': torch.tensor(label)}
    return sample

transform = { 
              'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(), # (H x W x C) [0, 255] to (C x H x W) [0.0, 1.0] 
                transforms.Normalize(mean=[0.5239860024529774, 0.42738061407253225, 0.37977362891997446],
                                    std=[0.2834680229098988, 0.25519093913452706, 0.25590995105180836])
                ]),
              'valid': transforms.Compose([
                transforms.ToTensor(), # (H x W x C) [0, 255] to (C x H x W) [0.0, 1.0] 
                transforms.Normalize(mean=[0.5239860024529774, 0.42738061407253225, 0.37977362891997446],
                                    std=[0.2834680229098988, 0.25519093913452706, 0.25590995105180836])
                ]),
            }


#####################################
# Data Loader                       #
#####################################

from torch.utils.data import DataLoader

def load_data(root_path, label_csv_path, batch_size):
    root = root_path
    train_dir = root + '/face_images_128x128/'
    train_csv = glob.glob(label_csv_path)[0]
    
    # Split train and validation data
    train_table = pd.read_csv(train_csv)
    valid_table = train_table.sample(frac=0.2, random_state=999)
    train_table.drop(index=valid_table.index, axis=0, inplace=True)

    # Reset index (0, 1, 2, ...)
    train_table.reset_index(inplace=True)
    valid_table.reset_index(inplace=True)
    
    train_dataset = FaceDataset(train_table=train_table, train_dir=train_dir, transform=transform, is_train=True)
    valid_dataset = FaceDataset(train_table=valid_table, train_dir=train_dir, transform=transform, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print("trainset : {}, validation set : {}".format(len(train_loader), len(valid_loader)))

    #####################################
    # Check                             #
    #####################################
    # batch = 0
    # cnt = 0
    # for i in train_loader:
    #   print(batch)
    #   batch += 1
    #   print(i['img'].size())
    #   npgrid = torchvision.utils.make_grid(i['img']).numpy()
    #   plt.figure(figsize=(18, 32))
    #   plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    #   plt.show()
    #   for j in range(BATCH_SIZE):
    #     img = i['img'][j]
    #     label = i['label'][j]
    #     cnt += 1
    #     print(torch.min(img), torch.max(img))
    #     print(cnt, np.shape(img), label)
    #     break
    #   break  
    
    return (train_loader, valid_loader, len(train_dataset), len(valid_dataset))

