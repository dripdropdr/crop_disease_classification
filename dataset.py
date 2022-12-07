import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import cv2
import json
from tqdm import tqdm
from glob import glob
import torch
from torch import nn
from torchvision import models
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, files, csv_feature_dict, label_encoder, label_list,labels=None, mode='train'):
        self.mode = mode
        self.files = files
        self.csv_feature_dict = csv_feature_dict
        self.csv_feature_check = [0]*len(self.files)
        self.csv_features = [None]*len(self.files)
        self.max_len = 24 * 6
        self.label_encoder = label_encoder
        self.label_list = label_list

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split('/')[-1]
        
        # csv
        if self.csv_feature_check[i] == 0:
            csv_path = f'{file}/{file_name}.csv'
            df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
            df = df.replace('-', 0)
            # MinMax scaling
            for col in df.columns:
                df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
                df[col] = df[col] / (self.csv_feature_dict[col][1]-self.csv_feature_dict[col][0])
            # zero padding
            pad = np.zeros((self.max_len, len(df.columns)))
            length = min(self.max_len, len(df))
            pad[-length:] = df.to_numpy()[-length:]
            # transpose to sequential data
            csv_feature = pad.T
            self.csv_features[i] = csv_feature
            self.csv_feature_check[i] = 1
        else:
            csv_feature = self.csv_features[i]
        
        # image
        image_path = f'{file}/{file_name}.jpg'
        img = cv2.imread(image_path)
        img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)/224
        img = np.transpose(img, (2,0,1))
        

        
        label = self.label_list[int(file_name)]
        
        return {
            'img' : torch.tensor(img, dtype=torch.float32),
            'csv_feature' : torch.tensor(csv_feature, dtype=torch.float32),
            'label' : torch.tensor(self.label_encoder[label], dtype=torch.long)
        }
    
