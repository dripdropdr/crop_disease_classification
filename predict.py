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
# from linformer import Linformer

from dataset import *
from models import *
from plant_main import *

def predict(dataset):
    model.eval()
    tqdm_dataset = tqdm(enumerate(dataset))
    results = []
    for batch, batch_item in tqdm_dataset:
        img = batch_item['img'].to(device)
        seq = batch_item['csv_feature'].to(device)
        with torch.no_grad():
            output = model(img, seq)
        output = torch.tensor(torch.argmax(output, dim=1), dtype=torch.int32).cpu().numpy()
        results.extend(output)
    return results
def accuracy_function(real, pred):    
    score = f1_score(real, pred, average='macro')
    return score


csv_feature_dict = csv_features()
# class
crop = {'1':'딸기','2':'토마토','3':'파프리카','4':'오이','5':'고추','6':'시설포도'}
disease = {'1':{'a1':'딸기잿빛곰팡이병','a2':'딸기흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
        '2':{'a5':'토마토흰가루병','a6':'토마토잿빛곰팡이병','b2':'열과','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
        '3':{'a9':'파프리카흰가루병','a10':'파프리카잘록병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
        '4':{'a3':'오이노균병','a4':'오이흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
        '5':{'a7':'고추탄저병','a8':'고추흰가루병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
        '6':{'a11':'시설포도탄저병','a12':'시설포도노균병','b4':'일소피해','b5':'축과병'}}

label_description = {}
# risk 제외함
for key, value in disease.items():
    label_description[f'{key}_00'] = f'{crop[key]}_정상'
    for disease_code in value:
        label = f'{key}_{disease_code}'
        label_description[label] = f'{crop[key]}_{disease[key][disease_code]}'

# numbering to each label
label_encoder = {key:idx for idx, key in enumerate(label_description)}
label_decoder = {val:key for key, val in label_encoder.items()}
test = sorted(glob('./dataset/test/*'))

labels = pd.read_csv('./test.csv')['label']
label_revised = labels.str.slice(start=0, stop=4)
labels = pd.read_csv('./test.csv')
label_list = labels.set_index('image', drop=True)["label"]
# 7:1:2

test_dataset = CustomDataset(test,label_list, csv_feature_dict, label_encoder)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=8, shuffle=True)


device = torch.device("cuda:0")
max_len = 24*6
batch_size = args.batch_size
num_classes = len(label_encoder)
learning_rate = 1e-4
embedding_dim = 512
num_features = len(csv_feature_dict)
dropout_rate = 0.1
epochs = args.epochs
model = ViT2RNN(max_len=max_len, embedding_dim=embedding_dim, num_features=num_features, num_classes=num_classes, rate=dropout_rate)
model.load_state_dict(torch.load("./res/weight/best_weight.pt", map_location=device))
model = model.to("cuda:0") # model.to('cuda:0')

prediction = predict(test_dataloader)
preds = np.array([label_decoder[int(val)] for val in prediction])
print(accuracy_function(label_list.values,preds.values))

