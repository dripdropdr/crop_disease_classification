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



os.environ["CUDA_VISIBLE_DEVICES"]="0"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Total training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--resdir', type=str, default = './res', help='Path of saving result directory')
    return parser.parse_args()


def csv_features():
    # 분석에 사용할 feature 선택
    csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']
    csv_files = sorted(glob('./data/*/*.csv'))
    temp_csv = pd.read_csv(csv_files[0])[csv_features]
    max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

    # feature 별 최대값, 최솟값 계산
    for csv_path in tqdm(csv_files[1:], desc='feature minmax calculating'):
        try:
            temp_csv = pd.read_csv(csv_path, encoding = 'utf-8')[csv_features]
        except:
            temp_csv = pd.read_csv(csv_path, encoding = 'cp949')[csv_features]
        temp_csv = temp_csv.replace('-',np.nan).dropna()
        if len(temp_csv) == 0:
            continue
        temp_csv = temp_csv.astype(float)
        temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
        max_arr = np.max([max_arr,temp_max], axis=0)
        min_arr = np.min([min_arr,temp_min], axis=0)
    
    return {csv_features[i]:[min_arr[i], max_arr[i]] for i in range(len(csv_features))}


def accuracy_function(real, pred):    
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score


def visualize(train_plot, val_plot, name, resdir):
    plt.figure(figsize=(10,7))
    plt.grid()
    plt.plot(train_plot, label='train_%s' %(name))
    plt.plot(val_plot, label='val_%s' %(name))
    plt.xlabel('epoch')
    plt.ylabel(name)
    plt.title(name, fontsize=25)
    plt.legend()
    plt.savefig(resdir+'%s.png'%(name), dpi=300)


def run():
    args = parse_args()
    
    # feature 별 최대값, 최솟값 dictionary 생성
    csv_feature_dict = csv_features()

    if not os.path.exist(args.resdir):
        os.mkdir(args.resdir)
    
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


    device = torch.device("cuda:0")
    batch_size = args.batch_size
    num_classes = len(label_encoder)
    learning_rate = 1e-4
    embedding_dim = 512
    num_features = len(csv_feature_dict)
    dropout_rate = 0.1
    epochs = args.epochs

    # data load
    train = sorted(glob('./dataset/train/*'))

    labels = pd.read_csv('./train.csv')['label']
    label_revised = labels.str.slice(start=0, stop=4)
    # 7:1:2
    train, val = train_test_split(train, test_size=0.125, stratify=label_revised)

    train_dataset = CustomDataset(train, csv_feature_dict, label_encoder)
    val_dataset = CustomDataset(val, csv_feature_dict, label_encoder)
    # test_dataset = CustomDataset(test, mode = 'test')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=args.workers, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=args.workers, shuffle=False)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=16, shuffle=False)
    

    model = ViT2RNN(max_len=max_len, embedding_dim=embedding_dim, num_features=num_features, num_classes=num_classes, rate=dropout_rate)
    model = model.to(device) # model.to('cuda:0')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer
    criterion = nn.CrossEntropyLoss()

    # For Visualization
    loss_plot, val_loss_plot = [], []
    metric_plot, val_metric_plot = [], []

    if not os.path.exist(args.resdir+'/weight'):
        os.mkdir(args.resdir+'/weight')

    # Train Start
    for epoch in range(epochs):
        total_loss, total_val_loss = 0, 0
        total_acc, total_val_acc = 0, 0

        # Train
        tqdm_train_dataset = tqdm(enumerate(train_dataloader), desc='Training ... ')
        for batch, batch_item in tqdm_dataset:
            img = batch_item['img'].to(device)
            csv_feature = batch_item['csv_feature'].to(device)
            label = batch_item['label'].to(device)
            if training is True:
                model.train()
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    output = model(img, csv_feature)
                    loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                score = accuracy_function(label, output)
                batch_loss =  loss
                batch_acc = score

            else:
                

            total_loss += batch_loss
            total_acc += batch_acc
            
            tqdm_train_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Loss': '{:06f}'.format(batch_loss.item()),
                'Mean Loss' : '{:06f}'.format(total_loss/(batch+1)),
                'Mean F-1' : '{:06f}'.format(total_acc/(batch+1))
            })
        loss_plot.append(total_loss/(batch+1))
        metric_plot.append(total_acc/(batch+1))
        
        # Valid
        tqdm_val_dataset = tqdm(enumerate(val_dataloader), desc='Validation ... ')
        training = False
        for batch, batch_item in tqdm_dataset:
            model.eval()
            with torch.no_grad():
                output = model(img, csv_feature)
                loss = criterion(output, label)
            score = accuracy_function(label, output)
            batch_loss =  loss
            batch_acc = score

            total_val_loss += batch_loss
            total_val_acc += batch_acc
            
            tqdm_val_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Val Loss': '{:06f}'.format(batch_loss.item()),
                'Mean Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
                'Mean Val F-1' : '{:06f}'.format(total_val_acc/(batch+1))
            })
        val_loss_plot.append(total_val_loss/(batch+1))
        val_metric_plot.append(total_val_acc/(batch+1))

        if np.max(val_metric_plot) == val_metric_plot[-1]:
            torch.save(model.state_dict(), args.resdir+'/weight')

        visualize(metric_plot, val_metric_plot, 'f1_score', resdir)
        visualize(loss_plot, val_loss_plot, 'loss', resdir)

if __name__ == "__main__":                    
    run()

    