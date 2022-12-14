# Business Analytics
This git repository is for 2022 term project of business analytics. 
Our task is the classification for plant and disease

Our theme is a project to enable farmers to diagnose pests to efficiently manage crops on smart farms.
We make models using image data of plants and time series data of plants.

<br>

# Method
### Model
<img src="https://user-images.githubusercontent.com/81093298/205440581-0a9d9f71-c076-4d0e-bd88-5ddf5541d62d.png" width="500px" title="model" alt="model"></img>
<br>
We ensemble Vision Transformer and LSTM by concatenating the final features that go through the Head, to handel multi -Image and Time Series- data both.
<br>
<br>

### Data-Preprocessing
We split the source data and label.csv to train and test in data_preprocessing.ipynb

Next, to select the features that we use in our analysis, we aggregate the all values of our time-series data and calculate min, max of them.   
The detail of it is in: https://github.com/dripdropdr/business_analytics/blob/main/plant_main.py#L32
<br>
<pre><code>def csv_features():
    # 분석에 사용할 feature 선택
    csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']
    csv_files = sorted(glob('./data/*/*.csv'))
    temp_csv = pd.read_csv(csv_files[0])[csv_features]
    max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

    # feature 별 최대값, 최솟값 계산
    for csv_path in tqdm(csv_files[1:], desc='feature minmax calculating'):
        
        ...
    
    return {csv_features[i]:[min_arr[i], max_arr[i]] for i in range(len(csv_features))}</code></pre>
<br>

### Training Implementations
Epoch : 10, Batch size : 32<br>
Loss function : Cross-entropy, Optimizer : Adam<br>
NVIDIA RTX A4000 GPU, 32GB RAM, Ubuntu 18.04<br>

  
# Run model
### Training
<br>
<pre><code>python plant_main.py --epochs {epoch} --batch_size {batch_size} --workers {data_worker} --resdir {output_directory}</code></pre>

### Testing
<br>
<pre><code>python predict.py --batch_size {batch_size} --workers {data_worker} --resdir {output_directory}</code></pre>
<br>
When training Vision Transformer, we use jx_vit_base_p16_224-80ecf9dd.pth pretrained model on ImaegNet-21K
We cannot find the pretrained model that trained with the crop & disease data which similar with our tasks, so we used this to get general features
<br>
To get pretrained model : https://www.kaggle.com/datasets/abhinand05/vit-base-models-pretrained-pytorch
<br>

# Result

### Train Result
<img src="https://user-images.githubusercontent.com/81093298/205441160-428f75ef-01da-4799-a90f-3411f46e0051.png" width="350px" title="f1-score" alt="f1-score"></img><br>
We use validation set to confirm the training status of our model in each epoch. The learning curve shows the fitting of our model.   
Also, we employed F1-score macro for metric because this dataset are class imbalanced.   
The detail of our dataset: https://github.com/dripdropdr/business_analytics/blob/main/EDA.ipynb
<br>
<br>
<img src="https://user-images.githubusercontent.com/81093298/205857783-22cb5794-cc4e-4780-9a3c-c776c839017a.png" width="350px" title="loss" alt="loss"></img>
<br>
This is the formulation of f1-score. In Imbalanced dataset, accuracy isn't reflected the performance of the model.    
F1-score is the harmonic mean of precision and recall to represent the performance more correctly.


<br>
<br>
<img src="https://user-images.githubusercontent.com/81093298/205441173-07f96acf-707e-4da4-bc37-2a13164c240e.png" width="350px" title="loss" alt="loss"></img>
We use Cross-Entropy Loss.
<br>
<img src="https://user-images.githubusercontent.com/81093298/205442385-de2b71fe-44fc-4147-b552-442c7b20bb11.png" width="400px" title="ceequation" alt="equation"></img>
<img src="https://user-images.githubusercontent.com/81093298/205442465-53c1389f-5f97-4fe3-b225-ccb4167d21ea.png" height="100px" title="softmax" alt="softmax"></img>
<img src="https://user-images.githubusercontent.com/81093298/205442467-e9025d90-57a7-4d5b-b5a0-d69564cfb5bd.png" height="100px" title="label" alt="label"></img>
<img src="https://user-images.githubusercontent.com/81093298/205442471-5d1d25de-cf1f-4f7a-86de-89f96fc876ba.png" width="400px" title="ce" alt="f1-score"></img>
<br>
Cross-Entropy Loss usually used on Classification model. The class labels that are represented to one-hot vector is multiplied with the probability distribution that is output of the model
<br>
<br>

### Test result & Ablation Study

|Metric|ViT+LSTM|ViT only|LSTM only|
|---|---|---|---|
|F1 score|0.9870|0.9655|0.7043|
|Precision|0.9875|0.9680|0.8001|
|Recall|0.9869|0.9650|0.7101|  
 
<br>

The results shows the fusion model that use time series data and image data both performed best result.   
F1 score of LSTM only is relatively lower than ViT only. We assume the reason is that time series data, including environment information of the crops, didn't represent the features of the crop and disease directly than the image data.    
Environment information may include about disease information of each crops, but cannot represent the information of the crop itself.    

# Data Description

Data : 57673<br>
Train : Valid : Test = 7 : 1 : 2<br>
Label : 20 classes<br>

### Distribution of classes
<img width="600px" alt="image" src="https://user-images.githubusercontent.com/59056821/205493076-1bbec5f9-8236-4f35-bf5b-b481fbcfa244.png">


### Label
 '1_00' : '딸기_정상'<br>
 '2_00' : '토마토_정상'<br>
 '2_a5' : '토마토_흰가루병'<br>
 '3_00' : '파프리카_정상'<br>
 '3_a9' : '파프리카_흰가루병'<br>
 '3_b3' : '파프리카_칼슘결핍'<br>
 '3_b6' : '파프리카_다량원소결핍(N)'<br>
 '3_b7' : '파프리카_다량원소결핍(P)'<br>
 '3_b8' : '파프리카_다량원소결핍(K)'<br>
 '4_00' : '오이_정상'<br>
 '5_00' : '고추_정상'<br>
 '5_a7' : '고추_탄저병'<br>
 '5_b6' : '고추_다량원소결핍(N)'<br>
 '5_b7' : '고추_다량원소결핍(P)'<br>
 '5_b8' : '고추_다량원소결핍(K)'<br>
 '6_00' : '시설포도_정상'<br>
 '6_a11' : '시설포도_탄저병'<br>
 '6_a12' : '시설포도_노균병'<br>
 '6_b4' : '시설포도_일소피해'<br>
 '6_b5' : '시설포도_축과병'<br>
 
### Visualize images by class
 
<img width="300px" height="200px" alt="image" src="https://user-images.githubusercontent.com/59056821/205496378-71533d5e-4dca-4b19-8df6-b839ec845701.png"></img>
<img width="300px" height="200px" alt="image" src="https://user-images.githubusercontent.com/59056821/205496399-3771d102-c38f-4e6d-8e6d-279c8215fec6.png"></img>
<img width="300px" height="200px" alt="image" src="https://user-images.githubusercontent.com/59056821/205496523-ffbb5a1d-b683-4cde-8992-4476f2a34dbb.png"></img>
<img width="300px" height="200px" alt="image" src="https://user-images.githubusercontent.com/59056821/205496855-8c5c2947-41d8-442c-91df-a4920c1bfd4d.png"></img>
<img width="300px" height="200px" alt="image" src="https://user-images.githubusercontent.com/59056821/205496758-56ee93aa-539c-48e5-890d-730c045954b7.png"></img>

### Visualize environment variables by class
<img width="300px" height="200px" alt="image" src="https://user-images.githubusercontent.com/59056821/205496983-f6b6607d-4f76-407a-8c20-f84c28a0e080.png"></img>
<img width="300px" height="200px" alt="image" src="https://user-images.githubusercontent.com/59056821/205496991-b08b4366-8c74-4912-aff9-74ca236412bb.png"></img>
<img width="300px" height="200px" alt="image" src="https://user-images.githubusercontent.com/59056821/205497037-afed00ca-cdc4-4665-923a-9350369c015f.png"></img>
<img width="300px" height="200px" alt="image" src="https://user-images.githubusercontent.com/59056821/205497053-9d3834c1-e949-41d2-9ea5-f1e600362f2e.png"></img>


