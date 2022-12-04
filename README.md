# Business Analytics
This git repository is for 2022 term project of business analaytics. 
Our task is the classification for plant and disease
<br>

# Method
<img src="https://user-images.githubusercontent.com/81093298/205440581-0a9d9f71-c076-4d0e-bd88-5ddf5541d62d.png" width="500px" title="model" alt="model"></img><br>
We ensemble Vision Transformer and LSTM by concatenating the final features that go through the Head, to handel multi -Image and Time Series- data both.
   
# Run model
### Training
<br>
<pre><code>python plant_main.py --epochs {epoch} --batch_size {batch_size} --workers {data_worker} --resdir {output_directory}</code></pre>

### Testing
<br>
<pre><code>python predict.py --batch_size {batch_size} --workers {data_worker} --resdir {output_directory}</code></pre>
<br>
When training Vision Transformer, we use jx_vit_base_p16_224-80ecf9dd.pth pretrained model on ImaegNet-21K
<br>
To get pretrained model : https://www.kaggle.com/datasets/abhinand05/vit-base-models-pretrained-pytorch
<br>

# Result
### Train Result
<img src="https://user-images.githubusercontent.com/81093298/205441160-428f75ef-01da-4799-a90f-3411f46e0051.png" width="350px" title="f1-score" alt="f1-score"></img><br>
We use F1-score macro, because this dataset are class imbalanced. The detail of our class distribution is in the EDA of our repository.
<br>
<br>
<img src="https://user-images.githubusercontent.com/81093298/205441173-07f96acf-707e-4da4-bc37-2a13164c240e.png" width="350px" title="loss" alt="loss"></img>
<br>
We use Cross-Entropy Loss.
<br>
<br>
<img src="https://user-images.githubusercontent.com/81093298/205442385-de2b71fe-44fc-4147-b552-442c7b20bb11.png" width="400px" title="ceequation" alt="equation"></img>
<br>
<img src="https://user-images.githubusercontent.com/81093298/205442465-53c1389f-5f97-4fe3-b225-ccb4167d21ea.png" height="100px" title="softmax" alt="softmax"></img>
<img src="https://user-images.githubusercontent.com/81093298/205442467-e9025d90-57a7-4d5b-b5a0-d69564cfb5bd.png" height="100px" title="label" alt="label"></img>
<br>
<img src="https://user-images.githubusercontent.com/81093298/205442471-5d1d25de-cf1f-4f7a-86de-89f96fc876ba.png" width="400px" title="ce" alt="f1-score"></img>
<br>
Cross-Entropy Loss usually used on Classification model. The class labels that are represented to one-hot vector is multiplied with the probability distribution that is output of the model
<br>
<br>
### Ablation Study   
|Metric|ViT+LSTM|ViT only|LSTM only|
|---|---|---|---|
|F1 score|0.9870|0.9655|0.7043|
|Precision|0.9875|0.9680|0.8001|
|Recall|0.9869|0.9650|0.7101|   
<br>

# Data Description

Data : 57673<br>
Train : Valid : Test = 7 : 1 : 2<br>
Label : 20 classes<br>
Epoch : 10, Batch size : 32<br>
Loss function : Cross-entropy, Optimizer : Adam<br>
NVIDIA RTX A4000 GPU, 32GB RAM, Ubuntu 18.04<br>

### Distribution of classes
<img width="600px" alt="image" src="https://user-images.githubusercontent.com/59056821/205493076-1bbec5f9-8236-4f35-bf5b-b481fbcfa244.png">


### Label
 '1_00_0' : '딸기_정상'<br>
 '2_00_0' : '토마토_정상'<br>
 '2_a5_2' : '토마토_흰가루병_중기'<br>
 '3_00_0' : '파프리카_정상'<br>
 '3_a9_1' : '파프리카_흰가루병_초기'<br>
 '3_a9_2' : '파프리카_흰가루병_중기'<br>
 '3_a9_3' : '파프리카_흰가루병_말기'<br>
 '3_b3_1' : '파프리카_칼슘결핍_초기'<br>
 '3_b6_1' : '파프리카_다량원소결핍(N)_초기'<br>
 '3_b7_1' : '파프리카_다량원소결핍(P)_초기'<br>
 '3_b8_1' : '파프리카_다량원소결핍(K)_초기'<br>
 '4_00_0' : '오이_정상'<br>
 '5_00_0' : '고추_정상'<br>
 '5_a7_2' : '고추_탄저병_중기'<br>
 '5_b6_1' : '고추_다량원소결핍(N)_초기'<br>
 '5_b7_1' : '고추_다량원소결핍(P)_초기'<br>
 '5_b8_1' : '고추_다량원소결핍(K)_초기'<br>
 '6_00_0' : '시설포도_정상'<br>
 '6_a11_1' : '시설포도_탄저병_초기'<br>
 '6_a11_2' : '시설포도_탄저병_중기'<br>
 '6_a12_1' : '시설포도_노균병_초기'<br>
 '6_a12_2' : '시설포도_노균병_중기'<br>
 '6_b4_1' : '시설포도_일소피해_초기'<br>
 '6_b4_3' : '시설포도_일소피해_말기'<br>
 '6_b5_1' : '시설포도_축과병_초기'<br>
 
 ### Visualize images by class
<img width="500px" alt="image" src="https://user-images.githubusercontent.com/59056821/205496378-71533d5e-4dca-4b19-8df6-b839ec845701.png">

<img width="500px" alt="image" src="https://user-images.githubusercontent.com/59056821/205496399-3771d102-c38f-4e6d-8e6d-279c8215fec6.png">

<img width="500px" alt="image" src="https://user-images.githubusercontent.com/59056821/205496523-ffbb5a1d-b683-4cde-8992-4476f2a34dbb.png">

<img width="500px" alt="image" src="https://user-images.githubusercontent.com/59056821/205496855-8c5c2947-41d8-442c-91df-a4920c1bfd4d.png">

<img width="500px" alt="image" src="https://user-images.githubusercontent.com/59056821/205496758-56ee93aa-539c-48e5-890d-730c045954b7.png">

### Visualize environment variables by class
<img width="500px" alt="image" src="https://user-images.githubusercontent.com/59056821/205496983-f6b6607d-4f76-407a-8c20-f84c28a0e080.png">

<img width="500px" alt="image" src="https://user-images.githubusercontent.com/59056821/205496991-b08b4366-8c74-4912-aff9-74ca236412bb.png">

<img width="500px" alt="image" src="https://user-images.githubusercontent.com/59056821/205497037-afed00ca-cdc4-4665-923a-9350369c015f.png">

<img width="500px" alt="image" src="https://user-images.githubusercontent.com/59056821/205497053-9d3834c1-e949-41d2-9ea5-f1e600362f2e.png">



# Reference
https://velog.io/@cha-suyeon/%EC%86%90%EC%8B%A4%ED%95%A8%EC%88%98loss-function-Cross-Entropy-Loss


