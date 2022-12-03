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
<br>
### Testing
<br>
<pre><code>python predict.py --batch_size {batch_size} --workers {data_worker} --resdir {output_directory}</code></pre>
<br>
When training Vision Transformer, we use jx_vit_base_p16_224-80ecf9dd.pth pretrained model on ImaegNet-21K
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
<img src="https://user-images.githubusercontent.com/81093298/205442385-de2b71fe-44fc-4147-b552-442c7b20bb11.png" width="400px" title="ceequation" alt="equation"></img>
<br>
<img src="https://user-images.githubusercontent.com/81093298/205442465-53c1389f-5f97-4fe3-b225-ccb4167d21ea.png" height="150px" title="softmax" alt="softmax"></img>
<img src="https://user-images.githubusercontent.com/81093298/205442467-e9025d90-57a7-4d5b-b5a0-d69564cfb5bd.png" height="150px" title="label" alt="label"></img>
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
상우야 여깅

# Reference
https://velog.io/@cha-suyeon/%EC%86%90%EC%8B%A4%ED%95%A8%EC%88%98loss-function-Cross-Entropy-Loss


