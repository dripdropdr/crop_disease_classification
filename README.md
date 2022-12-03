# business_analytics
2022 term project of business analaytics: Classification for plant and disease
<br>
![image](https://user-images.githubusercontent.com/81093298/205440581-0a9d9f71-c076-4d0e-bd88-5ddf5541d62d.png)
   
# Run model
***
<br>
### Training
<pre><code>
python plant_main.py --epochs {epoch} --batch_size {batch_size} --workers {data_worker} --resdir {output_directory}
</code></pre>
<br>
### Testing
<pre><code>
python plant_main.py --epochs {epoch} --batch_size {batch_size} --workers {data_worker} --resdir {output_directory}
</code></pre>
<br>>
To get Pretrain model for Vision Transformer: ![jx_vit_base_p16_224-80ecf9dd.pth](https://www.kaggle.com/datasets/abhinand05/vit-base-models-pretrained-pytorch)
<br>

# Result
### Train Result
<img src="https://user-images.githubusercontent.com/81093298/205441160-428f75ef-01da-4799-a90f-3411f46e0051.png" width="450px" title="f1-score" alt="f1-score"></img><br>
We use F1-score macro, because this dataset are class imbalanced. Refer the EDA of our repository.
<br>
<br>
<img src="https://user-images.githubusercontent.com/81093298/205441173-07f96acf-707e-4da4-bc37-2a13164c240e.png" width="450px" title="loss" alt="loss"></img>
<br>
We use Cross-Entropy Loss
<br>
<br>
### Ablation Study   
|Metric|ViT+LSTM|ViT only|LSTM only|
|------|---|---|
|F1 score|0.9870|0.9655|0.7043|
|Precision|0.9875|0.9680|0.8001|
|Recall|0.9869|0.9650|0.7101|
   
# Data Description
상우야 여깅
