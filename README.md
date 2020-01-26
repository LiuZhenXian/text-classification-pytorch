# Chinese Text Classification with CNN and RNN
<b> Update (January 23, 2020)</b> Pytorch implementation of [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882).

After opening a news website and choosing the mood as a result of reading the page, is there any relationship between **people's mood** and **different parts** of the news? 
<br/>

<br/>


## References
pytorch implementation: https://github.com/songyingxin/TextClassification-Pytorch

Another pytorch implementation: https://github.com/galsang/CNN-sentence-classification-pytorch

<br/>


## Getting Started

### Requirements

python 3.7

pytorch 1.1

tqdm

sklearn


### Getting Data and Pretrained Word2vec
Data: 

Some news contained people's emotional vote after reading the whole news or part of the news.

There were 6 choice of mood to choose : **Love, Fear, Joy, Sadness, Surprise, Anger**.

**News and its emotional label** were collected from websites.

**A part(now only shows the beginning of the news)** of these news are cut out as train data and test data.

*Pretrainded Word2vec: *

https://pan.baidu.com/s/14k-9jsspp43ZhMxqPmsWMQ（sogou pretrained）

More Chinese Word Vectors: https://github.com/Embedding/Chinese-Word-Vectors
 
 - Design your own dataset in /dataTest/data/
 
 - Put the pretrained file in /dataTest/data/


###

<br>

### Train the model

To train the model, run command below. 

If you want change another model, edit the model in models directory and change the default in train.py.

```bash
$ python train.py
```
<br>

<br>

### Evaluate the model

Edit test.txt and run eval.py.

```bash
$ python eval.py
```

<br/>

## Results
<br/>

#### CNN3

```
Test Loss:  0.72,  Test Acc: 71.58%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

        Love     0.7326    0.7829    0.7569       175
        Fear     0.7327    0.8030    0.7663       198
         Joy     0.6716    0.6207    0.6452       145
     Sadness     0.6923    0.5094    0.5870        53
    Surprise     0.0000    0.0000    0.0000         1
       Anger     0.0000    0.0000    0.0000         5

    accuracy                         0.7158       577
    
```

#### RNN2
```
Test Loss:  0.91,  Test Acc: 67.76%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

        Love     0.6256    0.8114    0.7065       175
        Fear     0.7333    0.7222    0.7277       198
         Joy     0.6822    0.6069    0.6423       145
     Sadness     0.6923    0.3396    0.4557        53
    Surprise     0.0000    0.0000    0.0000         1
       Anger     0.0000    0.0000    0.0000         5

    accuracy                         0.6776       577
   macro avg     0.4556    0.4134    0.4220       577
weighted avg     0.6764    0.6776    0.6673       577
```




#### Other lower acc model: FastText(59.12%),Transformer(60.00%),RNN3(CNN+LSTM+GRU/LSTM)

