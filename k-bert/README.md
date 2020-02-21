# References

pytorch implementation

[k-bert](https://github.com/autoliuweijie/K-BERT)

[URE](https://github.com/dbiir/UER-py)


## Requirements

Software:
```
Python3
Pytorch >= 1.0
argparse == 1.1
pkuseg == 0.0.22
```


## Prepare

* Download the ``google_model.bin`` from [here](https://share.weiyun.com/5GuzfVX), and save it to the ``models/`` directory.
* Download the ``CnDbpedia.spo`` from [here](https://share.weiyun.com/5BvtHyO), and save it to the ``brain/kgs/`` directory.
* Optional - Download the datasets for evaluation from [here](https://share.weiyun.com/5Id9PVZ), unzip and place them in the ``datasets/`` directory.

The directory tree of K-BERT:
```
K-BERT
├── brain
│   ├── config.py
│   ├── __init__.py
│   ├── kgs
│   │   ├── CnDbpedia.spo
│   │   ├── HowNet.spo
│   │   └── Medical.spo
│   └── knowgraph.py
|
├── models
│   ├── google_config.json
│   ├── google_model.bin
│   └── google_vocab.txt
├── outputs
├── uer
├── README.md
├── run_kbert_cls.py
```

## Data
Some news contained people's emotional vote after reading the whole news or part of the news.

There were 6 choice of mood to choose : **Love, Fear, Joy, Sadness, Surprise, Anger**.

**News and its emotional label** were collected from websites.

**A part(now only shows the beginning of the news)** of these news are cut out as train data and test data.


## K-BERT for text classification

### Classification example

Run example on Book review with CnDbpedia:
```sh
CUDA_VISIBLE_DEVICES='0' nohup python3 -u run_kbert_cls.py \
    --pretrained_model_path ./models/google_model.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --train_path ./datasets/book_review/train.tsv \
    --dev_path ./datasets/book_review/dev.tsv \
    --test_path ./datasets/book_review/test.tsv \
    --epochs_num 2 --batch_size 32 --kg_name CnDbpedia \
    --output_model_path ./outputs/kbert_bookreview_CnDbpedia.bin \
    > ./outputs/kbert_bookreview_CnDbpedia.log &
```

Results:
```
Best accuracy in dev : 62.80%
Best accuracy in test: 62.96%
```
```
            precision    recall      f1      support
          
     Love:    0.684      0.798      0.737      258
     Fear:    0.636      0.636      0.636      231
      Joy:    0.621      0.554      0.585      195
  Sadness:    0.421      0.375      0.397      64
 Surprise:    0.667      0.333      0.444      12
    Anger:    0.286      0.174      0.216      23
   
 accuracy:                          0.6296     783    
```

