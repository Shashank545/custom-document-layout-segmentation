## Introduction


In this notebook, we are going to fine-tune the pre-trained LayoutLM model by Microsoft Research on the FUNSD dataset, which is a collection of annotated form documents. The goal of our model is to learn the annotations of a number of labels (`question`, `answer`, `header` and `other`) on those forms, such that it can be used to identify form types in  unseen forms in the future during general inferencing.


## Steps to install dependencies

Currently you have to first install the `unilm` package, and then the `transformers` package (which updates the outdated `transformers` package that is included in the `unilm` package). The reason we also install the `unilm` package is because we need its preprocessing files. I've forked it, and removed some statements which introduced some issues.

To install UniLM
```code
! rm -r unilm
! git clone -b remove_torch_save https://github.com/NielsRogge/unilm.git

```
To install Pytorch and LayoutLM
```code
! pip install torch
! cd unilm/layoutlm
! pip install unilm/layoutlm
```

To install transformers from the LayoutLM compatibel version

```code
! rm -r transformers
! git clone https://github.com/huggingface/transformers.git
! cd transformers
! pip install ./transformers
! pip install pillow
```

## Download the FUNSD Dataset

Here we download the data of the [FUNSD dataset](https://guillaumejaume.github.io/FUNSD/) from the web. This results in a directory called "data" being created, which has 2 subdirectories, one for `training` and one for `testing`. Each of those has 2 subdirectories in turn, one containing the images as png files and one containing the annotations in json format.

```code
! wget https://guillaumejaume.github.io/FUNSD/dataset.zip
! unzip dataset.zip && mv dataset data && rm -rf dataset.zip __MACOSX
```


## Fine Tuning Pretrained LayoutLM model

As this is a sequence labeling task, we are going to load `LayoutLMForTokenClassification` (the base sized model) from the hub. We are going to fine-tune it on a downstream task, namely `FUNSD` dataset to assign 4 types of labels namely (`question`, `answer`, `header` and `other`) to new form document image.

#### Downloading and caching pretrained model

```code
from transformers import LayoutLMForTokenClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=num_labels)
model.to(device)
```



## Evaluating the fine-tuned model with `seqeval`

[seqeval](https://pypi.org/project/seqeval/) is a Python framework for sequence labeling evaluation. seqeval can evaluate the performance of chunking tasks such as named-entity recognition, part-of-speech tagging, semantic role labeling and so on.

This is well-tested by using the Perl script conlleval, which can be used for measuring the performance of a system that has processed the CoNLL-2000 shared task data.

Following metrics were used for evaluation purpose

|metrics	|description|
|----|----|
|accuracy_score(y_true, y_pred)	|Compute the accuracy|
|precision_score(y_true, y_pred)	|Compute the precision|
|recall_score(y_true, y_pred)	|Compute the recall|
|f1_score(y_true, y_pred)	|Compute the F1 score, also known as balanced F-score or F-measure|

#### How to use seqeval

```code
>>> from seqeval.metrics import accuracy_score
>>> from seqeval.metrics import classification_report
>>> from seqeval.metrics import f1_score
>>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
>>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
>>> f1_score(y_true, y_pred)
0.50
>>> classification_report(y_true, y_pred)
              precision    recall  f1-score   support

        MISC       0.00      0.00      0.00         1
         PER       1.00      1.00      1.00         1

   micro avg       0.50      0.50      0.50         2
   macro avg       0.50      0.50      0.50         2
weighted avg       0.50      0.50      0.50         2

```


## If you like this work, you may also like my blogs

Do not forget to follow me at, [Shashank's Medium Profile](https://medium.com/@Immaculate_sha2nk)
Also, consider to share your valuable comments and feedback by [BUYINGmePIZZA](https://www.buymeacoffee.com/mrtensorllm)
You can post me your business enquiries and suggestions at >>>   `mr.tensorflow@gmail.com`
