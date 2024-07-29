# ICVQG

## Requirements

1.refer to https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch

2.refer to https://github.com/wuzhe71/CPD

3.imageio spacy spacy_lookup sentence_transformers pandas random

## Preliminary Scene Graph Generation

We firstly follow Tang et al. to generate a preliminary scene graph and add the classification of attributes to the model during training. The data download, requirements and training are consistent with the original project, please refer to https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch

According to our changes, the generated scene graph will contain the detected objects with bounding boxes and attribute vectors, the confidence of the objects, the relationship triplets, and the confidence of the relations.

## Core Scene Graph Generation

1. We extract information from the text to adjust the confidence in the preliminary scene graph. testaddarticap.py, testaddcap.py, and testflickr.py respectively use the artificial captains, the captains sampled in VG, and the captains in Flickr30k. Run them and then run sortrel.py to sort the elements before the evaluation in ablation study.

2. The core scene graph is generated according to the adjusted confidence and salient object detection. gmnewsg.py, gmgtsg.py, and gmflickrsg.py can respectively generate the core scene graph from our data, ground truth graph, and Flickr30k. The salient object detection follows Wu et al. Please refer to  https://github.com/wuzhe71/CPD  and download the pretrained model CPD.th

## Question Generation

Run generate_questions_new.py to generate questions. Set the --input_scene_dir as the path of the core scene graph generated above. We have preseted a part of the designed function templates in the code.

Run generate_questions_new_gt.py to generate questions in ablation study. Set the --input_scene_dir as the path of the ground truth scene graph. 

## Evaluation

The detailed evaluation method is introduced in our paper.

Run evl.py and evlgt.py to evaluate on our new metrics.

Run evlrouge.py to evaluate on ROUGE.

Run hevldata.py to prepare  the data and then run hevltk.py for human evaluation.

Run reformimg.py and reformq.py to build our new dataset InVQA.

## Dataset

The method can be used to construct VQA data. Please download from the link below:

https://pan.baidu.com/s/1iA0ehEn141xZH6l4grBTrQ  (Extraction code: qmpb)

## Note

Since our module is detachable, the code is currently rough. We will clean it and unify the saved path soon.

A complete and clear version will be published after the paper is published.
