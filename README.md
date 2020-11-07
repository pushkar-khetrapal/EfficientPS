# EfficientPS

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11Q4H-5nYq6F6K8tVqfSB1tl1adDQPwsB?usp=sharing)


[EfficientPS](http://panoptic.cs.uni-freiburg.de/) was developed by Rohit Mohan, Abhinav Valada ( University of Freiburg, Germany ) in 2020. EfficientPS is a Panoptic Segmentation architecture which includes Mask R-CNN, Semantic Segmentation and fusion of both outputs. Currently, their paper is under review so, the code isn't publicly available. Therefore, I decided to write code from scratch. All the code is written in Pytorch. Panoptic Segmentation was firstly developed by Kirillov in 2019 [link](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf) In order to understand Panoptic Segmentation, First we need to understand what's things and what's object in a scene. Let's suppose you are driving a car and you see sky, trees, cars, humans, bikes, buildings etc. Classify each pixel into it's category we call them "things". But we can't count number of cars, humans, bikes from the things. To count the number of objects we have look on groups of pixels not every pixels. This is called "objects". The things we can get from semantic segmentation and for localization we have to look into instance segmentation. Instance segmentation is nothing but bounding boxes with mask of every object. By merging the outputs from both heads pixelwise we can get Panoptic Segmentation.

## Architecture

![](/media/efficientpsarchitecture.png)

The Architecture contains Mask R-CNN for instance segmentation and semantic segmentation, which is based on Large Scale Feature Extractor (LSFE), DPC (Dense Prediction Cell) and MC (Mismatch Correction). Both heads have same backbone of EfficientNet-B5. The output of backbone block 2, 3, 5, and 9 corresponds to downsampling factors ×4,×8,×16 and ×32 respectively with respect to image size. These four outputs further feed into 2 way-FPN to correlated feature map from all scales. Further these outputs fed into both heads. They removed SE (Squeeze-and-Excitation) layers from backbone. All the activation and batchnorm layers were replaced by iABN Sync layer (in-Place activation annd batchnorm layer) with leaky relu as activation. The output of 2 way-FPN further feed into semantic head, Region Proposal Network, RoI align.

### Semantic Head

The approach used in the paper:
1. Large Scale Feature Extractor (LSFE) module for large-scale.
2. Modified DPC module for small-scale, the network should be able to capture long-range context. 
3. Mismatch Correction Module (MC) which helps in correlating the small-scale features with respect to large-scale features.

### Instance Head
Instance is similar to Mask R-CNN. But they conserve 2.09M parameters in comparison to the conventional Mask R-CNN by replacing convolutional to seperable conventional layer. 

## To do list

1. Due to less availablity of resources, the training is not done yet. 

## How to use code
