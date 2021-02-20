<img src='paper/shinjuku.jpg' align="left" width=1000>

<br><br><br>

# [CVPR2020]Learning to Cartoonize Using White-box Cartoon Representations
[project page](https://systemerrorwang.github.io/White-box-Cartoonization/) |   [paper](https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/paper/06791.pdf) |   [twitter](https://twitter.com/IlIIlIIIllIllII/status/1243108510423896065) |   [zhihu](https://zhuanlan.zhihu.com/p/117422157) |   [bilibili](https://www.bilibili.com/video/av56708333) |  [facial model](https://github.com/SystemErrorWang/FacialCartoonization)

- Tensorflow implementation for CVPR2020 paper “Learning to Cartoonize Using White-box Cartoon Representations”.
- Improved method for facial images are now available:
- https://github.com/SystemErrorWang/FacialCartoonization

<img src="images/method.jpg" width="1000px"/>
<img src="images/use_cases.jpg" width="1000px"/>

## Use cases

### Scenery
<img src="images/city1.jpg" width="1000px"/>
<img src="images/city2.jpg" width="1000px"/>

### Food
<img src="images/food.jpg" width="1000px"/>

### Indoor Scenes
<img src="images/home.jpg" width="1000px"/>

### People
<img src="images/person1.jpg" width="1000px"/>
<img src="images/person2.jpg" width="1000px"/>

### More Images Are Shown In The Supplementary Materials


## Online demo

- Some kind people made online demo for this project
- Demo link: https://cartoonize-lkqov62dia-de.a.run.app/cartoonize
- Code: https://github.com/experience-ml/cartoonize
- Sample Demo: https://www.youtube.com/watch?v=GqduSLcmhto&feature=emb_title

## Prerequisites

- Training code: Linux or Windows
- NVIDIA GPU + CUDA CuDNN for performance
- Inference code: Linux, Windows and MacOS


## How To Use

### Installation

- Assume you already have NVIDIA GPU and CUDA CuDNN installed 
- Install the python3 requirements with `pip install -r requirements.txt` or if you have pipenv then run `pipenv install`
- Tested with python version 3.8


### Inference with Pre-trained Model

- The code can all be run throught the `test_code/run.py` and is simple to use, for extra help you can run `cd test_code && python3 run.py -h`
- Running the example images: `cd test_code && python3 run.py -i test_images/actress2.jpg` the image should appear but also be saved to `test_code/cartoonize_output`
- Running example video: `cd test_code && python3 run.py -v test_images/color_clip.mp4` the video will also be displayed and saved to `test_code/cartoonize_output`

### Train

- Place your training data in corresponding folders in /dataset 
- Run pretrain.py, results will be saved in /pretrain folder
- Run train.py, results will be saved in /train_cartoon folder
- Codes are cleaned from production environment and untested
- There may be minor problems but should be easy to resolve
- Pretrained VGG_19 model can be found at following url:
https://drive.google.com/file/d/1j0jDENjdwxCDb36meP6-u5xDBzmKBOjJ/view?usp=sharing



### Datasets

- Due to copyright issues, we cannot provide cartoon images used for training
- However, these training datasets are easy to prepare
- Scenery images are collected from Shinkai Makoto, Miyazaki Hayao and Hosoda Mamoru films
- Clip films into frames and random crop and resize to 256x256
- Portrait images are from Kyoto animations and PA Works
- We use this repo(https://github.com/nagadomi/lbpcascade_animeface) to detect facial areas
- Manual data cleaning will greatly increace both datasets quality

## Acknowledgement

We are grateful for the help from Lvmin Zhang and Style2Paints Research

## License
- Copyright (C) Xinrui Wang All rights reserved. Licensed under the CC BY-NC-SA 4.0 
- license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
- Commercial application is prohibited, please remain this license if you clone this repo

## Citation

If you use this code for your research, please cite our [paper](https://systemerrorwang.github.io/White-box-Cartoonization/):

@InProceedings{Wang_2020_CVPR,
author = {Wang, Xinrui and Yu, Jinze},
title = {Learning to Cartoonize Using White-Box Cartoon Representations},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}


# 中文社区

我们有一个除了技术什么东西都聊的以技术交流为主的宇宙超一流二次元相关技术交流吹水群“纸片协会”。如果你一次加群失败，可以多次尝试。

    纸片协会总舵：184467946
