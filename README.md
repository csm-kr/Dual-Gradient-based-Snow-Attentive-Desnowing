# Dual Gradient based Snow Attentive Desnowing

:octocat: Official code for IEEE Access "Dual Gradient based Snow Attentive Desnowing" in Pytorch
 
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10070780
 
### Introduction 

- A pytorch implementation of dual-gradient-based snow-attentive desnowing

### Requirements
- libtiff
- matplotlib
- pytorch >= 1.5.0
- visdom
- torchvision
- argparse


### training

- batch : 2
- scheduler : ExponentialLR
- dataset : snow100k dataset
- iterations : 300k
- gpu : nvidia geforce rtx 2080ti 
 
### Results

- qualitative reuslt

|Subset                    | Snow100K-S (PSNR/SSIM) | Snow100K-M (PSNR/SSIM) | Snow100K-M (PSNR/SSIM) |   
|--------------------------|------------------------|------------------------|------------------------| 
|DesnowNet                 | 32.3331 / **0.95**     |  30.8692 / **0.9409**  | 27.1699 / **0.8983**   |
|Ours                      | **34.5051** / 0.9458   | **32.7869** / 0.9345   | **28.7576** / 0.8843   |

- quantitative reuslt

![](./demo/desnow_sidewalk%20winter%20-grayscale%20-gray_01857.jpg)
![](./real_snow_img/sidewalk%20winter%20-grayscale%20-gray_01857.jpg)
![](./demo/desnow_sidewalk%20winter%20-grayscale%20-gray_02810.jpg)
![](./real_snow_img/sidewalk%20winter%20-grayscale%20-gray_02810.jpg)


### Quick Start Guide

before test and demo, we must have trained file(weight params) 
you have to 2 weight to run this model, 
- classification model weight : https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/EcpFP9Up9-ZNvQ5k5Pj9DQEB_UdJH0CoDMhOI-d3EFSGAA?e=3fNs5X

- dehaze model weight : https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/EV02szOp7ElKjpVkXsOEOlYBhwJGi5XuBHvyZXVNVWKgYA?e=P7Kqn1

- and then make ./models file place the weights in the file.

- for testing

```
# python test.py 
usage: test.py [-h] [--data_type] [--root] 

  -h, --help            show this help message and exit
  --data_type           which dataset you want to use snow100k or srrs (default='snow100k')
  --root                The dataset root path 
```

- for demo

```
# python demo.py 
usage: demo.py [-h] [--demo_path] [--demo_type] 
               [--no_save] [--no_vis]

  -h, --help        show this help message and exit
  --demo_type       which dataset you trained snow100k or srrs (default='snow100k')
  --demo_path       The path that contains the images you want to desnow (available .jpg, .png, .tif type)
  --no_save         When you don't want to save (save as .jpg type) (default=True)
  --no_vis          When you don't want to display (default=True)
```
