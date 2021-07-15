# Dual Gradient based Snow Attentive Desnowing

### TODO List

:octocat:

- Official code for Dual Gradient based Snow Attentive Desnowing
 
### Introduction 

- A pytorch implementation of dual-gradient-based snow-attentive desnowing

### Requirements
- libtiff
- matplotlib
- pytorch >= 1.5.0


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
  --root                The root path that contains the dataset
```

- for demo

```
# python demo.py 
usage: demo.py [-h] [--demo_img_path] [--demo_img_type] [--save] [--visualization]

  -h, --help            show this help message and exit
  --demo_type       The path that contains the image you want to detect
  --demo_path       The path that contains the image you want to detect (available .jpg, .png, .tif)
```
