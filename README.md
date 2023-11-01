# CSDNet

CSDNet: Image Inpainting with Contextual and Spatial Coherence for Sequential Traffic Scenarios

https://github.com/chappyer/CSDNet/assets/83496119/2b07fddd-2059-483f-8a23-41829eb913a4

## Prerequisites

- Linux or Windows
- Python 3
- NVIDIA GPU (16G memory or larger) + CUDA cuDNN

Clone this reop

```
git clone https://github.com/chappyer/CSDNet
cd CSDNet
```

## Getting Started

### Installation

This code requires some dependencies. Please install dependencies by

`pip install -r requirements.txt`

### Dataset Preparation 

Before training the model, the dataset needs to be prepared

The dataset should be divided into the following forms

- `train`
  - `origin_1` The original view images that include all the training images
  - `origin_2` The neighbor original images that include all the training images
  - `origin_1_ref` The original view masked images that include all images in the dataset
  - `origin_2_ref` The neighbor original images that include all images in the dataset
  - `real_semantic_view_1` The original view semantic segmentation images that include all images in the dataset
  - `real_semantic_view_2` The neighbor original view semantic segmentation images that include all images in the dataset
  - `mask_semantic` The single class original view semantic segmentation images that include all images in the dataset
- `test`
  - The file structure of the testing dataset is the same as the training dataset, the only difference is that you need to change the images from training to testing.


### Training
Training on your own dataset

```
python train.py --name [train_model_name] --model pix2pixflow --dataset_mode kittiflow --dataroot [path for training dataset] --batchSize 2 --gpu_ids 0
```

### Testing

After training your own model, the model can be tested by the following command

```
python test.py --name [test_model_name] --model pix2pixflow --dataset_mode kittiflow --dataroot [path for testing dataset] --preprocess_mode [choose your preprocess mode] --how_many [select the number of images the program tests at a time] --batchSize 2 --gpu_ids 0
```

## More Options

There are many hyperparameters in the dictionary that can be customized to facilitate the parameter adjustment of model training and testing.

- `options`
  - `base_options` 
    - The basic parameters of the model. The modification of these parameters will take effect in both model training and testing.
  
  - `train_options` 
    - The training parameters of the model. The modification of these parameters will only take effect in both model training.
  
  - `test_options` 
    - The training parameters of the model. The modification of these parameters will only take effect in both model testing.
  


## Code Structure

- `CSDNet`
    - `data`
        - The dataset processing module. This module converts images in the dataset into tensor that can be trained by the model.

    - `models`
        - The neural network module. This module defines the architecture of all models. 

    - `trainers`
        - This module is responsible for creating the network and reporting the training/testing process information.

    - `util`
        - The util module contains code that has nothing to do with model training, such as visualization tools.

    - `scripts`
        - This module contains configuration information for model training and testing in scripts.

    - `train.py`, `test.py` The entry files for model training and testing.

## Training/Matching Time

The specific training and matching time of the model under the specified dataset is given below for **reference only**.

- **Dataset Information**

  - The image size for both training and matching is 1120*320.
  - Training Dataset size is 3594.
  - Matching Dataset size is 1065.

- **Hardware Information**

  - For training part, we training on the 4 Nvidia Tesla V100s 32G GPU.
  - For matching part，we matching on the Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz.
    - Matching with GPU can greatly decreasing matching time, but requires a graphics card with larger memory (larger than 32G).
    - *This data is matching with GPU (One Nvidia Tesla V100s 32G GPU)*. [*]

- **A reference table for training and matching time is given below**

  **Table One** Training/ Matching Time With Different Patch Size 

  - Training/Matching Configuration: 1. Use semantic 2. Use coarse match 3. Trainging/Matching image size is 1120*320 4. Training/Matching on One View

| Patch Size | Matching Time (Sec/One Image) | Training Time (Min/One Epoch) |
| ---------- | ----------------------------- | ----------------------------- |
| 3x3        | 33.10                         | 29.57                         |
| 7x7        | 31.76                         | 17.26 (0.61sec on GPU*)       |
| 11x11      | 30.18                         | 15.02                         |

[\*] *This data is matching with GPU (One Nvidia Tesla V100s 32G GPU)*

​	**Table Two** Training/ Matching Time With Different View 

- Training/Matching Configuration: 1. Use semantic 2. Use coarse match 3. Trainging/Matching image size is 1120*320 4. All Patch Size is 7\*7


| Views | Matching Time (Sec/One Image) | Training Time (Min/One Epoch) |
| ----- | ----------------------------- | ----------------------------- |
| 4     | 35.55                         | 69.89                         |
| 3     | 33.55                         | 57.65                         |
| 2     | 32.19                         | 38.75                         |
| 1     | 31.76                         | 17.26 (0.61sec on GPU*)       |

[\*] *This data is matching with GPU (One Nvidia Tesla V100s 32G GPU)*







