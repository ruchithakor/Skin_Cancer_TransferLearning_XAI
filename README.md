# Skin Cancer Class | Transfer Learning | XAI

This repository contains python codes and documentations for the project - "Empirical analysis of CNN & Transfer Learning Models with focus on Explainable AI techniques for 
interpretable Skin Cancer Classification". This project compared traditional CNN models with advanced transfer learning models such as EfficientNet_B0, EfficientNet_v2_B0, 
ResNet50, VGG16, and Vision Transformers. Further few Explainable AI like GRAD-CAM, LIME and Attention mapping on baseline CNN model and best performer - Vision Transformer model.

For this project, I have used Kaggle version of skin cancer classification data from International Skin Imaging Collaboration(ISIC). It is a collaboration between academia and industry, aims to advance the use of digital skin imaging to help lower the death rate from melanoma. Data is available on "data" folder in this repository or it isalso avialable on [Kaggle](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign).

### Implementation

Prerequisites -  tensorflow(version between 2.13.0 - 2.16.0), Keras, tensorflow-hub, tensorflow_addons, vit-keras, tf-keras-vis, lime, scikit-plot, matplotlib, seanborn numpy, pandas.

Recommended to execute on P-100 GPUs.

All the implementation is originally done in jupyter notebooks. There already executed notebooks with outputs are available in "jupyter_notebooks" folder.

These notebooks are transformed into .py(python files) and all are available in main repository. File structures and execution details are provided below:

#### Load data. EDA and pre-processing

1. [load_data.py](load_data.py): This file sets paths to the location where data is located and fetch paths for each iamge and respective labels in train and test dataframes for ease of use. No need to run this file seperately, each model execution file will directly call the function to load data.
3. [load_image.py](load_image.py): This file contains function to loas/view image from given image_path. This fucnction is used in multiple places to visualize images, create data tensors etc.
4. [eda.py](eda.py): Run this file to see Exploratory Data Analysis on the dataset. It will perform tasks to check class balance and visualise sample images from training data. These results are stored in result folder - "[class_balance.pdf](results/class_balance.pdf)", "[sample_images.pdf](results/sample_images.pdf)"
5. [pre_processing.py](pre_processing.py): This file pre-process the data (split training data into train and validation sets, prepare tensors to fit in models, preapre batches) and returns train_dataset, val_dataset and test_dataset. There is no need to run this file seperately. Each model execution file will use this function to get processed datasets. 

#### Training-testing models and implementing XAI methods

1. [model_cnn.py](model_cnn.py): Execute this file to train baseline Vanilla CNN model on training data, evaluate on testing data and implement XAI methods - GRAD-CAM and LIME. There is no need to pass any arguments. After execution all the results will be stored in results folder - [CNN_CM.pdf](results/CNN_CM.pdf), [CNN_ROC.pdf](results/CNN_ROC.pdf), [CNN_GRAD.pdf](results/CNN_GRAD.pdf), [CNN_LIME.pdf](results/CNN_LIME.pdf)
2. [model_efficientnet_b0.py](model_efficientnet_b0.py): Run this file to train EfficientNet_B0 model on training data and evaluate with testing data. There is no need to pass any arguments. After execution all the results will be stored in results folder - [EfficientNet_CM.pdf](results/EfficientNet_CM.pdf), [EfficientNet_ROC.pdf](results/EfficientNet_ROC.pdf)
3. model_efficientnet_v2_b0.py
4. model_resnet50.py
5. model_vgg.py
6. model_vit.py

#### Data similarit check is performed in "data_similarity.py" file. It store results of PCA visualisation and Image clustering in results folder - "PCA_data.pdf", "Image_clustering_data.pdf"
