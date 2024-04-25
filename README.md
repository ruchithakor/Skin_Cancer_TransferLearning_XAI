# Skin Cancer Class | Transfer Learning | XAI

This repository contains python codes and documentations for the project - "Empirical analysis of CNN & Transfer Learning Models with focus on Explainable AI techniques for 
interpretable Skin Cancer Classification". This project compared traditional CNN models with advanced transfer learning models such as EfficientNet_B0, EfficientNet_v2_B0, 
ResNet50, VGG16, and Vision Transformers. Further few Explainable AI like GRAD-CAM, LIME and Attention mapping on baseline CNN model and best performer - Vision Transformer model.

For this project, I have used Kaggle version of skin cancer classification data from International Skin Imaging Collaboration(ISIC). It is a collaboration between academia and industry, aims to advance the use of digital skin imaging to help lower the death rate from melanoma. Data is available on "data" folder in this repository or it isalso avialable on [Kaggle](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign).

### Implementation

Prerequisites -  tensorglow(version between 2.13.0 - 2.16.0), Keras, tensorflow-hub, tensorflow_addons, vit-keras, tf-keras-vis, lime, scikit-plot, matplotlib, seanborn numpy, pandas.

Recommended to execute on P-100 GPUs.

All the implementation is originally done in jupyter notebooks. There already executed notebooks with outputs are available in "jupyter_notebooks" folder.

These notebooks are transformed into .py(python files) and all are available in main repository. File structures and execution details are provided below:

#### Load data. EDA and pre-processing

1. load_data.py
2. load_image.py
3. eda.py
4. pre_processing.py

#### Training-testing models and implementing XAI methods

1. model_cnn.py
2. model_efficientnet_b0.py
3. model_efficientnet_v2_b0.py
4. model_resnet50.py
5. model_vgg.py
6. model_vit.py

#### Data similarit check is performed in "data_similarity.py" file. It store results of PCA visualisation and Image clustering in results folder - "PCA_data.pdf", "Image_clustering_data.pdf"
