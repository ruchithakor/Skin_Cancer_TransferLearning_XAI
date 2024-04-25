# Skin Cancer Class | Transfer Learning | XAI

This repository contains python code and documentation for the project - "Empirical analysis of CNN & Transfer Learning Models with focus on Explainable AI techniques for 
interpretable Skin Cancer Classification". This project compares traditional CNN models with advanced transfer learning models such as EfficientNet_B0, EfficientNet_v2_B0, 
ResNet50, VGG16, and Vision Transformers. It also includes few Explainable AI techniques like GRAD-CAM, LIME and Attention mapping on baseline CNN model and best performer - Vision Transformer model.

For this project, I have used Kaggle version of skin cancer classification data from International Skin Imaging Collaboration(ISIC). It is a collaboration between academia and industry, aims to advance the use of digital skin imaging to help lower the death rate from melanoma. The data is available in the "data" folder in this repository on [Kaggle](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign).

### Implementation

* Prerequisites -  tensorflow(version between 2.13.0 - 2.16.0), Keras, tensorflow-hub, tensorflow_addons, vit-keras, tf-keras-vis, lime, scikit-plot, matplotlib, seanborn numpy, pandas.
* Recommendation -  Execute on P-100 GPUs.

* All the implementation is originally done in jupyter notebooks. Executed notebooks with outputs are available in "jupyter_notebooks" folder for reference.
* These notebooks are transformed into .py(python files) as per requirement and are available in the main repository. File structures and execution details are provided below:

#### Load data. EDA and pre-processing

1. [load_data.py](load_data.py): Sets paths to the data location and fetches paths for each iamge and respective labels in train and test dataframes for ease of use. There is no need to run this file seperately, each model execution file will directly call the function from file to load data.
3. [load_image.py](load_image.py): Contains function to load/view image from given image path. This fucnction is used in multiple places to visualize images, create data tensors etc.
4. [eda.py](eda.py): Run this file to perform Exploratory Data Analysis on the dataset. It will check class balance for train and test data, and visualise sample images from training data. These results are stored in results folder - "[class_balance.pdf](results/class_balance.pdf)", "[sample_images.pdf](results/sample_images.pdf)".
5. [pre_processing.py](pre_processing.py): Pre-processes the data (splits training data into train and validation sets, prepares tensors to fit in models, preapres batches) and returns train_dataset, val_dataset and test_dataset. There is no need to run this file seperately. Each model execution file will use the function from file to get processed datasets.

#### Training-testing models and implementing XAI methods

1. [model_cnn.py](model_cnn.py): Execute this file to train baseline Vanilla CNN model on training data, evaluate on testing data and implement XAI methods - GRAD-CAM and LIME. There is no need to pass any arguments. Results will be stored in results folder - "[CNN_CM.pdf](results/CNN_CM.pdf)", "[CNN_ROC.pdf](results/CNN_ROC.pdf)", "[CNN_GRAD.pdf](results/CNN_GRAD.pdf)", "[CNN_LIME.pdf](results/CNN_LIME.pdf)".
2. [model_efficientnet_b0.py](model_efficientnet_b0.py): Run this file to train EfficientNet_B0 model on training data and evaluate with testing data. There is no need to pass any arguments. Results will be stored in results folder - "[EfficientNet_CM.pdf](results/EfficientNet_CM.pdf)", "[EfficientNet_ROC.pdf](results/EfficientNet_ROC.pdf)".
3. [model_efficientnet_v2_b0.py](model_efficientnet_v2_b0.py): Run this file to train EfficientNet_v2_B0 model with transfer Learning model on training data and evaluate with testing data. There is no need to pass any arguments. Results will be stored in results folder - "[EfficientNet_v2_CM.pdf](results/EfficientNet_v2_CM.pdf)", "[EfficientNet_v2_ROC.pdf](results/EfficientNet_v2_ROC.pdf)".
4. [model_resnet50.py](model_resnet50.py): Execute this file to train ResNet50 model with transfer Learning model on training data and evaluate with testing data. There is no need to pass any arguments. Results will be stored in results folder - "[ResNet50_CM.pdf](results/ResNet50_CM.pdf)", "[ResNet50_ROC.pdf](results/ResNet50_ROC.pdf)".
5. [model_vgg.py](model_vgg.py): Run this file to train enhanced version of VGG16 model using Transfer Learning on training data and evaluate on testing data. There is no need yo pass any arguments. Results will be stored in results folder - [VGG16_CM.pdf](results/VGG16_CM.pdf), [VGG16_ROC.pdf](results/VGG16_ROC.pdf)
6. [model_vit.py](model_vit.py): Run this file to train Vision Transformer model with Transfer Learning on training data, evaluate with testing data. Further XAI methods- Attention mapping and LIME is implemented on sampled data. There is no need ot pass any arguments. Results are stored in results folder - "[Vit_Cm.pdf](results/Vit_Cm.pdf)", "[ViT_ROC.pdf](results/ViT_ROC.pdf)", "[ViT_LIME.pdf](results/ViT_LIME.pdf)", "[ViT_Attentions_0.pdf](results/ViT_Attentions_0.pdf)", "[ViT_Attentions_5.pdf](results/ViT_Attentions_5.pdf)", "[ViT_Attentions_final.pdf](results/ViT_Attentions_final.pdf)"
7. [data_similarity.py](data_similarity.py): Performs data similarity check. It store results of PCA visualisation and Image clustering in results folder - "[PCA_data.pdf](results/PCA_data.pdf)", "[Image_clustering_data.pdf](results/Image_clustering_data.pdf)"

#### Results

[Results](results) folder contains all the results of EDA, Evaluation results for all models, XAI results and Data similarity check.

#### Reports

* Final report is located in reports folder - [DS8013_project_report_parmar.pdf](reports/DS8013_project_report_parmar.pdf)
* Initial proposal - [DS8013_project_proposal_Parmar.pdf](reports/DS8013_project_proposal_Parmar.pdf) (Scope of the project is expaned to add implementation of Explainable AI).
