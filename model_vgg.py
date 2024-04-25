"""
DS 8013 - Deep Learning - Final Project
Ruchi Parmar - 501034872

This code builds and train Transfer Learning-enhanced version of VGG16 model from vit-keras, check validation and test accuracy and other performance parameters
"""

# Import required packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scikitplot.metrics import plot_roc

import tensorflow as tf
from keras import layers, Sequential
from keras.applications.vgg16 import VGG16

from load_data import load_data
from load_image import load_image
from pre_processing import pre_processing, pre_process_df

def vgg16_model_sequential():
    """Initialize and compile the model

    Returns:
        model: compiled VGG16 model
    """
    # Load the VGG16 Model model from keras
    vgg16 = VGG16(input_shape = (224,224,3), weights='imagenet', include_top = False, pooling= 'max')

    # Freeze the convolutional base
    vgg16.trainable = False

    # transfer the features learnt from VGG16 to out model
    vgg16_model_sequential = Sequential([
        layers.Input(shape = (224, 224, 3), dtype = tf.float32, name = 'input_image'), # input layer
        vgg16, # vgg16 layer
        layers.Dropout(0.2), # dropout layer
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)), # dense layer
        layers.Dense(2, dtype=tf.float32, activation='sigmoid')], name='resnet50_sequential_model') # dense layer 2

    tf.random.set_seed(42) # set random seed

    # Compile the model
    vgg16_model_sequential.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                                  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
                                  metrics = ["accuracy"])

    return vgg16_model_sequential # return the model

def train_model(vgg16_model, train_dataset, val_dataset,  test_dataset):
    """train the model on training data and predict for test data

    Args:
        vgg16_model (model): compiled vgg16 model
        train_dataset (tensor data): tensors of training data
        val_dataset (tensor data): tensors of validaiton data
        test_dataset (tensor data): tensors of testing data
    """
    # Define early stopping mechanism wiht 5 step patience with val_loass
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

    # train model with training data with 50 epochs and early stopping mecahnism
    vgg16_training = vgg16_model.fit(train_dataset, epochs = 50, validation_data = val_dataset, validation_steps = int(len(val_dataset)),
                                 callbacks = early_stopping)

    # Evaluate the model on test dataframe
    vgg16_evaluation = vgg16_model.evaluate(test_dataset)

    # Generate model probabilities and associated predictions
    vgg16_test_prob = vgg16_model.predict(test_dataset, verbose=1)
    vgg16_test_pred = tf.argmax(vgg16_test_prob, axis=1)

    return vgg16_training, vgg16_test_prob, vgg16_test_pred

def evaluate_model(test_df, vgg16_test_prob, vgg16_test_pred):
    """evaluate the mdoel with test predictions and labels
    """
    # generate & plot confusion matrics to check False positives & False negatuves
    vgg16_confusion_matrix = confusion_matrix(test_df.label_encoded, vgg16_test_pred)

    # Set plot size
    plt1 = plt.figure(figsize = (4, 4))

    # Create confusion matrix heatmap
    disp = sns.heatmap(vgg16_confusion_matrix, annot = True, cmap = 'Blues',
                       annot_kws={"size": 6}, fmt='g',
                       linewidths=1, linecolor='black', clip_on=False,
                       xticklabels = ['benign', 'malignant'], yticklabels = ['benign', 'malignant'])

    # Set title and axis labels
    disp.set_title('VGG16 - Confusion Matrix', fontsize = 14)
    disp.set_xlabel('Predicted Label', fontsize = 10)
    disp.set_ylabel('True Label', fontsize = 10)
    plt.yticks(rotation=0)

    # save vgg16 confusion matrix to results folder
    plt1.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/VGG16_CM.pdf")

    # plot roc curve
    plt2 = plot_roc(test_df.label_encoded, vgg16_test_prob, figsize=(6, 6), title_fontsize='large')
    # save vgg16 ric curve to results folder
    plt2.figure.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/VGG16_ROC.pdf")

    # VGG16 classification report
    print(classification_report(test_df.label_encoded,
                                vgg16_test_pred,
                                target_names=["Benign", "Malignant"]))

########################################################################################################

# load the data using
train_df, test_df = load_data()
train_dataset, val_dataset, test_dataset = pre_processing(train_df, test_df) # pre-process the data

# initialize model
vgg16_model = vgg16_model_sequential()
print(vgg16_model.summary()) # print model summary

# train and predict using VGG16 model
vgg16_training, vgg16_test_prob, vgg16_test_pred = train_model(vgg16_model, train_dataset, val_dataset,  test_dataset)

# evaluate the model
evaluate_model(test_df, vgg16_test_prob, vgg16_test_pred)