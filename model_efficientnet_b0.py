"""
DS 8013 - Deep Learning - Final Project
Ruchi Parmar - 501034872

This code builds and train Transfer Learning-enhanced version of EfficientNet model from TensorFlow Hub, check validation and test accuracy and other performance parameters
"""
# Import required packages

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scikitplot.metrics import plot_roc

import tensorflow as tf
from keras import layers
from keras.applications import EfficientNetB0

from load_data import load_data
from load_image import load_image
from pre_processing import pre_processing, pre_process_df

def efficientnet_sequential():
    """Define and compile Efficientnet_B0 model

    Returns:
        model: compile efficientnet_B0 model
    """
    inputs = layers.Input(shape = (224, 224, 3), dtype = tf.float32, name = 'input_image') # input layer

    # Load the base EfficientNetB0 model from keraas
    outputs = EfficientNetB0(include_top = True, weights = None, classes = 2)(inputs)

    efficientnet_sequential = tf.keras.Model(inputs, outputs) # initialize the model

    tf.random.set_seed(42) # set the random seed

    # Compile the model
    efficientnet_sequential.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                                  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
                                  metrics = ["accuracy"])

    return efficientnet_sequential # return the model

def train_model(efficientnet_model, train_dataset, val_dataset, test_dataset):
    """train the model with training data and predicit for testing data

    Args:
        efficientnet_model (model): compiled efficientnet model
        train_dataset (tensor data): tensor training data
        val_dataset (tensor data): tensor validation data
        test_dataset (tensor data): tensor testing data
    """
    # Define early stopping mechanism with 5 step patience with val_loss
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

    # train model with training data for 50 epochs and early stop mechanism
    efficientnet_training = efficientnet_model.fit(train_dataset, epochs = 50, validation_data = val_dataset, validation_steps = int(len(val_dataset)),
                                                callbacks = early_stopping)

    # Evaluate the model on test dataframe
    efficientnet_evaluation = efficientnet_model.evaluate(test_dataset)

    # Generate model probabilities and associated predictions
    efficientnet_test_prob = efficientnet_model.predict(test_dataset, verbose=1)
    efficientnet_test_pred = tf.argmax(efficientnet_test_prob, axis=1)

    return efficientnet_training, efficientnet_test_prob, efficientnet_test_pred

def evaluate_model(test_df, efficientnet_test_prob, efficientnet_test_pred):
    """evaluate the model with test labels and predictions
    """
    # generate & plot confusion matrics to check False positives & False negatuves
    efficientnet_confusion_matrix = confusion_matrix(test_df.label_encoded, efficientnet_test_pred)

    # Set plot size
    plt1 = plt.figure(figsize = (4, 4))

    # Create confusion matrix heatmap
    disp = sns.heatmap(efficientnet_confusion_matrix, annot = True, cmap = 'Blues',
                       annot_kws={"size": 6}, fmt = 'g',
                       linewidths = 1, linecolor = 'black', clip_on = False,
                       xticklabels = ['benign', 'malignant'], yticklabels = ['benign', 'malignant'])

    # Set title and axis labels
    disp.set_title('EfficientNet - Confusion Matrix', fontsize = 14)
    disp.set_xlabel('Predicted Label', fontsize = 10)
    disp.set_ylabel('True Label', fontsize = 10)
    plt.yticks(rotation = 0)

    # save efficientnet_B0 confusion matrix to the results folder
    plt1.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/EfficientNet_CM.pdf")

    # plot roc curve
    plt2 = plot_roc(test_df.label_encoded, efficientnet_test_prob, figsize = (6, 6), title_fontsize='large')
    # save efficientnet_B0 roc curve to results folder
    plt2.figure.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/EfficientNet_ROC.pdf")

    # EfficientNet_B0 classification report
    print(classification_report(test_df.label_encoded,
                            efficientnet_test_pred,
                            target_names = ["Benign", "Malignant"]))


########################################################################################################

# load the data using
train_df, test_df = load_data()
train_dataset, val_dataset, test_dataset = pre_processing(train_df, test_df) # pre-process the data

# initialize model 
efficientnet_model = efficientnet_sequential()
print(efficientnet_model.summary()) # print the summary of model

# train and predict using efficientnet_B0 model
efficientnet_training, efficientnet_test_prob, efficientnet_test_pred = train_model(efficientnet_model, train_dataset, val_dataset, test_dataset)

# evaluate the model
evaluate_model(test_df, efficientnet_test_prob, efficientnet_test_pred)