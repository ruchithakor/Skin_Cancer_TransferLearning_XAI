"""
DS 8013 - Deep Learning - Final Project
Ruchi Parmar - 501034872

This code builds and train Transfer Learning-enhanced version of ResNet50 model from tenserflow, check validation and test accuracy and other performance parameters
"""

# Import required packages
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scikitplot.metrics import plot_roc

import tensorflow as tf
from keras import layers, Sequential
from keras.applications.resnet50 import ResNet50

from load_data import load_data
from load_image import load_image
from pre_processing import pre_processing, pre_process_df

def resnet50_model_sequential():
    """Function to define and compile ResNet50 model

    Returns:
        model: resnet50 model
    """
    # Load the ResNet50 Model model from keras
    resnet50 = ResNet50(input_shape = (224,224, 3), weights='imagenet', include_top = False, classes = 2)

    # Freeze the convolutional base to start with
    resnet50.trainable = False

    # transfer the features learnt from ResNet50 to our model
    resnet50_model_sequential = Sequential([
        layers.Input(shape = (224, 224, 3), dtype = tf.float32, name = 'input_image'), # input layer
        resnet50, # ResNet50
        layers.GlobalAveragePooling2D(), #Maxpooling
        layers.Dropout(0.2), # dropout
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)), # Dense layer
        layers.Dense(2, dtype=tf.float32, activation='sigmoid')], name='resnet50_sequential_model') # final dense layer

    tf.random.set_seed(42) # set random seed

    # Compile the model
    resnet50_model_sequential.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                                  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
                                  metrics = ["accuracy"])

    return resnet50_model_sequential # return the model


def train_model(resnet50_model, train_dataset, val_dataset,  test_dataset):
    """train and predict the model

    Args:
        resnet50_model (model): compiled ResNet50 model
        train_dataset (tensor data): train data tensors
        val_dataset (tensor data): validation data tensors
        test_dataset (tensor data): test data tensors
    """
    # Define early stopping mechanism wiht 5 step patience with val_loass
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

    # train model with training data for 50 epochs with earlystopping mechanism
    resnet50_training = resnet50_model.fit(train_dataset, epochs = 50, validation_data = val_dataset, validation_steps = int(len(val_dataset)),
                                 callbacks = early_stopping)

    # Evaluate the model on test dataframe
    resnet50_evaluation = resnet50_model.evaluate(test_dataset)

    # Generate model probabilities and associated predictions
    resnet50_test_prob = resnet50_model.predict(test_dataset, verbose=1)
    resnet50_test_pred = tf.argmax(resnet50_test_prob, axis=1)

    return resnet50_training, resnet50_test_prob, resnet50_test_pred

def evaluate_model(test_df, resnet50_test_prob, resnet50_test_pred):
    """Evaluate the model using test predictions and labels
    """
    # generate & plot confusion matrics to check False positives & False negatuves
    resnet50_confusion_matrix = confusion_matrix(test_df.label_encoded, resnet50_test_pred)

    # Set plot size
    plt1 = plt.figure(figsize = (4, 4))

    # Create confusion matrix heatmap
    disp = sns.heatmap(resnet50_confusion_matrix, annot = True, cmap = 'Blues',
                       annot_kws={"size": 6}, fmt='g',
                       linewidths=1, linecolor='black', clip_on=False,
                       xticklabels = ['benign', 'malignant'], yticklabels = ['benign', 'malignant'])

    # Set title and axis labels
    disp.set_title('ResNet50 - Confusion Matrix', fontsize = 14)
    disp.set_xlabel('Predicted Label', fontsize = 10)
    disp.set_ylabel('True Label', fontsize = 10)
    plt.yticks(rotation=0)

    # save the resnet50 confusion matrix to results folder
    plt1.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/ResNet50_CM.pdf")

    # plot roc curve 
    plt2 = plot_roc(test_df.label_encoded, resnet50_test_prob, figsize=(6, 6), title_fontsize='large')
    # save the roc_curve to results folder
    plt2.figure.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/ResNet50_ROC.pdf")

    # classsificaiton report for ResNet50 model
    print(classification_report(test_df.label_encoded,
                                resnet50_test_pred,
                                target_names=["Benign", "Malignant"]))

########################################################################################################

# load the data using
train_df, test_df = load_data()
train_dataset, val_dataset, test_dataset = pre_processing(train_df, test_df) # pre-process the data

# initialize model
resnet50_model = resnet50_model_sequential()
print(resnet50_model.summary())

# train the model on training data and predict for test data
resnet50_training, resnet50_test_prob, resnet50_test_pred = train_model(resnet50_model, train_dataset, val_dataset,  test_dataset)

# evaluate the model
evaluate_model(test_df, resnet50_test_prob, resnet50_test_pred)