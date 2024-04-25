"""
DS 8013 - Deep Learning - Final Project
Ruchi Parmar - 501034872

This code builds and train Transfer Learning-enhanced version of EfficientNet_v2 model from TensorFlow Hub, check validation and test accuracy and other performance parameters
"""
# Import required packages
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scikitplot.metrics import plot_roc

import tensorflow as tf
import tensorflow_hub as hub
from keras import layers, Sequential

from load_data import load_data
from load_image import load_image
from pre_processing import pre_processing, pre_process_df

# Define Efficient V2 model
def efficientnet_v2():
    """Initialize and compile the efficientnet_v2_b0 model

    Returns:
        model: compiled efficientneet_v2_b0 model
    """
    model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2"
    efficientnet_v2_b0 = hub.KerasLayer(model_url, trainable = False, name = 'efficientnet_v2_b0')
    
    # Transfer the knowledge from model to our model
    efficientnet_v2_sequential = Sequential([
        layers.Input(shape = (224, 224, 3), dtype = tf.float32, name = 'input_image'), # input layer
        efficientnet_v2_b0, # EfficientNetV2_B0 model
        layers.Dropout(0.2), # dropour layer
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)), # dense layer 1
        layers.Dense(2, dtype=tf.float32, activation='sigmoid') # dense layer 2
        ], name='efficientnet_v2_sequential_model')

    tf.random.set_seed(42) # set random seed

    # compile the model
    efficientnet_v2_sequential.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                                  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
                                  metrics = ["accuracy"])

    return efficientnet_v2_sequential # return the model

def train_model(efficientnet_v2_model, train_dataset, val_dataset, test_dataset):
    """train the model with training data and predicit for testing data

    Args:
        efficientnet_v2_model (model): compiled efficientnet_v2_b0 model
        train_dataset (tensor data): tensor training data
        val_dataset (tensor data): tensor validation data
        test_dataset (tensor data): tensor test data
    """
    # Define early stopping mechanism with 5 step patience with val_loss
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

    # train model with training data for 50 epochs and early stopping mechanism
    efficientnet_v2_training = efficientnet_v2_model.fit(train_dataset, epochs = 50, validation_data = val_dataset, validation_steps = int(len(val_dataset)),
                                                callbacks = early_stopping)

    # Evaluate the model on test dataframe
    efficientnet_v2_evaluation = efficientnet_v2_model.evaluate(test_dataset)

    # Generate model probabilities and associated predictions
    efficientnet_v2_test_prob = efficientnet_v2_model.predict(test_dataset, verbose=1)
    efficientnet_v2_test_pred = tf.argmax(efficientnet_v2_test_prob, axis=1)

    return efficientnet_v2_training, efficientnet_v2_test_prob, efficientnet_v2_test_pred

def evaluate_model(test_df, efficientnet_v2_test_prob, efficientnet_v2_test_pred):
    """evaluate the model with test labels and predictions
    """

    # generate & plot confusion matrics to check False positives & False negatuves
    efficientnet_confusion_matrix = confusion_matrix(test_df.label_encoded, efficientnet_v2_test_pred)

    # Set plot size
    plt3 = plt.figure(figsize = (4, 4))

    # Create confusion matrix heatmap
    disp = sns.heatmap(efficientnet_confusion_matrix, annot = True, cmap = 'Blues',
                       annot_kws={"size": 6}, fmt='g',
                       linewidths=1, linecolor='black', clip_on=False,
                       xticklabels = ['benign', 'malignant'], yticklabels = ['benign', 'malignant'])

    # Set title and axis labels
    disp.set_title('EfficientNet - Confusion Matrix', fontsize = 14)
    disp.set_xlabel('Predicted Label', fontsize = 10)
    disp.set_ylabel('True Label', fontsize = 10)
    plt.yticks(rotation=0)

    # save the efficientnet_v2_B0 confusion matrix to results folder
    plt3.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/EfficientNet_v2_CM.pdf")

    # plot roc curve
    plt4 = plot_roc(test_df.label_encoded, efficientnet_v2_test_prob, figsize=(6, 6), title_fontsize='large')
    # save the efficientnet_v2_b0 roc to results folder
    plt4.figure.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/EfficientNet_v2_ROC.pdf")

    # efficientnet_v2_b0 classification report
    print(classification_report(test_df.label_encoded,
                                efficientnet_v2_test_pred,
                                target_names=["Benign", "Malignant"]))

########################################################################################################

# load the data using
train_df, test_df = load_data()
train_dataset, val_dataset, test_dataset = pre_processing(train_df, test_df) # pre-process the data

# initialize model
# Load the EfficientNetV2-B0 model
efficientnet_v2_model = efficientnet_v2()
print(efficientnet_v2_model.summary()) # print model summary

# train and predict with efficientnet_v2_B0 model test data
efficientnet_v2_training, efficientnet_v2_test_prob, efficientnet_v2_test_pred = train_model(efficientnet_v2_model, train_dataset, val_dataset, test_dataset)

# evaluate the model
evaluate_model(test_df, efficientnet_v2_test_prob, efficientnet_v2_test_pred)