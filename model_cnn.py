"""
DS 8013 - Deep Learning - Final Project
Ruchi Parmar - 501034872

This code builds and train Vanila CNN model, check validation and test accuracy and other performance parameters.
It also implements XAI methods on CNN mdoel using sampled data.
"""

# Import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scikitplot.metrics import plot_roc

import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.models import Model

from lime import lime_image
from tf_keras_vis.gradcam import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

from load_data import load_data
from load_image import load_image
from pre_processing import pre_processing, pre_process_df

def vanilla_cnn():
    """Function to define Vanilla cnn model

    Returns:
        model: compiled cnn model
    """
    # define layers
    input_img = Input(shape=(224, 224, 3)) # input image tensors
    x = Conv2D(16, kernel_size=3, activation='relu')(input_img) # first convolutional layer
    x = MaxPooling2D(pool_size=2, padding='valid')(x) # maxpooling
    x = Conv2D(8, kernel_size=3, activation='relu', name="conv2d_1")(x) # last convolutional layer
    x = MaxPooling2D(pool_size=2)(x) # maxpooling
    x = Flatten()(x) # flattern layer
    x = Dropout(0.2)(x) # Dropuout layer
    x = Dense(128, activation='relu')(x) # Dense layer 1
    output = Dense(2, activation='sigmoid')(x) # Dense layer 2
    
    # initialize the model
    cnn_sequential = Model(inputs = input_img, outputs = output, name = 'vanila_cnn_model')

    tf.random.set_seed(42) # set random seed

    #compile and return the model
    cnn_sequential.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                           optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
                           metrics = ["accuracy"] )

    return cnn_sequential


def train_model(cnn_model, train_dataset, val_dataset, test_dataset):
    """Function to train the mdoel

    Args:
        cnn_model (model): compiled CNN model
        train_dataset (tensor data): train data tensors
        val_dataset (tensor data): validation data tensors
        test_dataset (tensor data): test data tensors
    """
    
    # Define early stopping till 5 validation loss patience
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
    
    # train model on training data for 50 epochs with earlystopping mechanism
    cnn_training = cnn_model.fit(train_dataset, epochs = 50, validation_data = val_dataset, validation_steps = int(len(val_dataset)), 
                                 callbacks = early_stopping)

    # Evaluate the model on test dataframe
    cnn_evaluation = cnn_model.evaluate(test_dataset)

    # Generate model probabilities and associated predictions for test data
    cnn_test_prob = cnn_model.predict(test_dataset, verbose = 1)
    cnn_test_pred = tf.argmax(cnn_test_prob, axis = 1)

    return cnn_training, cnn_test_prob, cnn_test_pred

def evaluate_model(test_df, cnn_test_prob, cnn_test_pred):
    """ Function to evaluate the model
    """
    # generate & plot confusion matrics to check False positives & False negatuves
    cnn_confusion_matrix = confusion_matrix(test_df.label_encoded, cnn_test_pred)

    # set plot size
    plt1 = plt.figure(figsize = (4, 4))

    # create confusion matrix heatmap
    disp = sns.heatmap(cnn_confusion_matrix, annot = True, cmap = 'Blues',
                       annot_kws={"size": 6}, fmt='g',
                       linewidths=1, linecolor='black', clip_on=False,
                       xticklabels = ['benign', 'malignant'], yticklabels = ['benign', 'malignant'])

    # set title and axis labels
    disp.set_title('CNN - Confusion Matrix', fontsize = 14)
    disp.set_xlabel('Predicted Label', fontsize = 10)
    disp.set_ylabel('True Label', fontsize = 10)
    plt.yticks(rotation=0)

    # save confusion matrix plot to the results folder
    plt1.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/CNN_CM.pdf")

    # plot roc curve
    # plot roc curve
    plt2 = plot_roc(test_df.label_encoded, cnn_test_prob, figsize=(6, 6), title_fontsize='large')
    # save roc plot to the results folder
    plt2.figure.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/CNN_ROC.pdf")

    # CNN classification report
    print(classification_report(test_df.label_encoded,
                                cnn_test_pred,
                                target_names=["Benign", "Malignant"]))

########################################################################################################
    
# XAI implementation
    
# Grad-CAM implementation

def normalize(array):
    """Function to normalize the array

    Args:
        array (array): cam array to normalize

    Returns:
        array: normalised array
    """
    min_val = np.min(array) # get min value
    max_val = np.max(array) # get max value
    return (array - min_val) / (max_val - min_val) if max_val > min_val else array

def make_gradcam_heatmap(img_array, model, layer_name, class_idx):
    """using tf-keras-vis to apply Grad-CAM

    Args:
        img_array (array): image array
        model (model): cnn model
        layer_name (string): name of the last convolutional layer
        class_idx (int): index of predicted class

    Returns:
        array: gradcam
    """
    # Create a GradCAM++ object
    gradcam = GradcamPlusPlus(model,
                              model_modifier=ReplaceToLinear(),
                              clone=True)

    # Score for a specific class index
    score = CategoricalScore(class_idx)

    # Generate heatmap with GradCAM++
    cam = gradcam(score, img_array, penultimate_layer=layer_name)
    cam = normalize(cam)

    return cam

def cnn_grad_cam(cnn_model, sampled_dataset):
    """Function to get Grad-cam for sample data

    Args:
        cnn_model (model): cnn_model
        sampled_dataset (DataFrame): sampled dataset
    """

    # Setup the figure and axes for a 5 rows x 4 columns grid
    fig1, axs = plt.subplots(5, 4, figsize=(10, 10))  # Adjusted size for better display

    # Extract images and labels for plotting
    for (images, labels), i in zip(sampled_dataset.unbatch().take(10), range(10)):

        row = i // 2  # Each row contains pairs of image and heatmap
        col = (i % 2) * 2  # Calculate column index, alternating between starting points 0 and 2

        img_array = tf.expand_dims(images, axis = 0)  # Add batch dimension
        true_label = np.argmax(labels) # get true label

        # Prediction and heatmap generation
        preds = cnn_model.predict(img_array, verbose = 0) # predict for image
        pred_index = np.argmax(preds[0]) # get predicted class
        heatmap = make_gradcam_heatmap(img_array, cnn_model, 'conv2d_1', pred_index) # generate gradcam heatmap

        # plot original image
        axs[row, col].imshow(images.numpy())
        axs[row, col].set_title(f'Original: {true_label} - Predicted: {pred_index}')
        axs[row, col].axis('off')

        # plot heatmap
        axs[row, col + 1].imshow(heatmap[0], cmap = 'viridis')
        axs[row, col + 1].set_title('Heatmap')
        axs[row, col + 1].axis('off')

    # save the grad-cams to results folder
    fig1.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/CNN_GRAD.pdf")

########################################################################################################

# LIME implementation
    
def cnn_lime(explainer, predict_fn, cnn_model, sampled_dataset):
    """Function to perform LIME on sapled data

    Args:
        explainer : LIME image explainer
        predict_fn (function): fucniton to predict for data
        cnn_model (model): CNN model
        sampled_dataset (dataframe): sampeled dataframe
    """

    # Prepare figure for plotting
    fig2, axs = plt.subplots(5, 4, figsize=(10, 10))  # Adjusted size for better display

    # Process each image in the sampled dataset
    for i, (images, labels) in enumerate(sampled_dataset.unbatch().take(10)):

        row = i // 2  # Each row contains pairs of image and heatmap
        col = (i % 2) * 2  # Calculate column index, alternating between starting points 0 and 2
        
        img = images.numpy()  # Convert TensorFlow tensor to numpy array
        img_batch = tf.expand_dims(images, axis = 0).numpy()  # Add batch dimension for model prediction

        # Generate LIME explanation
        explanation = explainer.explain_instance(img, predict_fn, top_labels = 5, hide_color = 0, num_samples = 1000)

        # Get image and mask for the top prediction
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(top_label, positive_only = False, num_features = 5, hide_rest = False)

        # plot original image
        axs[row, col].imshow(images.numpy())
        true_label = np.argmax(labels.numpy())
        pred = cnn_model.predict(img_batch)
        pred_index = np.argmax(pred[0])
        axs[row, col].title.set_text(f'Original: {true_label} - Predicted: {pred_index}')
        axs[row, col].axis('off')

        # plot LIME output
        axs[row, col+1].imshow(mask,  cmap='RdBu', alpha=0.5)
        axs[row, col+1].axis('off')

        # Add colorbar to explain the heatmap intensity
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axs[row, col+1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        mappable = plt.cm.ScalarMappable(cmap='RdBu', norm=plt.Normalize(vmin=-1, vmax=1))
        cbar = plt.colorbar(mappable, cax=cax, orientation='vertical')

    # save CNN_lime output to results folder
    fig2.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/CNN_LIME.pdf")

########################################################################################################

# load the data using
train_df, test_df = load_data()
train_dataset, val_dataset, test_dataset = pre_processing(train_df, test_df) # pre-process the data

# initialize cnn model
cnn_model = vanilla_cnn()
print(cnn_model.summary()) # print cnn model archtecture

# train the model and predict for the test data
cnn_training, cnn_test_prob, cnn_test_pred = train_model(cnn_model, train_dataset, val_dataset, test_dataset)

# evaluate the model
evaluate_model(test_df, cnn_test_prob, cnn_test_pred)

# Sample 5 from each class to apply XAI
class0 = test_df[test_df['label_encoded'] == 0].sample(5, random_state = 10)
class1 = test_df[test_df['label_encoded'] == 1].sample(5, random_state = 10)
sampled_df = pd.concat([class0, class1]) # generate dataframe
sampled_dataset = pre_process_df(sampled_df, augmentation = False) # pre-process sampled data

# apply grad_CAM on sampled data
cnn_grad_cam(cnn_model, sampled_dataset)

# apply LIME on sampled data
# Initialize the LIME Image Explainer
explainer = lime_image.LimeImageExplainer(verbose = False)

# Function to predict model on batches for LIME
def predict_fn(images):
    return cnn_model.predict(images)

cnn_lime(explainer, predict_fn, cnn_model, sampled_dataset)