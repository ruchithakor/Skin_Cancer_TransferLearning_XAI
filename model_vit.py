"""
DS 8013 - Deep Learning - Final Project
Ruchi Parmar - 501034872

This code builds and train Transfer Learning-enhanced version of Vision Transformer(ViT) model from vit-keras, check validation and test accuracy and other performance parameters
It also implements XAI methods on ViT mdoel using sampled data.
"""
# Import required packages
import pandas as pd
import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scikitplot.metrics import plot_roc

import tensorflow as tf
from keras import layers, Sequential
from vit_keras import vit
from keras.models import Model
from skimage.transform import resize
from lime import lime_image

from load_data import load_data
from load_image import load_image
from pre_processing import pre_processing, pre_process_df

def vit_b16_model(vit_model_16):
    """Initialized and compile Vision Transformer TL model

    Args:
        vit_model_16 (model): Vision transformer pre trained model

    Returns:
        model: transfer learning version of Vision Transformer model compiled
    """
    # Freeze model layers for inference-mode only
    for layer in vit_model_16.layers:
        layer.trainable = False

    # Transger the features learnt to  enhanced model
    vit_model_sequential = Sequential([
        layers.Input(shape = (224, 224, 3), dtype = tf.float32, name = 'input_image'), # Input image
        vit_model_16, # ViT mmodel
        layers.Dropout(0.2), # Dropout
        layers.Dense(128, activation='relu'), # Dense layer 1
        layers.Dense(2, dtype=tf.float32, activation='sigmoid')], name='vit_b16_sequential_model') # Dense layer 2

    tf.random.set_seed(42) # set random seed

    # Compile the model
    vit_model_sequential.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                                  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
                                  metrics = "accuracy")

    return vit_model_sequential # return the model

def train_model(vit_model, train_dataset, val_dataset, test_dataset):
    """Train the model with training data and predict for the test data

    Args:
        vit_model (model): Vision transformer compiled model
        train_dataset (tensor data): tensor data
        val_dataset (tensor data): tensor data
        test_dataset (tensor data): tensor data
    """
    # Define early stopping till 5 validation loss patience
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
    
    # train model with training data for 50 epochs and early stopping mechanism
    vit_training = vit_model.fit(train_dataset, epochs = 50, validation_data = val_dataset, validation_steps = int(len(val_dataset)),
                                 callbacks = early_stopping)

    # Evaluate the model on test dataframe
    vit_evaluation = vit_model.evaluate(test_dataset)

    # Generate model probabilities and associated predictions
    vit_test_prob = vit_model.predict(test_dataset, verbose=1)
    vit_test_pred = tf.argmax(vit_test_prob, axis=1)

    return vit_training, vit_test_prob, vit_test_pred

def evaluate_model(test_df, vit_test_prob, vit_test_pred):
    """evaluate the model on test labels and presictions
    """
    # generate & plot confusion matrics to check False positives & False negatuves
    vit_confusion_matrix = confusion_matrix(test_df.label_encoded, vit_test_pred)

    # Set plot size
    plt1 = plt.figure(figsize = (4, 4))

    # Create confusion matrix heatmap
    disp = sns.heatmap(vit_confusion_matrix, annot = True, cmap = 'Blues',
                       annot_kws={"size": 6}, fmt='g',
                       linewidths=1, linecolor='black', clip_on=False,
                       xticklabels = ['benign', 'malignant'], yticklabels = ['benign', 'malignant'])

    # Set title and axis labels
    disp.set_title('ViT - Confusion Matrix', fontsize = 14)
    disp.set_xlabel('Predicted Label', fontsize = 10)
    disp.set_ylabel('True Label', fontsize = 10)
    plt.yticks(rotation=0)

    # save ViT confusion matrix to the results folder
    plt1.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/ViT_CM.pdf")

    # plot roc curve
    plt2 = plot_roc(test_df.label_encoded, vit_test_prob, figsize=(6, 6), title_fontsize='large')
    # save ViT ROC to the results folder
    plt2.figure.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/ViT_ROC.pdf")

    # ViT classification report 
    print(classification_report(test_df.label_encoded,
                               vit_test_pred,
                                target_names=["Benign", "Malignant"]))

########################################################################################################

# XAI implementation

# LIME implementation

def vit_lime(explainer, predict_fn, vit_model, sampled_dataset):
    """Function to apply LIME method on the sampled images
    """
    # Prepare figure for plotting
    fig1, axs = plt.subplots(5, 4, figsize=(10, 10))  # Adjusted size for better display

    # Process each image in the sampled dataset
    for ind, (images, labels) in enumerate(sampled_dataset.unbatch().take(10)):
    
        row = ind // 2  # Each row contains pairs of image and heatmap
        col = (ind % 2) * 2  # Calculate column index, alternating between starting points 0 and 2
        
        img = images.numpy()  # Convert TensorFlow tensor to numpy array
        img_batch = tf.expand_dims(images, axis = 0).numpy()  # Add batch dimension for model prediction

        # Generate LIME explanation
        explanation = explainer.explain_instance(img, predict_fn, top_labels = 5, hide_color = 0, num_samples = 1000)

        # Get image and mask for the top prediction
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(top_label, positive_only = False, num_features = 5, hide_rest = False)

        # show original image
        axs[row, col].imshow(images.numpy())
        true_label = np.argmax(labels.numpy())
        pred = vit_model.predict(img_batch)
        predicted_label = np.argmax(pred[0])
        axs[row, col].title.set_text(f'Original: {true_label} - Predicted: {predicted_label}')
        axs[row, col].axis('off')

        # show lime output
        axs[row, col+1].imshow(mask,  cmap='RdBu', alpha=0.5)
        axs[row, col+1].axis('off')

        # Add colorbar to explain the heatmap intensity
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axs[row, col + 1])
        cax = divider.append_axes('right', size = '5%', pad = 0.05)
        mappable = plt.cm.ScalarMappable(cmap = 'RdBu', norm = plt.Normalize(vmin = -1, vmax = 1))
        cbar = plt.colorbar(mappable, cax = cax, orientation = 'vertical')

    # save the LIME output on sampled data to results folder
    fig1.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/ViT_LIME.pdf")

########################################################################################################

# Attention mapping implementation

def build_attention_model(base_model):
    """fucniton to build model to get respective attension layers output

    Args:
        base_model (model): base ViT model

    Returns:
        models: attention models for 3 differnet encoder blocks
    """
    # Define the input exactly like the base model's input
    input_layer = base_model.input

    # Define the outputs: the original model's output and attention weights from a chosen transformer block
    # We select the first, middle and last block's attention weights for visualization
    final_output = base_model.output
    attention_weights_0 = base_model.get_layer("Transformer/encoderblock_0").output[1]
    attention_weights_4 = base_model.get_layer("Transformer/encoderblock_4").output[1]
    attention_weights_11 = base_model.get_layer("Transformer/encoderblock_11").output[1]

    # Create a new model that outputs both the final predictions and the attention weights
    model_with_attention_0 = Model(inputs = input_layer, outputs = [final_output, attention_weights_0])
    model_with_attention_4 = Model(inputs = input_layer, outputs = [final_output, attention_weights_4])
    model_with_attention_11 = Model(inputs = input_layer, outputs = [final_output, attention_weights_11])

    # return the models with attentions
    return model_with_attention_0, model_with_attention_4, model_with_attention_11

def vit_attentions(sampled_dataset, vit_attention_model, filename):
    """Funciton to visualize the attentions over original images

    Args:
        sampled_dataset (dataframe): sampled data
        vit_attention_model (model): model with attentions
        filename (string): name of the file to save output
    """
    for images, _ in sampled_dataset.take(1):  # Process the first batch
        input_images = images.numpy()

        # Predict and extract attention weights
        _, attention_weights = vit_attention_model.predict(input_images)

        fig2, axs = plt.subplots(2, 5, figsize=(20, 10)) # Initialize the figure for visualization

        # Loop through each image in the batch
        for ind in range(input_images.shape[0]):

            # Prepare the original image, normalize if needed
            original_image = input_images[ind]
            if original_image.min() < 0 or original_image.max() > 1:
                original_image = (original_image + 1) / 2  # Normalize to [0, 1] if needed
            original_image = np.clip(original_image, 0, 1)  # Ensure values are within [0, 1]

            # Normalize the attention map
            attention_map = np.mean(attention_weights[ind], axis = 0)  # Average across all heads
            attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))

            # Resize the attention map to match the original image size
            height, width, _ = original_image.shape
            attention_map_resized = resize(attention_map, (height, width), mode = 'reflect', anti_aliasing = True)

            # Create heatmap
            heatmap = np.uint8(255 * attention_map_resized)  # Convert to uint8
            heatmap = plt.cm.viridis(heatmap)[:, :, :3]  # Apply colormap by cutting off alpha channel

            # Overlay the heatmap onto the original image
            alpha = 0.6  # Define the transparency level
            overlayed_image = heatmap * alpha + original_image * (1 - alpha)

            row = ind // 5
            col = ind % 5
            ax = axs[row, col]
            im = ax.imshow(overlayed_image)
            ax.axis('off')

            # Create a colorbar with the correct scale
            axins = ax.inset_axes([1.05, 0, 0.05, 1], transform = ax.transAxes) 
            mappable = plt.cm.ScalarMappable(cmap = 'viridis', norm = plt.Normalize(vmin = 0, vmax = 1))
            mappable.set_array([])
            cbar = plt.colorbar(mappable, cax = axins)

        # save the attentions to results folder
        fig2.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/" + filename + ".pdf")

########################################################################################################

# load the data using
train_df, test_df = load_data()
train_dataset, val_dataset, test_dataset = pre_processing(train_df, test_df) # pre-process the data

# initialize model
# Load the Vision Transformer Model model
vit_model_16 = vit.vit_b16(image_size = 224, activation = 'sigmoid', pretrained = True, include_top = False, pretrained_top = False, classes = 2)
vit_model = vit_b16_model(vit_model_16)
print(vit_model.summary()) # print the summary of the model

# train the model and predict for test data
vit_training, vit_test_prob, vit_test_pred = train_model(vit_model, train_dataset, val_dataset, test_dataset)

# evaluate the model
evaluate_model(test_df, vit_test_prob, vit_test_pred)

# Sample 5 from each class
class0 = test_df[test_df['label_encoded'] == 0].sample(5, random_state = 10)
class1 = test_df[test_df['label_encoded'] == 1].sample(5, random_state = 10)
sampled_df = pd.concat([class0, class1]) # create sample dataframe
sampled_dataset = pre_process_df(sampled_df, augmentation=False) # pre-process the sampled data

# apply LIME on sampled data
# Initialize the LIME Image Explainer
explainer = lime_image.LimeImageExplainer(verbose = False)

# Function to predict model on batches for LIME
def predict_fn(images):
    return vit_model.predict(images)

vit_lime(explainer, predict_fn, vit_model, sampled_dataset)

# Apply Attention mapping to the sampled data
vit_attention_model_0, vit_attention_model_4, vit_attention_model_11 = build_attention_model(vit_model_16)
vit_attentions(sampled_dataset, vit_attention_model_0, "ViT_Attentions_0")
vit_attentions(sampled_dataset, vit_attention_model_4, "ViT_Attentions_5")
vit_attentions(sampled_dataset, vit_attention_model_11, "ViT_Attentions_final")