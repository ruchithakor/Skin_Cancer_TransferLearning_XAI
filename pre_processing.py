"""
DS 8013 - Deep Learning - Final Project
Ruchi Parmar - 501034872

This file performs pre-processing on the data - 
- split training into train-validation split
- add custom augmengtation layer(can be replaced with ImageDataGenerator) - here data is preety much balanced already so no need 
to do higher augmentation, only randomflip & randomzoom is performed.
- prepare tensor dataset and batches using tensorfloe.data.Dataset to fit in the training data
"""

# Import required packages
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, Sequential

from load_image import load_image # import function to load image

def pre_processing(train_df, test_df):
    """This function performs few pre-processing steps on data

    Args:
        train_df (DataFrame): Dataframe with path to train images and labels
        test_df (DataFrame): Dataframe with path to test images and labels

    Returns:
        dataframes: train, validation and test data tensors 
    """

    # split the training data into train-validation set with 80-20 ratio
    train_idx, val_ind, _, _ = train_test_split(train_df.index, train_df.label_encoded, test_size = 0.20,
                                            stratify = train_df.label_encoded, random_state = 42)

    # get training and validation data
    train_new_df = train_df.iloc[train_idx].reset_index(drop = True)
    val_df = train_df.iloc[val_ind].reset_index(drop = True)

    # view shapes for train and validation data after split
    print(f"training size: {train_new_df.shape}, validation size: {val_df.shape}")

    # Check class balance in both trainng and validation set
    train_class_count = train_new_df['label'].value_counts()
    val_class_count = val_df['label'].value_counts()

    print(f"train_class_count: {train_class_count}")
    print(f"test_class_count: {val_class_count}")

    # Build custome augmentation layer
    augmentation_layer = Sequential([
        layers.RandomFlip(mode='horizontal_and_vertical', seed = 42),
        layers.RandomZoom(height_factor = (-0.1, 0.1), width_factor = (-0.1, 0.1), seed = 42),
    ], name='augmentation_layer')

    # pre-process training data

    # for training data augmentation layer will be added
    augmentation = True

    # Get image paths and labels from DataFrame
    image_paths = train_new_df.image_path
    image_labels = tf.one_hot(train_new_df.label_encoded, depth = 2).numpy()

    # create tensor dataset from original data
    train_dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))

    # apply augmentation layer to the data
    if augmentation:
          train_dataset = train_dataset.map(lambda x, y: (augmentation_layer(load_image(x)), y), num_parallel_calls = tf.data.AUTOTUNE)
    else:
          train_dataset = train_dataset.map(lambda x, y: (load_image(x), y), num_parallel_calls = tf.data.AUTOTUNE)

    # apply batching
    train_dataset = train_dataset.batch(32)

    # pre-process validation data

    # for validation and testing data ugmentation layer will not be added
    augmentation = False

    # Get image paths and labels from DataFrame
    image_paths = val_df.image_path
    image_labels = tf.one_hot(val_df.label_encoded, depth = 2).numpy()

    # create tensor dataset from original data
    val_dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))

    # apply augmentation layer to the data
    if augmentation:
          val_dataset = val_dataset.map(lambda x, y: (augmentation_layer(load_image(x)), y), num_parallel_calls = tf.data.AUTOTUNE)
    else:
          val_dataset = val_dataset.map(lambda x, y: (load_image(x), y), num_parallel_calls = tf.data.AUTOTUNE)

    # apply batching
    val_dataset = val_dataset.batch(32)

    # pre-process test data

    # for validation and testing data ugmentation layer will not be added
    augmentation = False

    # Get image paths and labels from DataFrame
    image_paths = test_df.image_path
    image_labels = tf.one_hot(test_df.label_encoded, depth = 2).numpy()

    # create tensor dataset from original data
    test_dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))

    # apply augmentation layer to the data
    if augmentation:
          test_dataset = test_dataset.map(lambda x, y: (augmentation_layer(load_image(x)), y), num_parallel_calls = tf.data.AUTOTUNE)
    else:
          test_dataset = test_dataset.map(lambda x, y: (load_image(x), y), num_parallel_calls = tf.data.AUTOTUNE)

    # apply batching
    test_dataset = test_dataset.batch(32)

    return train_dataset, val_dataset, test_dataset


def pre_process_df(df, augmentation = False):
    """This function performs preprocessing on single dataframe in needed

      Args:
          df (DataFrame): dataframe to be process
          augmentation (bool, optional): if augmentation layer needed or not. Defaults to False.

      Returns:
          dataframe: datafreame with processed tensors
    """
    # Get image paths and labels from DataFrame
    image_paths = df.image_path
    image_labels = tf.one_hot(df.label_encoded, depth = 2).numpy()

    # Build custome augmentation layer
    augmentation_layer = Sequential([
        layers.RandomFlip(mode='horizontal_and_vertical', seed = 42),
        layers.RandomZoom(height_factor = (-0.1, 0.1), width_factor = (-0.1, 0.1), seed = 42),
    ], name='augmentation_layer')

    # create tensor dataset from original data
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))

    # apply augmentation layer to the data
    if augmentation:
          dataset = dataset.map(lambda x, y: (augmentation_layer(load_image(x)), y), num_parallel_calls = tf.data.AUTOTUNE)
    else:
          dataset = dataset.map(lambda x, y: (load_image(x), y), num_parallel_calls = tf.data.AUTOTUNE)

    # apply batching
    dataset = dataset.batch(32)

    return dataset