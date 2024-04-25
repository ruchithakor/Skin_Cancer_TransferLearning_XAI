"""
DS 8013 - Deep Learning - Final Project
Ruchi Parmar - 501034872

This file sets the path for folders containg traina nd test datasets(images) and return dataframes with path to image and labels.
"""

# Import required packages
import glob
import pandas as pd

def load_data():
    """Load the datapath, and labels to dataframes for ease of use
    """
    # define the path to dataset - data is stored in folder named data/train and data/test
    # paths are set as per my local system, need to be changed according to environment
    path_train = '/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/data/train/'
    path_test = '/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/data/test/'

    # use glob to get path for images withtin folders
    train_images = glob.glob(f"{path_train}**/*.jpg")
    test_images = glob.glob(f"{path_test}**/*.jpg")

    # get the size of training and testing data(images)
    train_samples = len(train_images)
    test_samples = len(test_images)

    # Print number of train and test samples in dataset
    print(f"train samples count: {train_samples}")
    print(f"test samples count: {test_samples}")

    # Create dataframe to store path to each image along with label

    # generate labels from the image path, as data is stored in seperate folders as per their labels - malignant and benign
    trian_labels = [_.split('/')[-2:][0] for _ in train_images]
    test_labels = [_.split('/')[-2:][0] for _ in test_images]

    # Create train dataframe
    train_df = pd.DataFrame({
        'image_path': train_images,
        'label': trian_labels
    })

    # Create test dataframe
    test_df = pd.DataFrame({
        'image_path': test_images,
        'label': test_labels
    })

    # Generate encoded labels malignant - 1, benign - 0
    train_df['label_encoded'] = train_df.apply(lambda row: 1 if row.label == 'malignant' else 0, axis=1)
    test_df['label_encoded'] = test_df.apply(lambda row: 1 if row.label == 'malignant' else 0, axis=1)

    return train_df, test_df
