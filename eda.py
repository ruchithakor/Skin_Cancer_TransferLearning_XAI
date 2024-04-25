"""
DS 8013 - Deep Learning - Final Project
Ruchi Parmar - 501034872

This file performs basic EDA on dataset such as - 
 - check class balance in both training and testing data
 - view sample training images of both classes
"""

# Import required packages
import random
import matplotlib.pyplot as plt

from load_data import load_data # import load_data() function to load data
from load_image import load_image # import function to load image

# get train and test dataframe containing path and labels
train_df, test_df = load_data()

# class balance check for train & test data

# get value count for each label - benign and malignant
train_class_count = train_df['label'].value_counts() # training data
test_class_count = test_df['label'].value_counts() # test data

print(f"train_class_count: {train_class_count}") # print counts for training
print(f"test_class_count: {test_class_count}") # print counts for testing

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 8)) # define figure to plot graph for class balance

# Set the spacing between subplots
fig1.tight_layout(pad = 6.0)

# Create a dictionary to map categories to colors
color_mapping = {'malignant': 'red', 'benign': 'blue'}

# Plot Train Labels Distribution
ax1.set_title('Train Labels Distribution', fontsize = 15)
train_labels = train_class_count.index.tolist()
train_values = train_class_count.values.tolist()
ax1.bar(train_labels, train_values, color=['red', 'blue'])

# Plot Test Labels Distribution
ax2.set_title('Test Labels Distribution', fontsize = 15)
test_labels = test_class_count.index.tolist()
test_values = test_class_count.values.tolist()
ax2.bar(test_labels, test_values, color=['red', 'blue'])

# To prevent the labels from overlapping with the plots
plt.setp(ax1.xaxis.get_majorticklabels(), rotation = 90)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation = 90)

# save the graph to results folder
fig1.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/class_balance.pdf")

# View some random sample images 

idx = random.sample(train_df.index.to_list(), 10) # get 10 ids from training dataset randomly
fig2 = plt.figure(figsize=(14,8)) # define figure to plot images

for id, _ in enumerate(idx):
    # plot images for all 10 ids
    plt.subplot(2, 5, id+1)
    plt.title(f'Label: {train_df.label[_]}')
    plt.imshow(load_image(train_df.image_path[_]))

# save the plot to results folder
fig2.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/sample_images.pdf")