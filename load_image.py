"""
DS 8013 - Deep Learning - Final Project
Ruchi Parmar - 501034872

This file contains function to laod image using path
"""
# import required packages
import tensorflow as tf

def load_image(image_path):
    """define function to view images
    This function will be used further also while preprocessing and training

    Args:
        image_path (string): path to the image location

    Returns:
        image: image
    """
    # read and decode an image file to a uint8 tensor using tensorflow
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels = 3) # set 3 chanels as images are in color
 
    # resize image in case not in correct size
    image = tf.image.resize(image, [224, 224],
                            method = tf.image.ResizeMethod.LANCZOS3)

    # convert image dtype to float32 and normalize them
    image = tf.cast(image, tf.float32)/255.

    # return image
    return image