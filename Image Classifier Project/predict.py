import json
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import warnings 
warnings.filterwarnings('ignore')
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
import json
import glob 
from PIL import Image

from tensorflow.keras import layers

import argparse

parser = argparse.ArgumentParser(description='Image Classifier - Prediction Part')
parser.add_argument('--input', default='./test_images/hard-leaved_pocket_orchid.jpg', action="store", type = str, help='image path')
parser.add_argument('--model', default='./flowers_model.h5', action="store", type = str, help='checkpoint file path/name')
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int, help='return top K most likely classes')
parser.add_argument('--category_names', dest="category_names", action="store", default='label_map.json', help='mapping the categories to real names')


arg_parser = parser.parse_args()

image_path = arg_parser.input
model_path = arg_parser.model
topk = arg_parser.top_k
category_names = arg_parser.category_names
 
def process_image(image):
    # Convert the image to a TensorFlow tensor
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Resize the image
    image_resized = tf.image.resize(image_tensor, (224, 224))
    
    # Normalize the pixel values
    image_normalized = image_resized / 255.0
    
    # Convert the image back to a NumPy array
    processed_image = image_normalized.numpy()
    
    return processed_image

def predict(image_path, model, top_k=5):
    # Load the image using PIL
    image = Image.open(image_path)
    
    # Convert the image to a NumPy array
    image_array = np.asarray(image)
    
    # Preprocess the image
    processed_image = process_image(image_array)
    
    # Add an extra dimension to represent the batch size
    input_image = np.expand_dims(processed_image, axis=0)
    
    # Make predictions
    predictions = model.predict(input_image)
    
    # Get the top k predicted class labels and probabilities
    top_k_indices = np.argsort(predictions[0])[::-1][:top_k]
    top_k_probs = predictions[0][top_k_indices]
    top_k_classes = [str(label) for label in top_k_indices]
    
    return top_k_probs, top_k_classes

if __name__== "__main__":

    print ("start Prediction ...")
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    
    reloaded_model = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})
    probs, classes = predict(image_path, reloaded_model, topk)
    label_names = [class_names[str(int(idd)+1)] for idd in classes]
    print(probs)
    print(classes)
    print(label_names)
print ("End Prediction ")
