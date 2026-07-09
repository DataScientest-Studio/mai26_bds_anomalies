import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, RandomRotation, RandomZoom, RandomContrast, RandomBrightness, RandomTranslation

class DataAugmentation(Sequential):
    def __init__(self):
        
        # Transformations des images
        s=[
            RandomBrightness(0.001, value_range=[0.0, 1.0]), 
            RandomContrast(0.001, value_range=[0.0, 1.0]), 
            RandomRotation(0.02, fill_mode='nearest'), 
            RandomTranslation(0.05, 0.05, fill_mode='nearest'), 
            RandomZoom((0,0.05)), 
        ]

        super().__init__(s)

    def normalize_augment(self, images):
        images = tf.cast(images, tf.float32) / 255.0
        return self(images)

    def normalize(self, images):
        images = tf.cast(images, tf.float32) / 255.0
        return images
    
    def augment(self, images):
        return self(images)