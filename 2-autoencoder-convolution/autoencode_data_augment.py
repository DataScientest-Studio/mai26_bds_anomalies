import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, RandomRotation, RandomZoom, RandomContrast, RandomBrightness, RandomTranslation

from preprocessing_screw import preprocess_screw_batch

class DataAugmentation(Sequential):
    def __init__(self, colors=True, moves=True, screw=False):
        
        self.transform=False
        s=[]

        # Transformations des images
        if colors:
            colors_sequence=[
                RandomBrightness(0.001, value_range=[0.0, 1.0]), 
                RandomContrast(0.001, value_range=[0.0, 1.0]), 
            ]
            s.extend(colors_sequence)
            self.transform=True

        if moves:
            moves_sequence=[
                RandomRotation(0.02, fill_mode='nearest'), 
                RandomTranslation(0.05, 0.05, fill_mode='nearest'), 
                RandomZoom((0,0.05)), 
            ]
            s.extend(moves_sequence)
            self.transform=True

        self.screw=screw
        super().__init__(s)

    def normalize(self, images):
        if self.screw:
            images = preprocess_screw_batch(images)

        images = tf.cast(images, tf.float32) / 255.0
        return images
    
    def augment(self, images):
        if self.screw:
            images = preprocess_screw_batch(images)
            
        if self.transform:
            images = tf.cast(images, tf.float32) / 255.0
            return self(images)
        else:
            images = tf.cast(images, tf.float32) / 255.0
            return images