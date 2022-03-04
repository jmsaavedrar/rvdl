"""
This is a simple MLP for designed for sketch classification
"""

import tensorflow as tf


class SketchMLP(tf.keras.Model):
    def __init__(self, number_of_classes):
        super(SketchMLP, self).__init__()
        #fc 1
        self.fc1 = tf.keras.layers.Dense(128, kernel_initializer='he_normal')
        self.relu = tf.keras.layers.ReLU();        
        self.bn_1 = tf.keras.layers.BatchNormalization()
        #fc 2
        self.fc2 = tf.keras.layers.Dense(64, kernel_initializer='he_normal')
        self.bn_2 = tf.keras.layers.BatchNormalization()
        #classifier
        self.classifier = tf.keras.layers.Dense(number_of_classes, kernel_initializer='he_normal')
    
    
    def call(self, inputs):   
        x = self.fc1(inputs)
        x = self.relu(self.bn_1(x))
        x = self.fc2(x)
        x = self.relu(self.bn_2(x))
        logits = tf.keras.activations.softmax(self.classifier(x))        
        return logits
        
        