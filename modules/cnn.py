import imp
import tensorflow as tf 
import numpy as np

class OneDCNN(tf.keras.models.Model):

    def __init__(self, num_cnn_blocks=3, max_filter_pow=5, num_classes=4):
        super().__init__()
        self.ls = []
        for i in range(num_cnn_blocks):
            self.ls.append(tf.keras.layers.Conv1D(filters=2**(max_filter_pow-i), kernel_size=3, activation=tf.keras.activations.tanh, name='1dconv_layer_' + str(i)))
            self.ls.append(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, name='maxpool_layer_' + str(i)))
        self.ls.append(tf.keras.layers.Flatten(name='flatten_layer'))
        self.ls.append(tf.keras.layers.Dense(num_classes, activation=tf.keras.activations.softmax, name='dense_softmax'))

    def call(self, x):
        x = x[:, :, np.newaxis]
        for layer in self.ls:
            x = layer(x)
        return x
