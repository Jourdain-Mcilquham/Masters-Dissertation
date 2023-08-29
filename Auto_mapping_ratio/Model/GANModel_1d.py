# -*- coding: utf-8 -*-

"""
FileName:               GANModel_1d
Author Name:            Arun M Saranathan
Description:            This file includes implementation of different models for the generator and discriminator that
                        we use in our model. We use keras models

Date Created:           05th December 2017
Last Modified:          03rd September 2019
"""

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set the logging level to ignore all messages except errors
import tensorflow as tf
# tf.get_logger().setLevel('ERROR')  # Set the logging level of TensorFlow logger to ignore all messages except errors
from keras.models import Model, Sequential
from keras.layers import *
# from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import Constant
from keras.optimizers import Adam

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Specify device placement to force TensorFlow to use the first GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')

class GANModel_1d():
    """
    --------------------------------------------------------------------------------------------------------------------
    FUNCTION NAME INTERPRETATION
    --------------------------------------------------------------------------------------------------------------------
    xxxModel -> the first three characters describe the kind of model 'gen' for
    generators and 'dis' for discriminators
    _XX -> the next two charcaters denote the type of connection 'FC' for fully
    connected and 'CV' for convolutional
    _Lysy -> L denotes layers and s the stride size, therefore 'L6s2' denotes 6
    layers with stride.'L2s2_L6s1' denotes 2 layers with stride 2 followed by 4
    layers with stride 1

     All padding is 'same'
     dropout = 0.4
     Batch normalization is applied
     Activation is 'relu'
    --------------------------------------------------------------------------------------------------------------------
    """

    'The Constructor'
    def __init__(self, img_rows=240, dropout=0.4, genFilters=250,
                 disFilters=20, filterSize=11, input_dim=50):
        self.img_rows = img_rows
        self.dropout = dropout
        self.genFilters = genFilters
        self.disFilters = disFilters
        self.filterSize = filterSize
        self.input_dim = input_dim

    # enable gpu


    def disModel_CV_L6s2(self):
        """
        # DISCRIMINATOR-1
        # 5 Layers
        # Downsampling Factor (Stride) 2 per layer (except last)
        # Output Size  = 1 X 1
        # Activation: 'relu'
        # bias_initializer = Constant value of 0.1

        :return: Returns a Keras model with 5 layers, 4 Convolutional and 1 FC
        """

        discriminator = Sequential()
        # 'LAYER -1'
        # 'In: 240 X 1 X 1, depth =1'
        # 'Out: 120 X 1 X 1, depth = 25'
        discriminator.add(Conv1D(filters=self.disFilters, kernel_size=self.filterSize, strides=2,
                                 input_shape=(self.img_rows, 1), bias_initializer=Constant(0.1),
                                 activation='relu', padding='same'))
        #discriminator.add(BatchNormalization(momentum=0.9))
        discriminator.add(Dropout(self.dropout))

        # 'LAYER -2'
        # 'In: 120 X 1 X 1, depth =25'
        # 'Out: 60 X 1 X 1, depth = 50'
        discriminator.add(Conv1D(filters=self.disFilters * 2,
                                 kernel_size=self.filterSize, strides=2,
                                 bias_initializer=Constant(0.1), activation='relu',
                                 padding='same'))
        #discriminator.add(BatchNormalization(momentum=0.9))
        discriminator.add(Dropout(self.dropout))

        # 'LAYER -3'
        # 'In: 60 X 1 X 1, depth =50'
        # 'Out: 30 X 1 X 1, depth = 75'
        discriminator.add(Conv1D(filters=self.disFilters * 4,
                                 kernel_size=self.filterSize, strides=2,
                                 bias_initializer=Constant(0.1), activation='relu',
                                 padding='same'))
        #discriminator.add(BatchNormalization(momentum=0.9))
        discriminator.add(Dropout(self.dropout))

        # 'LAYER -4'
        # 'In: 30 X 1 X 1, depth =50'
        # 'Out: 15 X 1 X 1, depth = 100'
        discriminator.add(Conv1D(filters=self.disFilters * 8,
                                 kernel_size=self.filterSize, strides=2,
                                 bias_initializer=Constant(0.1), activation='relu',
                                 padding='same'))
        #discriminator.add(BatchNormalization(momentum=0.9))
        discriminator.add(Dropout(self.dropout))

        # Output Layer
        discriminator.add(Flatten())
        discriminator.add(Dense(1))
        discriminator.add(Activation('sigmoid'))

        return discriminator

    def disModel_CV_L6s2_rep(self, initModel=''):
        """
        This function creates a discriminator model which creates the final representation

        :param initModel: The model from which the weights are to be extracted if any
        :return:
        """

        if not initModel:
            model_l2 = Sequential()
            model_l2.add(Conv1D(filters=20, kernel_size=11, strides=2, input_shape=(240, 1),
                                padding='same', activation='relu'))
            model_l2.add(Conv1D(filters=40, kernel_size=11, strides=2,
                                padding='same', activation='relu'))
            model_l2.add(Conv1D(filters=80, kernel_size=11, strides=2,
                                padding='same', activation='relu'))
            model_l2.add(Conv1D(filters=160, kernel_size=11, strides=2,
                                padding='same', activation='relu'))
            model_l2.add(Flatten())
            model_l2.compile(loss='binary_crossentropy', optimizer=Adam())
        else:
            model_l2 = Sequential()
            model_l2.add(Conv1D(filters=20, kernel_size=11, strides=2, input_shape=(240, 1),
                                weights=initModel.layers[0].get_weights(), padding='same',
                                activation='relu'))
            model_l2.add(Conv1D(filters=40, kernel_size=11, strides=2,
                                weights=initModel.layers[2].get_weights(), padding='same',
                                activation='relu'))
            model_l2.add(Conv1D(filters=80, kernel_size=11, strides=2,
                                weights=initModel.layers[4].get_weights(), padding='same',
                                activation='relu'))
            model_l2.add(Conv1D(filters=160, kernel_size=11, strides=2,
                                weights=initModel.layers[6].get_weights(), padding='same',
                                activation='relu'))
            model_l2.add(Flatten())
            model_l2.compile(loss='binary_crossentropy', optimizer=Adam())

        return model_l2
    


