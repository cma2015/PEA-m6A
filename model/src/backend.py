# -*- coding: utf-8 -*-
# Copyright 2021 Minggui Song.
# All rights reserved.

"""Represent a collect flnc information.

What's here:

Uses multiple layers to process the input tensor.
-------------------------------------------

"""
import tensorflow as tf
import tensorflow.keras as Kreas
import logging

from tensorflow.python import keras
from tensorflow.python.keras.layers.preprocessing.category_encoding import CategoryEncoding
from tensorflow.python.keras.layers.recurrent import LSTM

from config import *
from layers import MultiHeadSelfAttention, Self_Attention
from logging import getLogger
from sys_output import Output

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)  # pylint: disable=invalid-name
KERNEL_SIZE = 5
stddev = 1
tf.compat.v1.disable_eager_execution()

def sharedFeatureExtractor(t_sequence, name, reuse = False, is_train = False):
    
    w_init = tf.random_normal_initializer(stddev = stddev)
    b_init = None
    g_init = tf.random_normal_initializer(1., stddev)
    act = lambda x: tf.nn.leaky_relu(x, 0.2)

    kernels = config.TRAIN.KERNEL.split('_')

    with tf.compat.v1.variable_scope(name, reuse = reuse) as vs:

        t_sequence = Kreas.layers.Input(shape=t_sequence)
        
        feature_conv = Kreas.layers.Conv1D(filters=300, kernel_size=1, strides=1, dilation_rate=1, activation=None)(t_sequence)
        feature1 = Kreas.layers.BatchNormalization(beta_initializer=w_init, gamma_initializer=w_init, trainable=is_train)(feature_conv)
        feature1 = Kreas.layers.PReLU()(feature1)
        if config.TRAIN.DROPOUT:
            feature1 = Kreas.layers.Dropout(rate=config.TRAIN.DROPOUT_KEEP)(feature1)
        
        feature2 = Kreas.layers.Conv1D(300, 1, strides=1, dilation_rate=2, activation=None)(t_sequence)
        feature2 = Kreas.layers.BatchNormalization(beta_initializer=w_init, gamma_initializer=w_init, trainable=is_train)(feature2)
        feature2 = Kreas.layers.PReLU()(feature2)
        if config.TRAIN.DROPOUT:
            feature2 = Kreas.layers.Dropout(rate=config.TRAIN.DROPOUT_KEEP)(feature2)

        feature3 = Kreas.layers.Conv1D(300, 1, strides=1, dilation_rate=4, activation=None)(t_sequence)
        feature3 = Kreas.layers.BatchNormalization(beta_initializer=w_init, gamma_initializer=w_init, trainable=is_train)(feature3)
        feature3 = Kreas.layers.PReLU()(feature3)
        if config.TRAIN.DROPOUT:
            feature3 = Kreas.layers.Dropout(rate=config.TRAIN.DROPOUT_KEEP)(feature3)

        features = Kreas.layers.Concatenate()([feature1, feature2, feature3])
        print(features)

        features = Kreas.layers.Conv1D(32, 1, strides=1, dilation_rate=1, activation=None)(features)
        features = Kreas.layers.BatchNormalization(beta_initializer=w_init, gamma_initializer=w_init, trainable=is_train)(features)
        con_features = Kreas.layers.PReLU()(features)
        if config.TRAIN.DROPOUT:
            con_features = Kreas.layers.Dropout(rate=config.TRAIN.DROPOUT_KEEP)(con_features)
            
        features = Kreas.layers.Conv1D(32, 1, strides=1, dilation_rate=1, activation=None)(con_features)
        features = Kreas.layers.BatchNormalization(beta_initializer=w_init, gamma_initializer=w_init, trainable=is_train)(features)
        features = Kreas.layers.PReLU()(features)
        if config.TRAIN.DROPOUT:
            features = Kreas.layers.Dropout(rate=config.TRAIN.DROPOUT_KEEP)(features)

        features = Kreas.layers.Conv1D(32, 1, strides=1, dilation_rate=1, activation=None)(features)
        features = Kreas.layers.BatchNormalization(beta_initializer=w_init, gamma_initializer=w_init, trainable=is_train)(features)
        features = Kreas.layers.PReLU()(features)
        if config.TRAIN.DROPOUT:
            features = Kreas.layers.Dropout(rate=config.TRAIN.DROPOUT_KEEP)(features)
        
        features = Kreas.layers.Add()([features, con_features])
        print(features)

        features = Kreas.layers.Conv1D(32, 1, strides=1, dilation_rate=1, activation=None)(features)
        features = Kreas.layers.BatchNormalization(beta_initializer=w_init, gamma_initializer=w_init, trainable=is_train)(features)
        features = Kreas.layers.PReLU()(features)
        if config.TRAIN.DROPOUT:
            features = Kreas.layers.Dropout(rate=config.TRAIN.DROPOUT_KEEP)(features)

        features = Kreas.layers.Bidirectional(LSTM(config.TRAIN.RNN_HIDDEN, return_sequences=True))(features)

        print(features)
        #features = MultiHeadSelfAttention(128)(features)
        print(features)
        features = Self_Attention(1024)(features)

        return features

def classifier(features, name, reuse = False, is_train = True):

    w_init = tf.random_normal_initializer(stddev=stddev)
    b_init = None 
    g_init = tf.random_normal_initializer(1., stddev)
    act = lambda x: tf.nn.leaky_relu(x, 0.2)

    with tf.compat.v1.variable_scope(name, reuse = reuse) as vs:

        conv_features = Kreas.layers.Conv1D(32, 1, strides=1, dilation_rate=1, activation=None)(features)
        features = Kreas.layers.BatchNormalization(beta_initializer=w_init, gamma_initializer=w_init, trainable=is_train)(conv_features)
        features = Kreas.layers.PReLU()(features)
        # shape=(None, 101, 32)

        features = Kreas.layers.Conv1D(64, 1, strides=1, dilation_rate=1, activation=None)(features)
        features = Kreas.layers.BatchNormalization(beta_initializer=w_init, gamma_initializer=w_init, trainable=is_train)(features)
        fin_features = Kreas.layers.PReLU()(features)
        # shape=(None, 101, 64)

        features = Kreas.layers.Flatten()(fin_features)
        # shape=(None, 6464)

        features = Kreas.layers.Dense(config.TRAIN.FC)(features)
        features = Kreas.layers.BatchNormalization(beta_initializer=w_init, gamma_initializer=w_init, trainable=is_train)(features)
        hidden = Kreas.layers.PReLU()(features)

        category = Kreas.layers.Dense(2)(hidden)
        # shape=(None, 2)

        return category

if __name__ == '__main__':
    '''
    test model building
    '''
    print("start testing")
    sequences = tf.compat.v1.placeholder(tf.float32, [None, config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM])
    sequence = (config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM)
    selecting = sharedFeatureExtractor(sequence, 'extrator')
    category = classifier(selecting, 'classifier')
    
