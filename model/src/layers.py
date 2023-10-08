# -*- coding: utf-8 -*-
# Copyright 2021 Minggui Song.
# All rights reserved.
#

"""Represent a collect flnc information.

What's here:

Uses multiple layers to process the input tensor.
-------------------------------------------

Classes:
    - MultiHeadSelfAttention
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class MultiHeadSelfAttention(Layer):
    def __init__(self,
                embed_dim: int,
                num_heads: int = 8,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=None):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        if embed_dim % num_heads != 0:
            raise ValueError(
                f'embedding dimension = {embed_dim} should be '
                f'divisible by number of heads = {num_heads}')
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(
            units=embed_dim,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.key_dense = Dense(
            units=embed_dim,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.value_dense = Dense(
            units=embed_dim,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)
        self.combine_heads = Dense(
            units=embed_dim,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer)

    def attention(self, query, key, value, mask):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        if mask is not None:
            scaled_score += (mask * -1e9)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size: int):
        x = tf.reshape(x, (batch_size, -1,
                        self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, s3eq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
            )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
            )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
            )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value, mask)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
            )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, seq_len, self.embed_dim)
            )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
            )  # (batch_size, seq_len, embed_dim)
        return output

    def get_config(self):
        config = {'embed_dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'kernel_initializer': self.kernel_initializer,
                    'kernel_regularizer': self.kernel_regularizer}
        base_config = super(MultiHeadSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))