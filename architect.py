import tensorflow as tf
import numpy as np
import params
import model_utils


class CNN:
    """
        Set up CNN model
    """
    def __init__(self, trainable=True, dropout=0.5, node_fuly = params.NODE):
        self.trainable = trainable
        self.dropout = dropout
        self.node_fuly = node_fuly


    def inference(self):
        """
            CNN model
        """
        # _input = tf.reshape(self.input, [params.BATCH_SIZE, params.INPUT_SIZE, params.INPUT_SIZE, 1])
        net = tf.layers.conv2d(inputs=input,
                                filters=32,
                                kernel_size=[5, 5],
                                strides=[1, 1],
                                activation='relu',
                                padding='same',
                                name='conv1')
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], name='pool1')
        net = tf.layers.conv2d(inputs=net,
                                filters=64,
                                kernel_size=[5, 5],
                                strides=[1, 1],
                                activation='relu',
                                padding='valid',
                                name='conv2')
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], name='pool2')
        # flatten
        net = tf.reshape(net, [net.get_shape()[0], -1], name='flatten')
        net = tf.layers.dense(inputs=net, units=self.embedding_size, activation='relu', name='fully_1')
        net = tf.layers.dropout(net, rate=self.dropout, training=self.trainable, name='dropout')
        output = tf.layers.dense(inputs=net, units=params.CLASSES, activation='softmax', name='embeddings')

        return output
