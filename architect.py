import tensorflow
import numpy as np
import params


class CNN:
    """
        Set up CNN model
    """
    def __init__(self,input trainable=True, dropout=0.5, embedding_size=1024):
        self.input = input
        self.trainable = trainable
        self.dropout = dropout
        self.embedding_size = embedding_size


    def inference(self):
        """
            CNN model
        """
        _input = tf.reshape(self.input, [params.BATCH_SIZE, params.INPUT_SIZE, params.INPUT_SIZE, 1])
        net = tf.layers.Conv2D(filters=32,
                                kernel_size=[5, 5],
                                strides=[1, 1],
                                activation='relu',
                                padding='same',
                                name='conv1')
        net = tf.layers.MaxPool2D(pool_size=[2, 2], name='pool1')
        net = tf.layers.Conv2D(filters=64,
                                kernel_size=[5, 5],
                                strides=[1, 1],
                                activation='relu'
                                padding='valid',
                                name='conv2')
        net = tf.layers.MaxPool2D(pool_size=[2, 2], name='pool2')
        # flatten
        net = tf.reshape(net, [net.get_shape()[0], -1], name='flatten')
        net = tf.layers.dropout(net, rate=self.dropout, training=trainable, name='dropout')
        output = tf.layers.dense(inputs=net, units=params.CLASSES, activation='sigmoid')
