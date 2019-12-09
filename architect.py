import tensorflow as tf
import numpy as np
import params
import model_utils


class CNN:
    """
        Set up CNN model
    """
    def __init__(self, input,
                    label
                    trainable=True,
                    dropout=0.5,
                    node_fuly = params.NODE,
                    embedding_size=1024):
        self.input = input
        self.trainable = trainable
        self.dropout = dropout
        self.node_fuly = node_fuly
        self.embedding_size=1024
        self.operation(input, label)

    def operation(self, input, label):
        # get output
        output = self.model(input)
        # get center loss
        # loss = model_utils.center_loss(embedding=output, labels=label, num_classes=params.CLASSES)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=output, scope='softmax_loss')
        # get accuracy
        # accuracy = tf.metrics.accuracy(labels=label, predictions=output, name='metric_acc')
        correct_predict = tf.equal(tf.argmax(output, 1), tf.argmax(label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        # get accuracy confusion matrix
        confusion_matrix = tf.math.confusion_matrix(labels=tf.argmax(label, 1),
                                                    predictions=tf.argmax(output, 1),
                                                    num_classes=10,
                                                    dtype=tf.dtypes.int32, name='confusion_matrix')
        self.confusion_accuracy = model_utils.confusion_accuracy(confusion_matrix)
        # compile model
        self.optimizer = tf.train.AdamOptimizer(learning_rate=params.LR, name='Adam_optimizer')
        self.train_op = optimizer.minimize(self.loss, name='train_op')

    def model(self, __input):
        """
            CNN model
        """
        _input = tf.reshape(__input, [params.BATCH_SIZE, params.INPUT_SIZE, params.INPUT_SIZE, 1])
        net = tf.layers.conv2d(inputs=_input,
                                filters=32,
                                kernel_size=[5, 5],
                                strides=[1, 1],
                                activation='relu',
                                padding='same',
                                name='conv1')
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=[1, 1], name='pool1')
        net = tf.layers.conv2d(inputs=net,
                                filters=64,
                                kernel_size=[5, 5],
                                strides=[1, 1],
                                activation='relu',
                                padding='valid',
                                name='conv2')
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=[1, 1], name='pool2')
        # flatten
        net = tf.reshape(net, [net.get_shape()[0], -1], name='flatten')
        net = tf.layers.dense(inputs=net, units=self.embedding_size, activation='relu', name='fully_1')
        net = tf.layers.dropout(net, rate=self.dropout, training=self.trainable, name='dropout')
        output = tf.layers.dense(inputs=net, units=params.CLASSES, activation='softmax', name='embeddings')

        return output
