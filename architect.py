import tensorflow as tf
import numpy as np
import params
import model_utils


class CNN:
    """
        Set up CNN model
    """
    def __init__(self, input, label):
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
        self.train_op = self.optimizer.minimize(self.loss, name='train_op')

    def model(self, __input):
        """
            CNN model
        """
        _input = tf.reshape(__input, [params.BATCH_SIZE, params.INPUT_SIZE, params.INPUT_SIZE, 1])
        print("INPUT SHAPE: ", _input.get_shape())
        net = tf.layers.conv2d(inputs=_input,
                                filters=32,
                                kernel_size=[5, 5],
                                strides=[1, 1],
                                activation=tf.nn.relu,
                                padding='same',
                                bias_initializer=tf.zeros_initializer(),
                                name='conv1')
        # print("======OUTPUT CONV 1: ", net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=[1, 1], name='pool1')
        # print("======OUTPUT MAXPOOLING 1: ", net)
        net = tf.layers.conv2d(inputs=net,
                                filters=64,
                                kernel_size=[5, 5],
                                strides=[1, 1],
                                activation=tf.nn.relu,
                                padding='valid',
                                bias_initializer=tf.zeros_initializer(),
                                name='conv2')
        # print("======OUTPUT CONV 2: ", net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=[1, 1], name='pool2')
        # print("======OUTPUT MAXPOOLING 2: ", net)
        # flatten
        net = tf.reshape(net, [net.get_shape()[0], -1], name='flatten')
        # print("SHAPE FLATTEN: ", net.get_shape())
        net = tf.layers.dense(inputs=net,
                                units=params.NODE,
                                activation=tf.nn.relu,
                                bias_initializer=tf.zeros_initializer(),
                                name='fully_1')
        # print("======OUTPUT FULLY 1: ", net)
        net = tf.layers.dropout(inputs=net, rate=params.RATE, training=params.TRAIN_MODE, name='dropout')
        # print("======OUTPUT DROPOUT 1: ", net)
        output = tf.layers.dense(inputs=net,
                                    units=params.CLASSES,
                                    activation=tf.nn.softmax,
                                    bias_initializer=tf.zeros_initializer(),
                                    name='embeddings')
        # print("======OUTPUT OUTPUT: ", output)

        return output
