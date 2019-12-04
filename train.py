import tensorflow as tf
import os
import params
import data_utils
import architect


graph = tf.Graph()

with graph.as_default():

    # define placeholder
    input = tf.placeholder(shape=[params.BATCH_SIZE, params.INPUT_SIZE, params.INPUT_SIZE, 1], dtype=tf.float32)
    label = tf.placeholder(shape=[params.BATCH_SIZE, params.CLASSES], dtype=tf.int32])
    # get output
    model = architect.CNN()
    output = model.inference()
    # get center loss
    loss = model_utils.center_loss(embedding=output, labels=label, num_classes=params.CLASSES)
    # get accuracy
    # accuracy = tf.metrics.accuracy(labels=label, predictions=output, name='metric_acc')
    correct_predict = tf.equal(tf.argmax(output, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    # get accuracy confusion matrix
    confusion_matrix = tf.math.confusion_matrix(labels=label, predictions=output, dtype=tf.dtypes.int32, name='confusion_matrix')
    confusion_accuracy = model_utils.confusion_accuracy(confusion_matrix)
    # compile model
    optimizer = tf.train.AdamOptimizer(learning_rate=params.LR, name='Adam_optimizer')
    train_op = optimizer.minimizer(loss, name='train_op')
