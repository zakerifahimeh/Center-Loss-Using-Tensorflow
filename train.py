import tensorflow as tf
import os
import params
import data_utils
import architect


graph = tf.Graph()

with graph.as_default():
    with tf.device('/gpu:0'):
        train_data, train_labels, validation_data, validation_labels = data_utils.train_test_split()
        # define placeholder
        input = tf.placeholder(shape=[params.BATCH_SIZE, params.INPUT_SIZE, params.INPUT_SIZE, 1], dtype=tf.float32)
        label = tf.placeholder(shape=[params.BATCH_SIZE, params.CLASSES], dtype=tf.int32)
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

with graph.as_default():
    with tf.device('/gpu:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        global_step = tf.global_variables_initializer()
        for epoch in range(1, params.EPOCHS + 1):
            print("[INFO] Epoch {}/{} - Batch Size {}".format(epoch, params.EPOCHS, params.BATCH_SIZE))
            iterator = 0
            while iterator < 32:
                tensor_list = [global_step, loss, train_op, accuracy, confusion_accuracy]
                feed_dict = {
                        input: train_data,
                        label: train_labels
                }
                _step, _loss, _, acc, confusion_acc = sess.run(tensor_list, feed_dict=feed_dict)
                print("STEP: ", _step)
                print("LOSS: ", _loss)
                print("ACC: ", acc)
                print("CONFUSION: ", confusion_acc)
                iterator += 1
            print("=========================================")
