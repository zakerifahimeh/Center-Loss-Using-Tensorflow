import tensorflow as tf
import os
import params
import data_utils
import architect
import model_utils


graph = tf.Graph()

with graph.as_default():
    with tf.device('/gpu:0'):
        train_data, train_labels, validation_data, validation_labels = data_utils.train_test_split()
        # setup placeholder
        input = tf.placeholder(tf.float32, [None, params.INPUT_SIZE, params.INPUT_SIZE, params.NUM_CHANNELS])
        label = tf.placeholder(tf.int32, [None, params.CLASSES])
        #
        loader_data, num_images = data_utils.data_loader(train_data, train_labels)
        batch_image, batch_label = loader_data.get_next()
        model = architect.CNN(input=batch_image, label=batch_label)
with graph.as_default():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # image_batch, label_batch = loader_data.get_next()
    with tf.device('/gpu:0'):
        global_step = tf.global_variables_initializer()
        for epoch in range(1, params.EPOCHS + 1):
            print("[INFO] Epoch {}/{} - Batch Size {} - {} images".format(epoch, params.EPOCHS, params.BATCH_SIZE, num_images))
            iterator = 0
            while iterator < 32:
                tensor_list = [loader_data.initializer, global_step, model.loss, model.train_op, model.accuracy, model.confusion_accuracy]
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
