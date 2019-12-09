import tensorflow as tf
import os
import params
import data_utils
import architect
import model_utils


graph = tf.Graph()

with graph.as_default():
    # config session
    # sess = tf.Session()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with tf.device("/device:GPU:0"):
        train_data, train_labels, validation_data, validation_labels = data_utils.train_test_split()
        # setup placeholder
        input = tf.placeholder(tf.float32, [None, params.INPUT_SIZE, params.INPUT_SIZE, params.NUM_CHANNELS])
        label = tf.placeholder(tf.int32, [None, params.CLASSES])
        #
        num_images = train_data.shape[0]
        loader_data = data_utils.data_loader(train_data, train_labels, num_images)
        batch_image, batch_label = loader_data.get_next()
        model = architect.CNN(batch_image, batch_label)
with graph.as_default():

    # image_batch, label_batch = loader_data.get_next()
    with tf.device("/device:GPU:0"):
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, params.EPOCHS + 1):
            print("[INFO] Epoch {}/{} - Batch Size {} - {} images".format(epoch, params.EPOCHS, params.BATCH_SIZE, num_images))
            iterator = int(num_images/params.BATCH_SIZE)
            feed_dict = {
                    input: train_data,
                    label: train_labels
            }
            sess.run(loader_data.initializer, feed_dict=feed_dict)
            step = 0
            while step < iterator + 1:
                tensor_list = [model.loss, model.train_op, model.accuracy, model.confusion_accuracy]
                _loss, _, acc, confusion_acc = sess.run(tensor_list)
                print("STEP: ", step)
                print("LOSS: ", _loss)
                print("ACC: ", acc)
                print("CONFUSION: ", confusion_acc)
                print("**************")
                step += 1
            print("==============================================")
