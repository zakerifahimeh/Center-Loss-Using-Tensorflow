import tensorflow as tf
import params


def center_loss(embedding, labels, num_classes, name=''):
    '''
    embedding dim : (batch_size, num_features)
    '''
    label = tf.argmax(labels, 1)
    with tf.variable_scope(name):

        num_features = embedding.get_shape()[1]
        centroids = tf.get_variable('c',shape=[num_classes,num_features],
                            dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),trainable=False)
        centroids_delta = tf.get_variable('centroidsUpdateTempVariable',shape=[num_classes,num_features],dtype=tf.float32,
                            initializer=tf.zeros_initializer(),trainable=False)


        centroids_batch = tf.gather(centroids,label)

        loss = tf.nn.l2_loss(embedding - centroids_batch) / float(params.BATCH_SIZE) # Eq. 2

        diff = centroids_batch - embedding

        delta_c_nominator = tf.scatter_add(centroids_delta, label, diff)

        indices = tf.expand_dims(labels,-1)
        updates = tf.constant(value=1,shape=[indices.get_shape()[0]],dtype=tf.float32)
        shape = tf.constant([num_classes])
        labels_sum = tf.expand_dims(tf.scatter_nd(indices, updates, shape),-1)


        centroids = centroids.assign_sub(
        params.ALPHA * delta_c_nominator / (1.0 + labels_sum))

        centroids_delta = centroids_delta.assign(tf.zeros([num_classes,num_features]))

        '''
        # The same as with scatter_nd
        one_hot_labels = tf.one_hot(y,num_classes)
        labels_sum = tf.expand_dims(tf.reduce_sum(one_hot_labels,reduction_indices=[0]),-1)
        centroids = centroids.assign_sub(ALPHA * delta_c_nominator / (1.0 + labels_sum))
        '''
        return loss, centroids


def non_nan_average(x):
    """
        Computes the average of all elements that are not NaN in a rank 1 tensor
    """
    nan_mask = tf.debugging.is_nan(x)
    x = tf.boolean_mask(x, tf.logical_not(nan_mask))
    return tf.keras.backend.mean(x)


def confusion_accuracy(confusion_matrix):
    """
        return accuracy of confusion matrix
    """
    diag = tf.linalg.tensor_diag_part(confusion_matrix)
    total_per_calss = tf.reduce_sum(confusion_matrix, axis=1)
    acc_per_class = diag / tf.maximum(1, total_per_calss)
    accuracy = non_nan_average(acc_per_class)

    return accuracy
