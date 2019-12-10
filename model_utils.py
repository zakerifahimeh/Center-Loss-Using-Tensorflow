import tensorflow as tf
import params


def center_loss_logits(features, labels, num_classes):
    """
    embedding dim : (batch_size, num_features)
    """
    print("======feature: ", features)
    print("======LABEL: ", labels)
    len_features = features.get_shape()[1]
    print("======OUTPUT len feature: ", len_features)
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    print("======OUTPUT CENTER: ", centers)
    labels = tf.reshape(labels, [-1])
    print("======OUTPUT LABEL: ", labels)

    centers_batch = tf.gather(centers, labels)
    print("======OUTPUT CENTER BATCH: ", centers_batch)
    loss = tf.nn.l2_loss(features - centers_batch)
    print("======OUTPUT LOSS: ", loss)

    diff = centers_batch - features
    print("======OUTPUT CONV 1: ", net)

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    print("======OUTPUT CONV 1: ", net)
    appear_times = tf.gather(unique_count, unique_idx)
    print("======OUTPUT CONV 1: ", net)
    appear_times = tf.reshape(appear_times, [-1, 1])
    print("======OUTPUT CONV 1: ", net)

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op


def center_loss_one_hot(embeddings, labels, num_classes):
    """

    """
    centers = tf.get_variable(name='centers', shape=[num_classes, params.EMBEDDING_SIZE],
                              initializer=tf.random_normal_initializer(stddev=0.1), trainable=False)
    label_indices = tf.argmax(labels, 1)
    centers_batch = tf.nn.embedding_lookup(centers, label_indices)
    center_loss = params.LAMBDA * tf.nn.l2_loss(embeddings - centers_batch) / tf.to_float(tf.shape(embeddings)[0])
    new_centers = centers_batch - embeddings
    labels_unique, row_indices, counts = tf.unique_with_counts(label_indices)

    centers_update = tf.unsorted_segment_sum(new_centers, row_indices, tf.shape(labels_unique)[0]) / tf.to_float(counts)
    centers = tf.scatter_sub(centers, labels_unique, params.ALPHA * centers_update)
    return center_loss, centers


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


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
