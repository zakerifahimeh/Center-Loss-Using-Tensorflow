import tensorflow as tf
import numpy as np
import params
import os
import gzip
from urllib.request import urlretrieve as download


def pull_dataset(filename):
    """
        Download dataset from Yann's website if dataset do not exists in data directory
    """
    if not os.path.exists(params.DATA_DIR):
        os.mkdir(params.DATA_DIR)
    file_path = os.path.join(params.DATA_DIR, filename)
    if not os.path.exists(file_path):
        file_path, _ = download(params.SOURCE_URL + filename, file_path)
        with tf.gfile.GFile(file_path) as f:
            size = f.size()
        print("Successfully dowloaded dataset: {} - {}".format(filename, size))

    return file_path


def extract_data(file, num_image):
    """
        Extract the images into a 4D tensor [img_index, y, x, channels]
    """
    print("Extracting: ", file)
    with gzip.open(file) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(params.INPUT_SIZE * params.INPUT_SIZE * num_image * params.NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # normalize data from [0, 255] to [-1, 1]
        data = 2 * (data / 255.0) - 1
        data = data.reshape(num_image, params.INPUT_SIZE, params.INPUT_SIZE, params.NUM_CHANNELS)
        # data = np.reshape(data, [num_image, -1])

    return data


def extract_label(file, num_image):
    """
        Extract the labels into on hot vector with int32 type
    """
    with gzip.open(file) as bytestream:
        bytestream.read(8)
        buf = bytestream.read( 1 * num_image)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data, params.CLASSES))
        one_hot_encoding[np.arange(num_labels_data), labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, params.CLASSES])

    return one_hot_encoding


def train_test_split():
    """"
        dowload, extract and split dataset
    """
    # Get the data.
    print("==========[INFO] DOWNLOADING DATA TRAIN==========")
    train_data_filename = pull_dataset('train-images-idx3-ubyte.gz')
    print("==========[INFO] DOWNLOADING LABELS TRAIN==========")
    train_labels_filename = pull_dataset('train-labels-idx1-ubyte.gz')
    print("==========[INFO] DOWNLOADING DATA TEST==========")
    test_data_filename = pull_dataset('t10k-images-idx3-ubyte.gz')
    print("==========[INFO] DOWNLOADING LABELS TEST==========")
    test_labels_filename = pull_dataset('t10k-labels-idx1-ubyte.gz')
    print("==========[INFO] DATA DOWNLOADED==========")

    # Extract it into numpy arrays.
    print("==========[INFO] EXTRACTING DATA DOWNLOADED==========")
    train_data = extract_data(train_data_filename, params.DATA_TRAIN)
    train_labels = extract_label(train_labels_filename, params.DATA_TRAIN)
    test_data = extract_data(test_data_filename, params.DATA_TEST)
    test_labels = extract_label(test_labels_filename, params.DATA_TEST)
    print("==========[INFO] DATA EXTRACTED==========")

    # split dataset
    val_size = int(params.DATA_TRAIN * params.RATIO)
     # Generate a validation set.
    validation_data = train_data[:val_size, :]
    validation_labels = train_labels[:val_size, :]
    train_data = train_data[val_size:, :]
    train_labels = train_labels[val_size:, :]

    return train_data, train_labels, validation_data, validation_labels


def data_loader(images, labels, num_image):
    """
        Setup data loader
    """
    # setup image
    image_ds = tf.data.Dataset.from_tensor_slices(images)
    print('shape: ', repr(image_ds.output_shapes))
    print('type: ', image_ds.output_types)
    print()
    # setup label
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    print('shape: ', repr(label_ds.output_shapes))
    print('type: ', label_ds.output_types)
    print()
    #
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    # Setting a shuffle buffer size as large as the dataset ensures that the data is
    # completely shuffled.
    train_ds = image_label_ds.cache(filename='./cache.tf-data-train')
    train_ds = train_ds.shuffle(buffer_size=num_image)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(params.BATCH_SIZE)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    iterator = train_ds.make_initializable_iterator()

    return iterator
