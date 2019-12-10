# hyper params
LR = 0.0001
ALPHA = 0.5
LAMBDA = 0.01
TRAIN_MODE = True
RATE = 0.5
NODE = 1024
EMBEDDING_SIZE = 128
# params
INPUT_SIZE = 28
NUM_CHANNELS = 1

CLASSES = 10

# params data
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIR = './data'
DATA_TRAIN = 60000
DATA_TEST = 10000
RATIO = 0.1

# params training
BATCH_SIZE = 64
EPOCHS = 15
LOG_DIR = './log'
SAVER_DIR = './model_save'
