import pickle
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Input, Layer
from tensorflow.keras.layers import Embedding
from tensorflow.keras import regularizers
from tqdm import tqdm

from absl import app
from absl import flags
from absl import logging

from models import *

FLAGS = flags.FLAGS

flags.DEFINE_string('train_data', None, "pickle file of training data")
flags.DEFINE_string('test_data', None, "TFRecords file of test data")
flags.DEFINE_integer('num_epochs', 100, "Number of training epochs")
flags.DEFINE_integer('steps_per_epoch', None, "Steps per training epoch")
flags.DEFINE_integer('batch_size', 32, "Training batch size")
flags.DEFINE_string('loss', 'categorical_crossentropy', "loss")
flags.DEFINE_string('optimizer', 'adam', "Optimizer")
flags.DEFINE_float('learning_rate', 0.001, "Learning rate")
flags.DEFINE_integer('filters', 512, "Number of filters")
flags.DEFINE_list('train_metrics', 'accuracy',
                  'Training metrics (comma separated)')

def main(argv):

    with open(FLAGS.train_data, "rb") as handle:
        train_x, train_y = pickle.load(handle)

        print(train_x)
        print(train_y)

if __name__ == "__main__":
    app.run(main)
