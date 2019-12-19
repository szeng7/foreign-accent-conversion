# Quick validation for testing purposes 
from keras.models import load_model
from sklearn.externals import joblib
from shared_util import *
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.filters
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import pickle
import librosa.display

with open('./data/lj/dataset.pickle', 'rb') as f:
    data = pickle.load(f)

with open('./data/lj/vocab.pickle', 'rb') as f:
    vocabulary = pickle.load(f)

# Unneeded decoder input during validation testing
zeros = np.zeros((1, MAX_MEL_TIME_LENGTH, N_MEL))

# Select sample number, 0 indexed
sample_num = 0
# Try with input from training set, Input text corresponds to the ordering in the metadata.csv
input_from_training = np.asarray([data[3][sample_num]])

# Generate new text to input and test
sentence = 'In the only sense'
text_input = np.asarray([encode_text(sentence, vocabulary)])

# load model
saved_model = load_model('results/model.h5')
predictions = saved_model.predict([text_input])

mel_pred = predictions[0]  # predicted mel spectrogram
mag_pred = predictions[1]  # predicted mag spectrogram

item_index = 0  # pick any index
predicted_spectro_item = mag_pred[item_index]
predicted_audio_item = convert_to_waveform(predicted_spectro_item)

plt.figure(figsize=(14, 5))
save(predicted_audio_item,'temp.wav')
librosa.display.waveplot(predicted_audio_item, sr=SAMPLING_RATE)
plt.show()
