# Quick validation for testing purposes 
# TODO create testing loop of audio signals
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
sentence = 'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
text_input = np.asarray([encode_text(sentence, vocabulary)])

# load model
saved_model = load_model('results/model-best.h5')
predictions = saved_model.predict([text_input, zeros])

mel_pred = predictions[0]  # predicted mel spectrogram
mag_pred = predictions[1]  # predicted mag spectrogram

item_index = 0  # pick any index
predicted_spectro_item = mag_pred[item_index]
predicted_audio_item = from_spectro_to_waveform(predicted_spectro_item, N_FFT,
                                                HOP_LENGTH, WIN_LENGTH,
                                                N_ITER, WINDOW_TYPE,
                                                MAX_DB, REF_DB, PREEMPHASIS)

plt.figure(figsize=(14, 5))
save_wav(predicted_audio_item,'temp.wav',sr=SAMPLING_RATE)
librosa.display.waveplot(predicted_audio_item, sr=SAMPLING_RATE)
plt.show()
