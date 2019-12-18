from shared_util import *
from keras.optimizers import Adam
from tacotron import tacotron
import pickle
import numpy as np
from keras.callbacks import ModelCheckpoint

with open('./data/lj/dataset.pickle', 'rb') as f:
    data = pickle.load(f)

with open('./data/lj/vocab.pickle', 'rb') as f:
    vocabulary = pickle.load(f)

spectro_training = np.asarray(data[0])
mel_spectro_training = np.asarray(data[1])
decoder_input_training = np.asarray(data[2])
text_input_training = np.asarray(data[3])

model = tacotron(N_MEL, R, K1, K2, NB_CHARS_MAX,
                           EMBEDDING_SIZE, MAX_MEL_TIME_LENGTH,
                           MAX_MAG_TIME_LENGTH, N_FFT,
                           len(vocabulary))

model.compile(optimizer=Adam(),
              loss=['mean_absolute_error', 'mean_absolute_error'])

checkpoint = ModelCheckpoint("results/model.h5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

train_history = model.fit([text_input_training, decoder_input_training],
                          [mel_spectro_training, spectro_training],
                          epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                          verbose=1, validation_split=0.20,
                          callbacks=[checkpoint])

print('------SAVING MODEL---------')
model.save('results/model.h5')
