from hparams import *
from keras.optimizers import Adam
from model.tacotron import get_tacotron_model
import pickle
import numpy as np

# import prepared data
# decoder_input_training = joblib.load('data/decoder_input_training.pkl')
# mel_spectro_training = joblib.load('data/mel_spectro_training.pkl')
# spectro_training = joblib.load('data/spectro_training.pkl')

# text_input_training = joblib.load('data/text_input_ml_training.pkl')
# vocabulary = joblib.load('data/vocabulary.pkl')

# with open('./data/lj/all.raw.pickle', 'rb') as f:
#     data = pickle.load(f)

#     print(data)

with open('./data/lj/small.pickle', 'rb') as f:
    data = pickle.load(f)

with open('./data/lj/vocab.pickle', 'rb') as f:
    vocabulary = pickle.load(f)

spectro_training = np.asarray(data[0])
mel_spectro_training = np.asarray(data[1])
decoder_input_training = np.asarray(data[2])
text_input_training = np.asarray(data[3])

model = get_tacotron_model(N_MEL, r, K1, K2, NB_CHARS_MAX,
                           EMBEDDING_SIZE, MAX_MEL_TIME_LENGTH,
                           MAX_MAG_TIME_LENGTH, N_FFT,
                           vocabulary)
#model.summary()
opt = Adam()
model.compile(optimizer=opt,
              loss=['mean_absolute_error', 'mean_absolute_error'])

train_history = model.fit([text_input_training, decoder_input_training],
                          [mel_spectro_training, spectro_training],
                          epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                          verbose=1, validation_split=0.15)

print('------SAVING MODEL---------')
# joblib.dump(train_history.history, 'results/training_history.pkl')
model.save('results/model.h5')
