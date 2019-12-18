# UTIL FILE WITH ALL SHARED VARIABLES AND FUNCTIONS
import os
import librosa
import numpy as np
import copy
from scipy import signal
from scipy.io import wavfile


# Model Parameters
K1 = 16  # Size of the convolution bank in the encoder CBHG
K2 = 8  # Size of the convolution bank in the post processing CBHG
BATCH_SIZE = 32
NB_EPOCHS = 10
EMBEDDING_SIZE = 256

# Signal Parameters
N_FFT = 1024 #number of fourier transform points
WINDOW_TYPE='hann' #type of window for FT
PREEMPHASIS = 0.97 #importance factor on high freq signals
SAMPLING_RATE = 16000
FRAME_LENGTH = 0.05  # seconds, window length
FRAME_SHIFT = 0.0125  # seconds, temporal shift
HOP_LENGTH = int(SAMPLING_RATE * FRAME_SHIFT)
WIN_LENGTH = int(SAMPLING_RATE * FRAME_LENGTH)
N_MEL = 80 #num of mel bands
REF_DB = 20 #reference level decibel
MAX_DB = 100 #max decibel
R = 5 #reduction factor
MAX_MEL_TIME_LENGTH = 200  #max of the time dimension for a mel spectrogram
MAX_MAG_TIME_LENGTH = 850  #max of the time dimension for a spectrogram
NB_CHARS_MAX = 200  #max of the input text 
N_ITER = 50 #griffin-lim iterations

def get_spectrograms(wav):
    waveform = wav.astype('float32')
    waveform, interval_data = librosa.effects.trim(waveform)

    #premphasis to get better audio results
    waveform = signal.lfilter([1, -PREEMPHASIS], [1], waveform)

    #FourierTransform
    stft = np.abs(librosa.stft(y=waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH))
    spectrogram = librosa.amplitude_to_db(stft)
    mel_spectrogram = librosa.feature.melspectrogram(waveform, sr=SAMPLING_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                    win_length=WIN_LENGTH, window=WINDOW_TYPE, n_mels=N_MEL)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    mel_spectrogram = normalize(mel_spectrogram)
    spectrogram = normalize(spectrogram)
    mel_spectrogram = mel_spectrogram.T.astype(np.float32)
    spectrogram = spectrogram.T.astype(np.float32)

    return mel_spectrogram, spectrogram

def get_padded_spectrograms(wav):
    mel_spectrogram, spectrogram = get_spectrograms(wav)
    time = mel_spectrogram.shape[0]
    if time % R != 0:
        nb_paddings = R - (time % R)
    else:
        nb_paddings = 0
    mel_spectrogram = np.pad(mel_spectrogram,
                        [[0, nb_paddings], [0, 0]],
                        mode="constant")
    spectrogram = np.pad(spectrogram,
                    [[0, nb_paddings], [0, 0]],
                    mode="constant")
    return mel_spectrogram.reshape((-1, N_MEL * R)), spectrogram

def normalize(data):
    return np.clip((data + MAX_DB - REF_DB) / MAX_DB, 0, 1)

def denormalize(data):
    return (np.clip(data, 0, 1) * MAX_DB) - MAX_DB + REF_DB

def save(wav, path):
	wavfile.write(path, SAMPLING_RATE, (convert_to_16bit(wav).astype(np.int16)))

def convert_to_16bit(wav):
    output =  wav * 32767 / max(0.01, np.max(np.abs(wav)))
    return output

def convert_to_waveform(spectrogram):
    spectrogram = spectrogram.T
    fft_amps = librosa.core.db_to_amplitude(denormalize(spectrogram))
    waveform = librosa.core.griffinlim(fft_amps, n_iter=N_ITER, hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH, window=WINDOW_TYPE)
    # invert pre-emphasis
    waveform = signal.lfilter([1], [1, -PREEMPHASIS], waveform)
    trimmed_signal, interval_data = librosa.effects.trim(waveform)
    return trimmed_signal.astype(np.float32)

# Encode input text into vocabulary
def encode_text(input, vocabulary):
    output = [vocabulary[char] for char in list(input.lower().replace(" ", ""))]

    if len(output) < NB_CHARS_MAX:
        for i in range(NB_CHARS_MAX - len(output)):
            output.append(vocabulary['P'])
    else:
        output = output[:NB_CHARS_MAX]   
    return output
 
