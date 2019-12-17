# UTIL FILE WITH ALL SHARED VARIABLES AND FUNCTIONS
import os
import librosa
import numpy as np
import copy
from scipy import signal
from scipy.io import wavfile

#constants to be used when creating spectrograms/preprocessing

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


# Deep Learning Model
K1 = 16  # Size of the convolution bank in the encoder CBHG
K2 = 8  # Size of the convolution bank in the post processing CBHG
BATCH_SIZE = 32
NB_EPOCHS = 10
EMBEDDING_SIZE = 256

# Other
TRAIN_SET_RATIO = 0.9


def get_spectrograms(wav):
    """
    Helper function to convert wav form to spectrograms
    """

    waveform = wav.astype('float')
    waveform, _ = librosa.effects.trim(waveform)

    #filter out lower frequencies
    waveform = np.append(waveform[0], waveform[1:] - PREEMPHASIS * waveform[:-1])

    #short time fourier calculated
    stft_matrix = librosa.stft(y=waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

    #magnitude and mel spectrograms calculated
    spectrogram = np.abs(stft_matrix)

    mel_transform_matrix = librosa.filters.mel(SAMPLING_RATE, N_FFT, N_MEL)
    mel_spectrogram = np.dot(mel_transform_matrix, spectrogram)

    #convert to DB
    mel_spectrogram = 20 * np.log10(np.maximum(1e-5, mel_spectrogram))
    spectrogram = 20 * np.log10(np.maximum(1e-5, spectrogram))

    #normalize
    mel_spectrogram = np.clip((mel_spectrogram - REF_DB + MAX_DB) / MAX_DB, 1e-8, 1)
    spectrogram = np.clip((spectrogram - REF_DB + MAX_DB) / MAX_DB, 1e-8, 1)

    #(time, freq)
    mel_spectrogram = mel_spectrogram.T.astype(np.float32)
    spectrogram = spectrogram.T.astype(np.float32)

    return mel_spectrogram, spectrogram

def get_padded_spectrograms(wav):
    """
    Helper function to pad the spectrograms to specific length
    """
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

def save_wav(wav, path, sr):
    # Change to 16 bit audio
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	wavfile.write(path, sr, wav.astype(np.int16))

def get_griffin_lim(spectrogram, n_fft, hop_length,
                    win_length, window_type, n_iter):

    spectro = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        estimated_wav = spectro_inversion(spectro, hop_length,
                                          win_length, window_type)
        est_stft = librosa.stft(estimated_wav, n_fft,
                                hop_length,
                                win_length=win_length)
        phase = est_stft / np.maximum(1e-8, np.abs(est_stft))
        spectro = spectrogram * phase
    estimated_wav = spectro_inversion(spectro, hop_length,
                                      win_length, window_type)
    result = np.real(estimated_wav)

    return result


def spectro_inversion(spectrogram, hop_length, win_length, window_type):
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window=window_type)


def from_spectro_to_waveform(spectro, n_fft, hop_length,
                             win_length, n_iter, window_type,
                             max_db, ref_db, preemphasis):
    # transpose
    spectro = spectro.T

    # de-noramlize
    spectro = (np.clip(spectro, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    spectro = np.power(10.0, spectro * 0.05)

    # wav reconstruction
    waveform = get_griffin_lim(spectro, n_fft, hop_length,
                               win_length,
                               window_type, n_iter)

    # de-preemphasis
    waveform = signal.lfilter([1], [1, -preemphasis], waveform)

    # trim
    waveform, _ = librosa.effects.trim(waveform)

    return waveform.astype(np.float32)

