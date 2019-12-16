"""
Preprocessing wav files
"""
from scipy.io import wavfile
import numpy as np
import pickle
import argparse as ap
import os
import csv
import tensorflow as tf

import matplotlib.pyplot as plt
import librosa

import sys



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
NB_CHARS_MAX = 200  #max of the input text data


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

def main():

    p = ap.ArgumentParser()

    p.add_argument('--raw-data-dir', required=False, \
        help='directory of raw wav files for training')
    p.add_argument('--data-dir', required=False, \
        help='data directory of wav files')
    ARGS = p.parse_args()

    if ARGS.raw_data_dir:
        #download and parse the data from the local repo
        id_to_wav_dict = {}

        print("starting")
        i = 0

        for wav in os.listdir(ARGS.raw_data_dir + "/wavs"):
            if ".wav" in wav:
                id = wav.rstrip('.wav')
                path = ARGS.raw_data_dir + "/wavs/" + wav

                data, rate = librosa.load(path, sr=SAMPLING_RATE)
                if id not in id_to_wav_dict:
                    id_to_wav_dict[id] = [data]
            i += 1
            if i % 100 == 0:
                print(f"Done with {i}")

        print("done extracting all wavs")

        with open(ARGS.raw_data_dir + "/metadata.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='|')
            for row in csv_reader:
                id = row[0]
                raw_transcription = row[1]
                if id in id_to_wav_dict:
                    id_to_wav_dict[id].append(raw_transcription)

        with open(ARGS.data_dir + "/all.raw.pickle", 'wb') as f:
            pickle.dump(id_to_wav_dict, f)

    else:

        with open(ARGS.data_dir + "/all.raw.pickle", 'rb') as f:
            id_to_wav_dict = pickle.load(f)
            path = '/Volumes/Elements/lj/LJSpeech-1.1/wavs/LJ001-0001.wav'
            #print(id_to_wav_dict['LJ001-0001'])
            wav, rate = librosa.load(path, sr=SAMPLING_RATE)
            mel_spectrogram, spectrogram = get_padded_spectrograms(wav)
            sess = tf.Session()
            decod_inp_tensor = tf.concat((tf.zeros_like(mel_spectrogram[:1, :]),
                                mel_spectrogram[:-1, :]), 0)
            decod_inp = sess.run(decod_inp_tensor)
            decod_inp = decod_inp[:, -N_MEL:]

            #padding time dimension
            padded_mel_spectrogram = np.zeros((MAX_MEL_TIME_LENGTH, mel_spectrogram.shape[1]))
            padded_mel_spectrogram[:mel_spectrogram.shape[0], :mel_spectrogram.shape[1]] = mel_spectrogram[:MAX_MEL_TIME_LENGTH]

            padded_decod_input = np.zeros((MAX_MEL_TIME_LENGTH, decod_inp.shape[1]))
            padded_decod_input[:decod_inp.shape[0], :decod_inp.shape[1]] = decod_inp[:MAX_MEL_TIME_LENGTH]

            padded_spectrogram = np.zeros((MAX_MAG_TIME_LENGTH, spectrogram.shape[1]))
            padded_spectrogram[:spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram[:MAX_MAG_TIME_LENGTH]

            tuple = (padded_mel_spectrogram, padded_spectrogram, padded_decod_input)

            """
            #40-40-20 split
            #train1-train2-test, splitting train since datasets would be too large to pickle
            split1 = 0.4*len(id_to_wav_dict.items())
            split2 = 0.8*len(id_to_wav_dict.items())

            train1_dataset = []
            train2_dataset = []
            test_dataset = []
            toy_dataset = []

            counter = 0
            vocab = {} #for encoding text, character by character
            vocab['P'] = 0
            i = 1

            temp = {}
            temp['LJ001-0001'] = data
            id_to_wav_dict = temp

            for key, value in id_to_wav_dict.items():
                counter += 1
                wav = value[0]
                sentence = value[1]

                mel_spectrogram, spectrogram = get_padded_spectrograms(wav)

                list_of_existing_chars = list(set(sentence.lower().replace(" ", "")))
                for char in list_of_existing_chars:
                    if char not in vocab:
                        vocab[char] = i
                        i += 1

                list_of_char_id = [vocab[char] for char in list(sentence.lower().replace(" ", ""))]

                if len(list_of_char_id) < 200:
                    for i in range(200 - len(list_of_char_id)):
                        list_of_char_id.append(vocab['P'])
                else:
                    list_of_char_id = list_of_char_id[:200]

                sess = tf.Session()
                decod_inp_tensor = tf.concat((tf.zeros_like(mel_spectrogram[:1, :]),
                                  mel_spectrogram[:-1, :]), 0)
                decod_inp = sess.run(decod_inp_tensor)
                decod_inp = decod_inp[:, -N_MEL:]

                #padding time dimension
                padded_mel_spectrogram = np.zeros((MAX_MEL_TIME_LENGTH, mel_spectrogram.shape[1]))
                padded_mel_spectrogram[:mel_spectrogram.shape[0], :mel_spectrogram.shape[1]] = mel_spectrogram[:MAX_MEL_TIME_LENGTH]

                padded_decod_input = np.zeros((MAX_MEL_TIME_LENGTH, decod_inp.shape[1]))
                padded_decod_input[:decod_inp.shape[0], :decod_inp.shape[1]] = decod_inp[:MAX_MEL_TIME_LENGTH]

                padded_spectrogram = np.zeros((MAX_MAG_TIME_LENGTH, spectrogram.shape[1]))
                padded_spectrogram[:spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram[:MAX_MAG_TIME_LENGTH]

                tuple = (padded_spectrogram, padded_mel_spectrogram, padded_decod_input, list_of_char_id)
                toy_dataset.append(tuple)

                if counter < split1:
                    train1_dataset.append(tuple)
                elif split1 <= counter <= split2:
                    train2_dataset.append(tuple)
                else:
                    test_dataset.append(tuple)

                if counter % 1000 == 0:
                    print(f"Files done preprocesing: {counter}")

                tuple = (padded_mel_spectrogram, padded_spectrogram, padded_decod_input)

                #if counter % split1 == 0:
                #    break
            """

        with open("hopefully_gold.pickle", 'wb') as f:
            pickle.dump(tuple, f)

        #with open(ARGS.data_dir + "/vocab.pickle", 'wb') as f:
        #    pickle.dump(vocab, f)

        #with open(ARGS.data_dir + "/small.pickle", 'wb') as f:
        #    pickle.dump(toy_dataset, f)

        #with open(ARGS.data_dir + "/train1.pickle", 'wb') as f:
        #    pickle.dump(train1_dataset, f)

        #with open(ARGS.data_dir + "/train2.pickle", 'wb') as f:
        #    pickle.dump(train2_dataset, f)

        #with open(ARGS.data_dir + "/test.pickle", 'wb') as f:
        #    pickle.dump(test_dataset, f)

if __name__ == "__main__":
    main()