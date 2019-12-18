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
import librosa

from shared_util import *

import matplotlib.pyplot as plt

import sys

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

        # Dict written as ['filename' : [wav_file, text]]
        with open(ARGS.data_dir + "/all.raw.pickle", 'wb') as f:
            pickle.dump(id_to_wav_dict, f)

    else:

        with open(ARGS.data_dir + "/all.raw.pickle", 'rb') as f:
            id_to_wav_dict = pickle.load(f)

            # list of all of the training data
            all_padded_spectro = []
            all_mel_spectro = []
            all_padded_decod = []
            all_list_char_id = []

            counter = 0
            vocab = {} #for encoding text, character by character
            vocab['P'] = 0
            i = 1

            for key, value in id_to_wav_dict.items():
                counter += 1
                wav = value[0]
                sentence = value[1]

                mel_spectrogram, spectrogram = get_padded_spectrograms(wav)

                # Validate signal reconstruction
                predicted_spectro_item = spectrogram
                predicted_audio_item = convert_to_waveform(predicted_spectro_item)
                save(predicted_audio_item, 'test.wav')
                exit(0)

                list_of_existing_chars = list(set(sentence.lower().replace(" ", "")))
                for char in list_of_existing_chars:
                    if char not in vocab:
                        vocab[char] = i
                        i += 1

                list_of_char_id = [vocab[char] for char in list(sentence.lower().replace(" ", ""))]

                if len(list_of_char_id) < NB_CHARS_MAX:
                    for i in range(NB_CHARS_MAX - len(list_of_char_id)):
                        list_of_char_id.append(vocab['P'])
                else:
                    list_of_char_id = list_of_char_id[:NB_CHARS_MAX]

                sess = tf.Session()
                decod_inp_tensor = tf.concat((tf.zeros_like(mel_spectrogram[:1, :]),
                                  mel_spectrogram[:-1, :]), 0)
                decod_inp = sess.run(decod_inp_tensor)
                decod_inp = decod_inp[:, -N_MEL:]

                padded_mel_spectrogram = np.zeros((MAX_MEL_TIME_LENGTH, mel_spectrogram.shape[1]))
                padded_mel_spectrogram[:mel_spectrogram.shape[0], :mel_spectrogram.shape[1]] = mel_spectrogram[:MAX_MEL_TIME_LENGTH]

                padded_decod_input = np.zeros((MAX_MEL_TIME_LENGTH, decod_inp.shape[1]))
                padded_decod_input[:decod_inp.shape[0], :decod_inp.shape[1]] = decod_inp[:MAX_MEL_TIME_LENGTH]

                padded_spectrogram = np.zeros((MAX_MAG_TIME_LENGTH, spectrogram.shape[1]))
                padded_spectrogram[:spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram[:MAX_MAG_TIME_LENGTH]

                all_padded_spectro.append(padded_spectrogram)
                all_mel_spectro.append(padded_mel_spectrogram)
                all_padded_decod.append(padded_decod_input)
                all_list_char_id.append(list_of_char_id)

                if counter % 100 == 0:
                    print(f"datapoints saved {counter}")
                    dataset = (all_padded_spectro, all_mel_spectro, all_padded_decod, all_list_char_id)

                    with open(ARGS.data_dir + "/vocab.pickle", 'wb') as f:
                        pickle.dump(vocab, f)

                    with open(ARGS.data_dir + "/dataset.pickle", 'wb') as f:
                        pickle.dump(dataset, f)


        dataset = (all_padded_spectro, all_mel_spectro, all_padded_decod, all_list_char_id)

        with open(ARGS.data_dir + "/vocab.pickle", 'wb') as f:
            pickle.dump(vocab, f)

        with open(ARGS.data_dir + "/dataset.pickle", 'wb') as f:
            pickle.dump(dataset, f)

if __name__ == "__main__":
    main()