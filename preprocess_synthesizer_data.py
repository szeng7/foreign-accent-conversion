"""
Preprocessing wav files
"""
from scipy.io import wavfile
import numpy as np
import pickle
import argparse as ap
import os
import csv

import matplotlib.pyplot as plt

import librosa

import sys
#np.set_printoptions(threshold=sys.maxsize)


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

        for wav in os.listdir(ARGS.raw_data_dir + "/wavs"):
            if ".wav" in wav:
                id = wav.rstrip('.wav')
                path = ARGS.raw_data_dir + "/wavs/" + wav
                rate, data = wavfile.read(path)
                if id not in id_to_wav_dict:
                    id_to_wav_dict[id] = [data]

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
        """
        Add more on the exact preprocessing later
        when we know exactly what format we need

        """

        n_mels = 128
        n_fft = 2048
        hop_length = 512
        sr = 22050

        with open(ARGS.data_dir + "/all.raw.pickle", 'rb') as f:
            id_to_wav_dict = pickle.load(f)

            #40-40-20 split
            #train1-train2-test, splitting train since datasets would be too large to pickle
            split1 = 0.4*len(id_to_wav_dict.items())
            split2 = 0.8*len(id_to_wav_dict.items())

            train1_dataset = []
            train2_dataset = []
            test_dataset = []
            toy_dataset = []

            counter = 0

            for key, value in id_to_wav_dict.items():
                counter += 1
                wav = value[0]
                sentence = value[1]

                wav = wav.astype('float')
                wav /= wav.max()

                #wav to melspectrogram conversion
                S = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

                #wav to spectrogram conversion
                D = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
                D = np.maximum(1e-5, D)
                DB = librosa.amplitude_to_db(D)

                n_frame = S.shape[1]


                list_of_existing_chars = set(sentence.lower().replace(" ", ""))
                vocab = {}
                i = 0
                for char in list(list_of_existing_chars):
                    vocab[char] = i
                    i += 1

                list_of_char_id = [vocab[char] for char in list(sentence.lower().replace(" ", ""))]

                #(spectrogram_filename, mel_spectrogram_filename, n_frames, text)
                tuple = (DB, S, n_frame, list_of_char_id)
                toy_dataset.append(tuple)

                break

                #if counter < split1:
                #    train1_dataset.append(tuple)
                #elif split1 <= counter <= split2:
                #    train2_dataset.append(tuple)
                #else:
                #    test_dataset.append(tuple)

                #if counter % 1000 == 0:
                #    print(f"Files done preprocesing: {counter}")

                #if counter % split1 == 0:
                #    break


        with open(ARGS.data_dir + "/vocab.pickle", 'wb') as f:
            pickle.dump(vocab, f)

        with open(ARGS.data_dir + "/small.pickle", 'wb') as f:
            pickle.dump(toy_dataset, f)

        #with open(ARGS.data_dir + "/train1.pickle", 'wb') as f:
        #    pickle.dump(train1_dataset, f)

        #with open(ARGS.data_dir + "/train2.pickle", 'wb') as f:
        #    pickle.dump(train2_dataset, f)

        #with open(ARGS.data_dir + "/test.pickle", 'wb') as f:
        #    pickle.dump(test_dataset, f)

if __name__ == "__main__":
    main()