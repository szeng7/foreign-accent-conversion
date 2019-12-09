"""
Preprocessing wav files
"""
from scipy.io import wavfile
import numpy as np
import pickle
import argparse as ap
import os
import matplotlib.pyplot as plt
import csv


def main():

    p = ap.ArgumentParser()

    p.add_argument('--raw-data-dir', required=False, \
        help='directory of raw wav files for training')
    p.add_argument('--data-dir', required=False, \
        help='data directory of wav files')
    p.add_argument('--model-dir', required=False, \
        help='data directory of model files')
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
                try:
                    normalized_transcription = row[2]
                except:
                    normalized_transcription = row[1]
                if id in id_to_wav_dict:
                    id_to_wav_dict[id].append(normalized_transcription)

        with open(ARGS.data_dir + "/all.raw.pickle", 'wb') as f:
            pickle.dump(id_to_wav_dict, f)

    else:
        """
        Add more on the exact preprocessing later
        when we know exactly what format we need

        """
        with open(ARGS.data_dir + "/all.raw.pickle", 'rb') as f:
            id_to_wav_dict = pickle.load(f)

            train_split = 0.8 #80/20 split

            toy_x = []
            toy_y = []

            counter = 0

            for key, value in id_to_wav_dict.items():
                counter += 1
                wav = value[0]
                sentence = value[1]
                toy_x.append(sentence)
                toy_y.append(wav)
                if counter == 20:
                    break

        with open(ARGS.data_dir + "/small.pickle", 'wb') as f:
            pickle.dump([toy_x, toy_y], f)

if __name__ == "__main__":
    main()