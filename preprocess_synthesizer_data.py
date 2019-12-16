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
from processing.proc_audio import from_spectro_to_waveform

import matplotlib.pyplot as plt
from hparams import *

import sys
#np.set_printoptions(threshold=sys.maxsize)

def get_spectros(wav, preemphasis, n_fft,
                 hop_length, win_length,
                 sampling_rate, n_mel,
                 ref_db, max_db):


    wav = wav.astype('float')
    #wav /= wav.max()
    waveform = wav

    waveform, _ = librosa.effects.trim(waveform)

    # use pre-emphasis to filter out lower frequencies
    waveform = np.append(waveform[0],
                         waveform[1:] - preemphasis * waveform[:-1])

    # compute the stft
    stft_matrix = librosa.stft(y=waveform,
                               n_fft=n_fft,
                               hop_length=hop_length,
                               win_length=win_length)

    # compute magnitude and mel spectrograms
    spectro = np.abs(stft_matrix)

    mel_transform_matrix = librosa.filters.mel(sampling_rate,
                                               n_fft,
                                               n_mel)
    mel_spectro = np.dot(mel_transform_matrix,
                         spectro)

    # Use the decidel scale
    mel_spectro = 20 * np.log10(np.maximum(1e-5, mel_spectro))
    spectro = 20 * np.log10(np.maximum(1e-5, spectro))

    # Normalise the spectrograms
    mel_spectro = np.clip((mel_spectro - ref_db + max_db) / max_db, 1e-8, 1)
    spectro = np.clip((spectro - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose the spectrograms to have the time as first dimension
    # and the frequency as second dimension
    mel_spectro = mel_spectro.T.astype(np.float32)
    spectro = spectro.T.astype(np.float32)

    return mel_spectro, spectro

def get_padded_spectros(wav, r, preemphasis, n_fft,
                        hop_length, win_length, sampling_rate,
                        n_mel, ref_db, max_db):
    mel_spectro, spectro = get_spectros(wav, preemphasis, n_fft,
                                        hop_length, win_length, sampling_rate,
                                        n_mel, ref_db, max_db)
    t = mel_spectro.shape[0]
    nb_paddings = r - (t % r) if t % r != 0 else 0  # for reduction
    mel_spectro = np.pad(mel_spectro,
                        [[0, nb_paddings], [0, 0]],
                        mode="constant")
    spectro = np.pad(spectro,
                    [[0, nb_paddings], [0, 0]],
                    mode="constant")
    return wav, mel_spectro.reshape((-1, n_mel * r)), spectro

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

                # forces loading of library here?
                import librosa
                # librosa outperforms scipy wavfile by doing normalization as well
                data, rate = librosa.core.load(path, sr=SAMPLING_RATE)
                if id not in id_to_wav_dict:
                    id_to_wav_dict[id] = [data]

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

                wav, mel_spectro, spectro = get_padded_spectros(wav, r,
                                                      PREEMPHASIS, N_FFT,
                                                      HOP_LENGTH, WIN_LENGTH,
                                                      SAMPLING_RATE,
                                                      N_MEL, REF_DB,
                                                      MAX_DB)

                # Validate signal reconstruction
                # predicted_spectro_item = spectro
                # predicted_audio_item = from_spectro_to_waveform(predicted_spectro_item, N_FFT,
                #                                                 HOP_LENGTH, WIN_LENGTH,
                #                                                 N_ITER, WINDOW_TYPE,
                #                                                 MAX_DB, REF_DB, PREEMPHASIS)
                # save_wav(predicted_audio_item,'temp.wav',sr=SAMPLING_RATE)
                # exit(0)
                
                """
                #wav to melspectrogram conversion
                S = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

                #wav to spectrogram conversion
                D = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
                D = np.maximum(1e-5, D)
                DB = librosa.amplitude_to_db(D)

                n_frame = S.shape[1]
                """

                list_of_existing_chars = set(sentence.lower().replace(" ", ""))
                vocab = {}
                vocab['P'] = 0
                i = 1
                for char in list(list_of_existing_chars):
                    vocab[char] = i
                    i += 1

                list_of_char_id = [vocab[char] for char in list(sentence.lower().replace(" ", ""))]

                if len(list_of_char_id) < 200:
                    for i in range(200 - len(list_of_char_id)):
                        list_of_char_id.append(vocab['P'])

                sess = tf.Session()
                decod_inp_tensor = tf.concat((tf.zeros_like(mel_spectro[:1, :]),
                                  mel_spectro[:-1, :]), 0)
                decod_inp = sess.run(decod_inp_tensor)
                decod_inp = decod_inp[:, -N_MEL:]

                print(decod_inp.shape)
                print(mel_spectro.shape)
                print(spectro.shape)

                # Padding of the temporal dimension
                dim0_mel_spectro = mel_spectro.shape[0]
                dim1_mel_spectro = mel_spectro.shape[1]
                padded_mel_spectro = np.zeros((MAX_MEL_TIME_LENGTH, dim1_mel_spectro))
                padded_mel_spectro[:dim0_mel_spectro, :dim1_mel_spectro] = mel_spectro[:MAX_MEL_TIME_LENGTH]

                dim0_decod_inp = decod_inp.shape[0]
                dim1_decod_inp = decod_inp.shape[1]
                padded_decod_input = np.zeros((MAX_MEL_TIME_LENGTH, dim1_decod_inp))
                padded_decod_input[:dim0_decod_inp, :dim1_decod_inp] = decod_inp[:MAX_MEL_TIME_LENGTH]

                dim0_spectro = spectro.shape[0]
                dim1_spectro = spectro.shape[1]
                padded_spectro = np.zeros((MAX_MAG_TIME_LENGTH, dim1_spectro))
                padded_spectro[:dim0_spectro, :dim1_spectro] = spectro[:MAX_MAG_TIME_LENGTH]

                #mel_spectro_data.append(padded_mel_spectro)
                #spectro_data.append(padded_spectro)
                #decoder_input.append(padded_decod_input)

                #(spectrogram_filename, mel_spectrogram_filename, n_frames, text)
                tuple = (padded_spectro, padded_mel_spectro, padded_decod_input, list_of_char_id)
                toy_dataset.append(tuple)

                print(tuple)

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


def save_wav(wav, path, sr):
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	#proposed by @dsmiller
	wavfile.write(path, sr, wav.astype(np.int16))


if __name__ == "__main__":
    main()