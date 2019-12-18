from keras.models import Model
import keras.backend as K
import keras.initializers as k_init
from keras.layers import (Input, Embedding, concatenate, RepeatVector, Dense,
                          Reshape, Conv1D, Dense, Activation, MaxPooling1D, Add,
                          Concatenate, Bidirectional, GRU, Dropout,
                          BatchNormalization, Lambda, Dot, Multiply)


'''
Tacotron model architecture for training TTS model. Parameter explanations are
provided in the share_util file.
'''
def tacotron(n_mels, r, k1, k2, nb_char_max,embedding_size, mel_time_length,mag_time_length, n_fft,vocabulary_size):

    # Encoder:
    input_encoder = Input(shape=(nb_char_max,))

    prenet_encoding = get_pre_net(Embedding(vocabulary_size,embedding_size,input_length=nb_char_max)(input_encoder))

    # Decoder-part1-Prenet:
    input_decoder = Input(shape=(None, n_mels))

    # Attention RNN
    attention_rnn_output = GRU(256)(get_pre_net(input_decoder))

    # Attention
    attention_context = attentionContextRetrieval(CBHGEncoder(prenet_encoding,k1),
                                              RepeatVector(nb_char_max)(attention_rnn_output))

    context_shapes = (int(attention_context.shape[1]), int(attention_context.shape[2]))
    attention_rnn_output_reshaped = Reshape(context_shapes)(attention_rnn_output)

    # Decoder-part2:
    input_of_decoder_rnn = concatenate([attention_context, attention_rnn_output_reshaped])
    input_of_decoder_rnn_projected = Dense(256)(input_of_decoder_rnn)

    output_of_decoder_rnn = decoderRNNOutput(input_of_decoder_rnn_projected)

    mel_hat = Reshape((mel_time_length, n_mels * r))(Dense(mel_time_length * n_mels * r)(output_of_decoder_rnn))

    # Define our lambda function for slicing
    def slice(x):
        return x[:, :, -n_mels:]

    mel_hat_last_frame = Lambda(slice)(mel_hat)
    post_process_output = CBHGPostProcess(mel_hat_last_frame,k2)

    z_hat = Reshape((mag_time_length, (1 + n_fft // 2)))(Dense(mag_time_length * (1 + n_fft // 2))(post_process_output))

    input_list = [input_encoder, input_decoder]
    output_list = [mel_hat, z_hat]

    return Model(inputs=input_list, outputs=output_list)

def get_pre_net(input_data):
    prenet = Dense(256)(input_data)
    prenet = Activation('relu')(prenet)
    prenet = Dropout(0.5)(prenet)
    prenet = Dense(128)(prenet)
    prenet = Activation('relu')(prenet)
    prenet = Dropout(0.5)(prenet)
    return prenet

def get_conv1dstack(kernel_sizes, input_data):
    convolution = Conv1D(filters=128, kernel_size=1,
                  strides=1, padding='same')(input_data)
    convolution = BatchNormalization()(convolution)
    convolution = Activation('relu')(convolution)

    for kernel_size in range(2, kernel_sizes + 1):
        convolution = Conv1D(filters=128, kernel_size=kernel_size,
                      strides=1, padding='same')(convolution)
        convolution = BatchNormalization()(convolution)
        convolution = Activation('relu')(convolution)

    return convolution


def HiOut(hiIn, nb_layers=4, activation="relu", bias=-3):
    dim = K.int_shape(hiIn)[-1]  # dimension must be the same
    initial_bias = k_init.Constant(bias)
    for n in range(nb_layers):
        H = Dense(units=dim, bias_initializer=initial_bias)(hiIn)
        H = Activation("sigmoid")(H)
        carry = Lambda(lambda x: 1.0 - x,output_shape=(dim,))(H)
        transform = Dense(units=dim)(hiIn)
        transform = Activation(activation)(transform)
        transformed = Multiply()([H, transform])
        carried = Multiply()([carry, hiIn])
        hi_out = Add()([transformed, carried])
    return hi_out


def CBHGEncoder(input_data, K_CBHG):
    conv1dbank = get_conv1dstack(K_CBHG, input_data)
    conv1dbank = MaxPooling1D(pool_size=2, strides=1,
                              padding='same')(conv1dbank)
    conv1dbank = Conv1D(filters=128, kernel_size=3,
                        strides=1, padding='same')(conv1dbank)
    conv1dbank = BatchNormalization()(conv1dbank)
    conv1dbank = Activation('relu')(conv1dbank)
    conv1dbank = Conv1D(filters=128, kernel_size=3,
                        strides=1, padding='same')(conv1dbank)
    conv1dbank = BatchNormalization()(conv1dbank)
    residual = Add()([input_data, conv1dbank])

    highway_net = HiOut(residual)

    CBHG_encoder = Bidirectional(GRU(128, return_sequences=True))(highway_net)

    return CBHG_encoder


def CBHGPostProcess(input_data, K_CBHG):
    conv1dbank = get_conv1dstack(K_CBHG, input_data)
    conv1dbank = MaxPooling1D(pool_size=2, strides=1,
                              padding='same')(conv1dbank)
    conv1dbank = Conv1D(filters=256, kernel_size=3,
                        strides=1, padding='same')(conv1dbank)
    # We love batch norm
    conv1dbank = BatchNormalization()(conv1dbank)
    conv1dbank = Activation('relu')(conv1dbank)
    conv1dbank = Conv1D(filters=80, kernel_size=3,
                        strides=1, padding='same')(conv1dbank)
    conv1dbank = BatchNormalization()(conv1dbank)
    residual = Add()([input_data, conv1dbank])

    highway_net = HiOut(residual)

    CBHG_post_proc = Bidirectional(GRU(128))(highway_net)

    return CBHG_post_proc


def decoderRNNOutput(input_data):
    stuff_to_add = [input_data, GRU(256, return_sequences=True)(input_data)]
    inp2 = Add()(stuff_to_add)
    stuff_to_add_2 = [inp2, GRU(256)(inp2)]
    return Add()(stuff_to_add_2)

def attentionContextRetrieval(encoder_output, attention_rnn_output):
    attention_input = Concatenate(axis=-1)([encoder_output,
                                            attention_rnn_output])
    e = Dense(10, activation="tanh")(attention_input)
    energies = Dense(1, activation="relu")(e)
    attention_weights = Activation('softmax')(energies)
    return Dot(axes=1)([attention_weights,
                           encoder_output])
