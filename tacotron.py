from keras.models import Model
import keras.backend as K
import keras.initializers as k_init
from keras.layers import (Input, Embedding, concatenate, RepeatVector, Dense,
                          Reshape, Conv1D, Dense, Activation, MaxPooling1D, Add,
                          Concatenate, Bidirectional, GRU, Dropout,
                          BatchNormalization, Lambda, Dot, Multiply)


def tacotron(n_mels, r, k1, k2, nb_char_max,
                       embedding_size, mel_time_length,
                       mag_time_length, n_fft,
                       vocabulary_size):
    # Encoder:
    input_encoder = Input(shape=(nb_char_max,))

    embedded = Embedding(input_dim=vocabulary_size,
                         output_dim=embedding_size,
                         input_length=nb_char_max)(input_encoder)
    prenet_encoding = get_pre_net(embedded)

    cbhg_encoding = get_CBHG_encoder(prenet_encoding,
                                     k1)

    # Decoder-part1-Prenet:
    input_decoder = Input(shape=(None, n_mels))
    prenet_decoding = get_pre_net(input_decoder)
    attention_rnn_output = get_attention_RNN()(prenet_decoding)

    # Attention
    attention_rnn_output_repeated = RepeatVector(nb_char_max)(attention_rnn_output)

    attention_context = get_attention_context(cbhg_encoding, attention_rnn_output_repeated)

    context_shapes = (int(attention_context.shape[1]), int(attention_context.shape[2]))
    attention_rnn_output_reshaped = Reshape(context_shapes)(attention_rnn_output)

    # Decoder-part2:
    input_of_decoder_rnn = concatenate(
        [attention_context, attention_rnn_output_reshaped])
    input_of_decoder_rnn_projected = Dense(256)(input_of_decoder_rnn)

    output_of_decoder_rnn = get_decoder_RNN_output(
        input_of_decoder_rnn_projected)

    # mel_hat=TimeDistributed(Dense(n_mels*r))(output_of_decoder_rnn)
    mel_hat = Dense(mel_time_length * n_mels * r)(output_of_decoder_rnn)
    mel_hat_ = Reshape((mel_time_length, n_mels * r))(mel_hat)

    def slice(x):
        return x[:, :, -n_mels:]

    mel_hat_last_frame = Lambda(slice)(mel_hat_)
    post_process_output = get_CBHG_post_process(mel_hat_last_frame,
                                                k2)

    z_hat = Dense(mag_time_length * (1 + n_fft // 2))(post_process_output)
    z_hat_ = Reshape((mag_time_length, (1 + n_fft // 2)))(z_hat)

    input_list = [input_encoder, input_decoder]
    output_list = [mel_hat_, z_hat_]

    model = Model(inputs=input_list, outputs=output_list)
    return model

def get_pre_net(input_data):
    prenet = Dense(256)(input_data)
    prenet = Activation('relu')(prenet)
    prenet = Dropout(0.5)(prenet)
    prenet = Dense(128)(prenet)
    prenet = Activation('relu')(prenet)
    prenet = Dropout(0.5)(prenet)

    return prenet

def get_conv1dbank(K_, input_data):
    conv = Conv1D(filters=128, kernel_size=1,
                  strides=1, padding='same')(input_data)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    for k_ in range(2, K_ + 1):
        conv = Conv1D(filters=128, kernel_size=k_,
                      strides=1, padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

    return conv


def get_highway_output(highway_input, nb_layers, activation="tanh", bias=-3):
    dim = K.int_shape(highway_input)[-1]  # dimension must be the same
    initial_bias = k_init.Constant(bias)
    for n in range(nb_layers):
        H = Dense(units=dim, bias_initializer=initial_bias)(highway_input)
        H = Activation("sigmoid")(H)
        carry_gate = Lambda(lambda x: 1.0 - x,
                            output_shape=(dim,))(H)
        transform_gate = Dense(units=dim)(highway_input)
        transform_gate = Activation(activation)(transform_gate)
        transformed = Multiply()([H, transform_gate])
        carried = Multiply()([carry_gate, highway_input])
        highway_output = Add()([transformed, carried])
    return highway_output


def get_CBHG_encoder(input_data, K_CBHG):
    conv1dbank = get_conv1dbank(K_CBHG, input_data)
    conv1dbank = MaxPooling1D(pool_size=2, strides=1,
                              padding='same')(conv1dbank)
    conv1dbank = Conv1D(filters=128, kernel_size=3,
                        strides=1, padding='same')(conv1dbank)
    conv1dbank = BatchNormalization()(conv1dbank)
    conv1dbank = Activation('relu')(conv1dbank)
    conv1dbank = Conv1D(filters=128, kernel_size=3,
                        strides=1, padding='same')(conv1dbank)
    conv1dbank = BatchNormalization()(conv1dbank)

    # residual learning helps training deep networks
    # (https://arxiv.org/pdf/1512.03385.pdf)
    residual = Add()([input_data, conv1dbank])

    highway_net = get_highway_output(residual, 4, activation='relu')

    CBHG_encoder = Bidirectional(GRU(128, return_sequences=True))(highway_net)

    return CBHG_encoder


def get_CBHG_post_process(input_data, K_CBHG):
    conv1dbank = get_conv1dbank(K_CBHG, input_data)
    conv1dbank = MaxPooling1D(pool_size=2, strides=1,
                              padding='same')(conv1dbank)
    conv1dbank = Conv1D(filters=256, kernel_size=3,
                        strides=1, padding='same')(conv1dbank)
    conv1dbank = BatchNormalization()(conv1dbank)
    conv1dbank = Activation('relu')(conv1dbank)
    conv1dbank = Conv1D(filters=80, kernel_size=3,
                        strides=1, padding='same')(conv1dbank)
    conv1dbank = BatchNormalization()(conv1dbank)

    # residual learning helps training deep networks
    # (https://arxiv.org/pdf/1512.03385.pdf)
    residual = Add()([input_data, conv1dbank])

    highway_net = get_highway_output(residual, 4, activation='relu')

    CBHG_post_proc = Bidirectional(GRU(128))(highway_net)

    return CBHG_post_proc

# see https://arxiv.org/pdf/1609.08144.pdf (stack of GRUs with vertical
# residual connections)


def get_decoder_RNN_output(input_data):

    rnn1 = GRU(256, return_sequences=True)(input_data)

    inp2 = Add()([input_data, rnn1])
    rnn2 = GRU(256)(inp2)

    decoder_rnn = Add()([inp2, rnn2])

    return decoder_rnn


def get_attention_RNN():
    return GRU(256)


def get_attention_context(encoder_output, attention_rnn_output):
    attention_input = Concatenate(axis=-1)([encoder_output,
                                            attention_rnn_output])
    e = Dense(10, activation="tanh")(attention_input)
    energies = Dense(1, activation="relu")(e)
    attention_weights = Activation('softmax')(energies)
    context = Dot(axes=1)([attention_weights,
                           encoder_output])

    return context
