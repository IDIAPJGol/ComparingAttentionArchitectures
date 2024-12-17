import tensorflow as tf
import random


class LayerInput(tf.keras.layers.Layer):
    def __init__(self, input_dims, use_embedding=False):
        super().__init__()
        self.use_embedding = use_embedding

        self.masking_time_cat = tf.keras.layers.Masking(mask_value=-99,
                                                        input_shape=(input_dims[0][0], input_dims[0][1]))
        self.masking_time_num = tf.keras.layers.Masking(mask_value=-99,
                                                        input_shape=(input_dims[1][0], input_dims[1][1]))
        self.masking_missing = tf.keras.layers.Masking(mask_value=-100)
        self.normalizer = tf.keras.layers.Normalization(axis=2)
        self.concat = tf.keras.layers.Concatenate()
        if self.use_embedding:
            self.flatten_cat = tf.keras.layers.Flatten()
            self.embedding_cat = tf.keras.layers.Embedding(
                input_dim=int(input_dims[0][0] * input_dims[0][1]),
                output_dim=10
            )
            self.reshape_cat = tf.keras.layers.Reshape(
                (input_dims[0][0], int((10 * input_dims[0][0] * input_dims[0][1]) / input_dims[0][0]))
            )

    @tf.function()
    def call(self, categoric, numeric, static):
        print("Categoric:", categoric.shape)
        print("Numeric:", numeric.shape)
        print("Static:", static.shape)

        categoric = self.masking_time_cat(categoric)
        if self.use_embedding:
            categoric = tf.add(categoric, 100)
            categoric = self.flatten_cat(categoric)
            categoric = self.embedding_cat(categoric)
            categoric = self.reshape_cat(categoric)

        numeric = self.masking_time_num(numeric)
        numeric = self.normalizer(numeric)
        concat = self.concat([categoric, numeric])
        concat = self.masking_missing(concat)
        static = self.masking_missing(static)

        return concat, static


class LayerRNN(tf.keras.layers.Layer):
    def __init__(self, rnn_type, rnn_units, useBiRNN_vars, dropout_rate):
        super().__init__()
        self.rnn_units = rnn_units
        self.useBiRNN_vars = useBiRNN_vars
        if rnn_type == "GRU":
            self.rnn = tf.keras.layers.GRU(rnn_units,
                                           return_sequences=True,
                                           recurrent_initializer='glorot_uniform',
                                           recurrent_dropout=dropout_rate)
        elif rnn_type == "LSTM":
            self.rnn = tf.keras.layers.LSTM(rnn_units,
                                            return_sequences=True,
                                            recurrent_initializer='glorot_uniform',
                                            recurrent_dropout=dropout_rate)
        if useBiRNN_vars:
            self.rnn = tf.keras.layers.Bidirectional(self.rnn)

    @tf.function()
    def call(self, inputs, mask=None):
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=2)
        output = self.rnn(inputs)
        print("RNN_out:", output.shape)
        return output

    def compute_mask(self, inputs, mask=None):
        return mask


class LayerAttention(tf.keras.layers.Layer):
    def __init__(self, linearRegression, finalTanh):
        super().__init__()
        self.initializer = tf.keras.initializers.glorot_normal(seed=int(random.random()))
        self.zeros_initializer = tf.keras.initializers.Zeros()
        self.finalTanh = finalTanh
        self.linearRegression = linearRegression

    @tf.function()
    def call(self, inputs, mask=None):
        # Trainable parameters
        W = self.initializer([inputs.shape[2]])
        if self.linearRegression:
            b = self.zeros_initializer([inputs.shape[1]])
            vu = tf.tensordot(inputs, W, axes=1)
            vu = tf.tanh(tf.add(vu, b))
        else:
            vu = tf.tanh(inputs)
            vu = tf.tensordot(vu, W, axes=1)

        alphas = tf.nn.softmax(vu, name='alphas')
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if self.finalTanh:  # Final output with tanh
            output = tf.tanh(output)
        return output, alphas

    def compute_mask(self, inputs, mask=None):
        return mask


def conv_model(nn_units, num_layers=2, dropout_rate_cnn=0.05, use_decoder=False):
    conv_layers_list = []

    # ###############
    # ### Encoder ###
    # ###############

    for i in range(num_layers):
        conv_layers_list += [
            tf.keras.layers.Conv1D(nn_units, kernel_size=5, padding="same"),
            tf.keras.layers.BatchNormalization()
        ]

    conv_layers_list += [tf.keras.layers.Dropout(dropout_rate_cnn)]

    if use_decoder:
        conv_layers_list += [tf.keras.layers.MaxPool1D(pool_size=2)]

        for i in range(num_layers):
            conv_layers_list += [
                tf.keras.layers.Conv1D(2 * nn_units, kernel_size=5, padding="same"),
                tf.keras.layers.BatchNormalization()
            ]

        conv_layers_list += [tf.keras.layers.Dropout(dropout_rate_cnn)]

        # ###############
        # ### Decoder ###
        # ###############

        conv_layers_list += [
            tf.keras.layers.UpSampling1D(size=2),
        ]
        for i in range(num_layers):
            conv_layers_list += [
                tf.keras.layers.Conv1D(nn_units, kernel_size=5, padding="same"),
                tf.keras.layers.BatchNormalization()
            ]

        conv_layers_list += [tf.keras.layers.Dropout(dropout_rate_cnn)]

    return conv_layers_list


class LayerType(tf.keras.layers.Layer):
    def __init__(self, layer_type, nn_units, useBiRNN_vars, dropout_rate, features_number=0):
        super().__init__()
        self.nn_units = nn_units
        self.useBiRNN_vars = useBiRNN_vars

        use_decoder = True if layer_type == "CNN_decoder" else False
        self.layer_type = "CNN" if layer_type == "CNN_decoder" else layer_type

        self.batch = tf.keras.layers.BatchNormalization()

        if self.layer_type == "GRU":
            self.rnn = tf.keras.layers.GRU(self.nn_units,
                                           return_sequences=True,
                                           recurrent_initializer='glorot_uniform',
                                           recurrent_dropout=dropout_rate)
        elif self.layer_type == "LSTM":
            self.rnn = tf.keras.layers.LSTM(self.nn_units,
                                            return_sequences=True,
                                            recurrent_initializer='glorot_uniform',
                                            recurrent_dropout=dropout_rate)
        elif self.layer_type == "CNN":
            self.conv_layers = conv_model(self.nn_units, use_decoder=use_decoder)

        elif self.layer_type == "Dense":
            self.dense = tf.keras.layers.Dense(self.nn_units, activation='relu', input_shape=(features_number, 1))

        if useBiRNN_vars and (self.layer_type == "LSTM" or self.layer_type == "GRU"):
            self.rnn = tf.keras.layers.Bidirectional(self.rnn)

    @tf.function()
    def apply_layer_type(self, inputs):

        if self.layer_type == "GRU":
            output = self.rnn(inputs)

        elif self.layer_type == "LSTM":
            output = self.rnn(inputs)

        elif self.layer_type == "CNN":
            output = inputs
            for layer in self.conv_layers:
                output = layer(output)

        elif self.layer_type == "Dense":
            output = self.dense(inputs)
            output = self.batch(output)

        elif self.layer_type == "Embedding":
            output = self.rescale(inputs)
            output = self.embedding(output)

        else:
            output = inputs

        return output

    @tf.function()
    def call(self, inputs, mask=None):
        if len(inputs.shape) == 2 and self.layer_type != "Embedding":
            inputs = tf.expand_dims(inputs, axis=2)
        output = self.apply_layer_type(inputs)
        # output = self.rnn(inputs)
        print("NN_out:", output.shape)
        return output

    def compute_mask(self, inputs, mask=None):
        return mask


class LayerNNAttention(tf.keras.layers.Layer):
    def __init__(self, layer_rnn, layer_att, rnn_units):
        super().__init__()
        self.rnn = layer_rnn
        self.att = layer_att
        self.rnn_units = rnn_units

    @tf.function()
    def call(self, inputs, mask=None):
        print("Inputs_NNAttention", inputs.shape)
        if self.rnn == None:
            rnn_output = inputs
        else:
            rnn_output = self.rnn(inputs)
        if len(rnn_output.shape) == 2:
            rnn_output = tf.expand_dims(rnn_output, axis=2)
        print("OutputsRNN_NNAttention", rnn_output.shape)
        att_output, att_alphas = self.att(rnn_output)
        print("OutputsAtt_NNAttention", rnn_output.shape)
        return att_output, att_alphas

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        units = self.rnn_units
        return ((None, units), (None, input_shape[1]))