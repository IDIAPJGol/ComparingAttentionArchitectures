import tensorflow as tf
import random
import numpy as np


class layer_In(tf.keras.layers.Layer):
    def __init__(self, input_dims):
        super().__init__()
        self.masking_time_cat = tf.keras.layers.Masking(mask_value=-99,
                                                        input_shape=(input_dims[0][0], input_dims[0][1]))
        self.masking_time_num = tf.keras.layers.Masking(mask_value=-99,
                                                        input_shape=(input_dims[1][0], input_dims[1][1]))
        self.masking_missing = tf.keras.layers.Masking(mask_value=-100)
        self.normalizer = tf.keras.layers.Normalization(axis=2)
        self.concat = tf.keras.layers.Concatenate()

    @tf.function()
    def call(self, categoric, numeric, static):
        print("Categoric:", categoric.shape)
        print("Numeric:", numeric.shape)
        print("Static:", static.shape)
        categoric = self.masking_time_cat(categoric)
        numeric = self.masking_time_num(numeric)
        numeric = self.normalizer(numeric)
        concat = self.concat([categoric, numeric])
        concat = self.masking_missing(concat)
        static = self.masking_missing(static)
        return concat, static


class layer_Attention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.initializer = tf.keras.initializers.glorot_normal(seed=int(random.random()))

    @tf.function()
    def call(self, inputs, mask=None):
        # Trainable parameters
        hidden_size = inputs.shape[2]
        u_omega = self.initializer([hidden_size])
        v = tf.tanh(inputs)
        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        # Final output with tanh
        output = tf.tanh(output)
        print("Attention_out:", output.shape)
        print("Attention_weights:", alphas.shape)
        return output, alphas

    def compute_mask(self, inputs, mask=None):
        return mask


class Variable_Attention(tf.keras.layers.Layer):
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
        self.attention = layer_Attention()

    @tf.function()
    def call(self, inputs, mask=None):
        output = self.rnn(tf.expand_dims(inputs, axis=2))
        print("BiGRU_Vars:", output.shape)
        att_out, att_vars = self.attention(output)
        return att_out, att_vars

    def compute_output_shape(self, input_shape):
        if self.useBiRNN_vars:
            units = 2 * self.rnn_units
        else:
            units = self.rnn_units
        return ((None, units), (None, input_shape[1]))

    def compute_mask(self, inputs, mask=None):
        return mask


class Year_Attention(tf.keras.layers.Layer):
    def __init__(self, rnn_type, rnn_units, useBiRNN_years, dropout_rate):
        super().__init__()
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
        if useBiRNN_years:
            self.rnn = tf.keras.layers.Bidirectional(self.rnn)
        self.concat = tf.keras.layers.Concatenate()
        self.attention = layer_Attention()

    @tf.function()
    def call(self, inputs, vars_out):
        output = self.rnn(vars_out)  # output = self.rnn(vars_out) #
        print("BiGRU_Years:", output.shape)
        att_out, att_Years = self.attention(output)
        return att_out, att_Years


class Model(tf.keras.Model):
    def __init__(self, input_dims, rnn_type, rnn_units, useBiRNN_vars, useBiRNN_years, nn_layers, out_units,
                 dropout_rate, activation_func, use_tf_function=False):
        super().__init__()
        self.nn_layers = nn_layers
        self.inputs = layer_In(input_dims)
        self.varsAtt = Variable_Attention(rnn_type, rnn_units, useBiRNN_vars, dropout_rate)
        self.timeDist_varsAtt = tf.keras.layers.TimeDistributed(self.varsAtt)
        self.yearsAtt = Year_Attention(rnn_type, rnn_units, useBiRNN_years, dropout_rate)
        self.use_tf_function = use_tf_function
        self.fc0 = tf.keras.layers.Dense(int(rnn_units / 2))
        self.fc1 = tf.keras.layers.Dense(int(rnn_units / 4))
        self.fc2 = tf.keras.layers.Dense(int(rnn_units / 6))
        self.fc3 = tf.keras.layers.Dense(int(rnn_units / 8))
        self.fc_out = tf.keras.layers.Dense(out_units, activation=activation_func, name="output")

    @tf.function()
    def call(self, inputs, training):
        categoric, numeric, static = inputs
        inputs_merged, static = self.inputs(categoric, numeric, static)
        vars_out, varsAttention = self.timeDist_varsAtt(inputs_merged)
        print("Vars_out:", vars_out.shape)
        print("Vars_att:", varsAttention.shape)
        years_out, yearsAttention = self.yearsAtt(inputs_merged, vars_out)
        print("Years_out:", years_out.shape)
        print("Years_att:", yearsAttention.shape)
        print("Input logits:", tf.concat([years_out, static], axis=-1).shape)
        logits = self.fc0(tf.concat([years_out, static], axis=-1))
        if self.nn_layers > 1:
            logits = self.fc1(logits)
        if self.nn_layers > 2:
            logits = self.fc2(logits)
        if self.nn_layers > 3:
            logits = self.fc3(logits)
        logits = self.fc_out(logits)
        print("Logits:", logits.shape)
        return logits, varsAttention, yearsAttention

    @tf.function()
    def train_step(self, inputs):
        inputs_x, out_real = inputs
        categoric, numeric, static = inputs_x
        self.training = True
        with tf.GradientTape() as tape:
            out_pred, _, _ = self([categoric, numeric, static], training=True)
            step_loss = self.compiled_loss(out_real, out_pred)
        variables = self.trainable_variables
        gradients = tape.gradient(step_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.compiled_metrics.update_state(out_real, out_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function()
    def test_step(self, inputs):
        inputs_x, out_real = inputs
        categoric, numeric, static = inputs_x
        self.training = False
        out_pred, _, _ = self([categoric, numeric, static], training=False)
        self.compiled_loss(out_real, tf.squeeze(out_pred))
        self.compiled_metrics.update_state(out_real, tf.squeeze(out_pred))
        return {m.name: m.result() for m in self.metrics}


class MultiOutputModel(tf.keras.Model):
    def __init__(self, input_dims, rnn_type, rnn_units, useBiRNN_vars, useBiRNN_years, out_units, dropout_rate,
                 activation_func, use_tf_function=False):
        super().__init__()
        self.inputs = layer_In(input_dims)
        self.varsAtt = Variable_Attention(rnn_type, rnn_units, useBiRNN_vars, dropout_rate)
        self.timeDist_varsAtt = tf.keras.layers.TimeDistributed(self.varsAtt)
        self.yearsAtt = Year_Attention(rnn_type, rnn_units, useBiRNN_years, dropout_rate)
        self.use_tf_function = use_tf_function
        self.fc = tf.keras.layers.Dense(out_units, activation=activation_func, name="output")

    def call(self, inputs, training):
        categoric, numeric, static = inputs
        inputs_merged, static = self.inputs(categoric, numeric, static)
        vars_out, varsAttention = self.timeDist_varsAtt(inputs_merged)
        years_out, yearsAttention = self.yearsAtt(inputs_merged, vars_out)
        logitsI50 = self.fc(tf.concat([years_out, static], axis=-1))
        logitsJ44 = self.fc(tf.concat([years_out, static], axis=-1))
        # logitsS72 = self.fc(tf.concat([years_out, static], axis=-1))
        logitsI63 = self.fc(tf.concat([years_out, static], axis=-1))
        print("Logits:", logitsI50.shape)
        return logitsI50, logitsJ44, logitsI63, varsAttention, yearsAttention

    @tf.function
    def train_step(self, inputs):
        inputs_x, out_real = inputs
        categoric, numeric, static = inputs_x
        self.training = True

        with tf.GradientTape() as tape:
            out_pred_I50, out_pred_J44, out_pred_I63, _, _ = self([categoric, numeric, static], training=True)
            step_loss_I50 = self.compiled_loss(tf.squeeze(out_real[:, 0]), tf.squeeze(out_pred_I50))
            step_loss_J44 = self.compiled_loss(out_real[:, 1], tf.squeeze(out_pred_J44))
            # step_loss_S72 = self.compiled_loss(out_real[:,2], tf.squeeze(out_pred_S72))
            step_loss_I63 = self.compiled_loss(out_real[:, 2], tf.squeeze(out_pred_I63))
        variables = self.trainable_variables
        gradients = tape.gradient([step_loss_I50, step_loss_J44, step_loss_I63], variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.compiled_metrics.update_state(tf.squeeze([out_real[:, 0], out_real[:, 1], out_real[:, 2]]),
                                           tf.squeeze([out_pred_I50, out_pred_J44, out_pred_I63]))
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, inputs):
        inputs_x, out_real = inputs
        categoric, numeric, static = inputs_x
        self.training = False
        out_pred_I50, out_pred_J44, out_pred_I63, _, _ = self([categoric, numeric, static], training=False)
        self.compiled_metrics.update_state(tf.squeeze([out_real[:, 0], out_real[:, 1], out_real[:, 2]]),
                                           tf.squeeze([out_pred_I50, out_pred_J44, out_pred_I63]))
        return {m.name: m.result() for m in self.metrics}
