import tensorflow as tf

from Models.global_layers import LayerInput, LayerRNN, LayerAttention, LayerType, LayerNNAttention

class HierarchicalAttentionModel(tf.keras.Model):
    def __init__(self, input_dims, nn_type_vars, rnn_type_time, nn_units,
                 useBiRNN_years, out_units, dropout_rate, activation_func, linearRegression, finalTanh,
                 features_number=0, layer_inp_emb=False,
                 use_tf_function=False):
        super().__init__()
        self.inputs = LayerInput(input_dims, use_embedding=layer_inp_emb)
        self.nn_type_vars = nn_type_vars
        if self.nn_type_vars != None:
            self.varsNN = LayerType(nn_type_vars, nn_units, features_number, dropout_rate)
        else:
            self.varsNN = self.nn_type_vars
        self.varsAtt = LayerAttention(linearRegression, finalTanh)
        self.varsNN_Att = LayerNNAttention(self.varsNN, self.varsAtt, nn_units)
        self.timeDist_varsAtt = tf.keras.layers.TimeDistributed(self.varsNN_Att)
        self.yearsRNN = LayerRNN(rnn_type_time, nn_units, useBiRNN_years, dropout_rate)
        self.yearsAtt = LayerAttention(linearRegression, finalTanh)
        self.use_tf_function = use_tf_function
        self.fc0 = tf.keras.layers.Dense(int(nn_units/2))
        self.fc_out = tf.keras.layers.Dense(out_units, activation=activation_func, name="output")

    @tf.function()
    def call(self, inputs, training):
        categoric, numeric, static = inputs
        inputs_merged, static = self.inputs(categoric, numeric, static)
        vars_out, varsAttention = self.timeDist_varsAtt(inputs_merged)
        print("Vars_out:", vars_out.shape)
        print("Vars_att:", varsAttention.shape)
        years_out = self.yearsRNN(vars_out)
        years_out, yearsAttention = self.yearsAtt(years_out)
        print("Years_out:", years_out.shape)
        print("Years_att:", yearsAttention.shape)
        print("Input logits:", tf.concat([years_out, static], axis=-1).shape)
        logits = self.fc0(tf.concat([years_out, static], axis=-1))
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
