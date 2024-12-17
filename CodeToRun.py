import os
import pickle
import sklearn.metrics as sk_metrics
from sklearn.model_selection import RepeatedKFold

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from Funs import *
from Models.no_attention_model import NoAttentionModel
from Models.time_attention_model import TimeAttentionModel
from Models.domain_attention_model import DomainAttentionModel
from Models.hierarchical_attention_model import HierarchicalAttentionModel

with open('./SynData/TestTrainSets.pkl', 'rb') as f:
   [train_x, train_y, test_x, test_y] = pickle.load(f) # train_x, test_x have shape [patients, time, features]; train_y, test_y have shape [patients, label]
x_data = np.concatenate([train_x, test_x])
y_data = np.concatenate([train_y, test_y])
with open('./SynData/Columns.pkl', 'rb') as f:
    [cols, categoric_cols, numeric_cols, static_cols] = pickle.load(f) # cols is the list of feature names; categoric_cols, numeric_cols, and static_cols are the names of the features of each type.
train_x = [
    train_x[:, :, np.isin(cols, categoric_cols)],
    train_x[:, :, np.isin(cols, numeric_cols)],
    train_x[:, 0, np.isin(cols, static_cols)]
]
test_x = [
    test_x[:, :, np.isin(cols, categoric_cols)],
    test_x[:, :, np.isin(cols, numeric_cols)],
    test_x[:, 0, np.isin(cols, static_cols)]
]


# Parameters for the model
randomState = 13
epochs = 25
batch_size = 32
propBinary = 0.5
callback = [tf.keras.callbacks.EarlyStopping(monitor='auc', min_delta = 0.001, patience=6, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor = "auc", patience = 3)
           ]
input_dims = [(9, len(categoric_cols)), (9, len(numeric_cols)), (len(static_cols))] #categorics, numerics, statics
feature_num = 0
for dim in input_dims:
    if isinstance(dim, tuple):
        feature_num += dim[1]

# DISCRIMINABILITY: Parameters for the validation
threshold = 0.5
metrics = [
    sk_metrics.precision_score,
    sk_metrics.roc_auc_score,
    sk_metrics.average_precision_score,
    sk_metrics.recall_score,
    sk_metrics.cohen_kappa_score
]
n_splits = 5
rkf = RepeatedKFold(n_repeats=1, n_splits=n_splits, random_state=randomState)
keras_metrics = [tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.AUC(name="auc"),
                           tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
                           tf.keras.metrics.Recall(name="recall")]
res_test_list = []

for i_rkf, (train_idx, test_idx) in enumerate(rkf.split(x_data)):
    print("Iteration", i_rkf + 1, "of", n_splits)
    train_x_all = [
        x_data[train_idx][:, :, np.isin(cols, categoric_cols)],
        x_data[train_idx][:, :, np.isin(cols, numeric_cols)],
        x_data[train_idx][:, 0, np.isin(cols, static_cols)]
    ]

    test_x_all = [
        x_data[test_idx][:, :, np.isin(cols, categoric_cols)],
        x_data[test_idx][:, :, np.isin(cols, numeric_cols)],
        x_data[test_idx][:, 0, np.isin(cols, static_cols)]
    ]

    ## APPROACH 0
    model_name = "Approach_0_WithoutAttention"
    print("Model ->", model_name)
    model = NoAttentionModel(input_dims=input_dims, rnn_type = "GRU", rnn_units = 128, useBiRNN_years=True, out_units=1, dropout_rate=0.10, activation_func = "sigmoid", layer_inp_emb=False)
    model.compile(loss=two_loss_func,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=keras_metrics)
    history = model.fit(train_x_all, y_data[train_idx],
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callback)

    test_y_pred = model.predict(test_x_all)
    train_y_pred = model.predict(train_x_all)
    res_test_bootstrap = ci_auto(metrics, y_data[test_idx], test_y_pred > threshold, nboots=100, Sample=model_name, CV = str(i_rkf))
    res_test_list.append(res_test_bootstrap)

    ## APPROACH 1
    model_name = "Approach_1_TimeAttention"
    print("Model ->", model_name)
    # To change between attention layers, tune linearRegression = False, finalTanh = True
    model = TimeAttentionModel(input_dims=input_dims, rnn_type="GRU", rnn_units=128, useBiRNN_years=True,
                             out_units=1, dropout_rate=0.10, activation_func="sigmoid", linearRegression = False, finalTanh = True,
                             layer_inp_emb=False)
    model.compile(loss=two_loss_func,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=keras_metrics)
    history = model.fit(train_x_all, y_data[train_idx],
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callback)

    test_y_pred,time_attention_map = model.predict(test_x_all)
    res_test_bootstrap = ci_auto(metrics, y_data[test_idx], test_y_pred > threshold, nboots=100, Sample=model_name, CV = str(i_rkf))
    res_test_list.append(res_test_bootstrap)

    ## APPROACH 2
    model_name = "Approach_2_DomainAttention"
    print("Model ->", model_name)
    # To change between feature processing approaches, modify nn_type_vars: ["None", "Dense", "GRU", "LSTM", "CNN", "CNN_decoder"]
    model = DomainAttentionModel(input_dims=input_dims, nn_type_vars = "GRU", rnn_type_time = "GRU", nn_units = 128, useBiRNN_years=True, out_units=1, dropout_rate=0.10, activation_func = "sigmoid", linearRegression = False, finalTanh = True)
    model.compile(loss=two_loss_func,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=keras_metrics)
    history = model.fit(train_x_all, y_data[train_idx],
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callback)

    test_y_pred,_,_ = model.predict(test_x_all)
    res_test_bootstrap = ci_auto(metrics, y_data[test_idx], test_y_pred > threshold, nboots=100, Sample=model_name, CV = str(i_rkf))
    res_test_list.append(res_test_bootstrap)

    ## APPROACH 3
    model_name = "Approach_3_HierarchicalAttention"
    print("Model ->", model_name)
    # To change between feature processing approaches, modify nn_type_vars: ["None", "Dense", "GRU", "LSTM", "CNN", "CNN_decoder"]
    model = HierarchicalAttentionModel(input_dims=input_dims, nn_type_vars = "GRU", rnn_type_time = "GRU", nn_units = 128, useBiRNN_years=True, out_units=1, dropout_rate=0.10, linearRegression = False, finalTanh = True, activation_func = "sigmoid")
    model.compile(loss=two_loss_func,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=keras_metrics)
    history = model.fit(train_x_all, y_data[train_idx],
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callback)

    test_y_pred,_,_ = model.predict(test_x_all)
    sample_name = "CV_" + model_name + "_cv_" + str(i_rkf)
    res_test_bootstrap = ci_auto(metrics, y_data[test_idx], test_y_pred > threshold, nboots=100, Sample=model_name, CV = str(i_rkf))
    res_test_list.append(res_test_bootstrap)

df_all_cv = pd.concat(res_test_list)
df_all_cv.to_csv("Metrics.csv")
df_all_cv.groupby([df_all_cv.index.get_level_values(0),'Sample']).agg({
        'Observed': ["mean", "std"]
    }).to_csv("AggMetrics.csv")

# ATTENTION MAPS: For the clinical plausibility analysis, we trained the models with all the training data and obtained the attention maps for all the test set.

## APPROACH 1
model_name = "Approach_1_TimeAttention"
print("Model ->", model_name)
# To change between attention layers, tune linearRegression = False, finalTanh = True
model = TimeAttentionModel(input_dims=input_dims, rnn_type="GRU", rnn_units=128, useBiRNN_years=True,
                             out_units=1, dropout_rate=0.10, activation_func="sigmoid", linearRegression = False, finalTanh = True,
                             layer_inp_emb=False)
model.compile(loss=two_loss_func,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=keras_metrics)
model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callback)

_, time_time = model.predict(test_x)

## APPROACH 2
model_name = "Approach_2_DomainAttention"
print("Model ->", model_name)
# To change between feature processing approaches, modify nn_type_vars: ["None", "Dense", "GRU", "LSTM", "CNN", "CNN_decoder"]
model = DomainAttentionModel(input_dims=input_dims, nn_type_vars="GRU", rnn_type_time="GRU", nn_units=128,
                             useBiRNN_years=True, out_units=1, dropout_rate=0.10, activation_func="sigmoid",
                             linearRegression=False, finalTanh=True)
model.compile(loss=two_loss_func,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=keras_metrics)
model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callback)

_, domain_vars, domain_time = model.predict(test_x)

## APPROACH 3
model_name = "Approach_3_HierarchicalAttention"
print("Model ->", model_name)
# To change between feature processing approaches, modify nn_type_vars: ["None", "Dense", "GRU", "LSTM", "CNN", "CNN_decoder"]
model = HierarchicalAttentionModel(input_dims=input_dims, nn_type_vars="GRU", rnn_type_time="GRU", nn_units=128,
                                   useBiRNN_years=True, out_units=1, dropout_rate=0.10, linearRegression=False,
                                   finalTanh=True, activation_func="sigmoid")
model.compile(loss=two_loss_func,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=keras_metrics)
model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callback)

_, hierarchical_vars, hierarchical_time = model.predict(test_x)

with open('AttentionMaps.pkl', 'wb') as f:
    pickle.dump([time_time, domain_vars, domain_time, hierarchical_vars, hierarchical_time], f)

#Visualization was performed using ggplot in R