from tensorflow.keras.losses import binary_crossentropy
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime
from random import sample
from pad_sequences import pad_sequences_multi
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def dice_coef(y_true, y_pred, smooth=1):
    import tensorflow as tf
    import keras.backend as K
    y_true = tf.cast(y_true, dtype='float32')
    y_pred = tf.cast(y_pred, dtype='float32')
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def data_split_CSV(data, id_col, labels, yearsToTarget, validation_split):
    test_ids = pd.read_csv("./R/IDs_Test_" + str(yearsToTarget) + "year.csv")
    test_ids = test_ids["x"].to_list()
    train_ids = pd.read_csv("./R/IDs_Train_" + str(yearsToTarget) + "year.csv")
    train_ids = train_ids["x"].to_list()
    print(len(train_ids))
    if validation_split > 0:
        validation_ids = sample(train_ids, int(validation_split * len(train_ids)))
        train_ids = list(set(validation_ids) ^ set(train_ids))
    data = np.array(data, dtype="float32")
    test_x = data[id_col.isin(test_ids), :, :]
    train_x = data[id_col.isin(train_ids), :, :]
    test_y = labels[id_col.isin(test_ids)]
    train_y = labels[id_col.isin(train_ids)]

    if validation_split > 0:
        val_x = data[id_col.isin(validation_ids), :, :]
        val_y = labels[id_col.isin(validation_ids)]
    if len(np.unique(train_y)) > 2:
        train_y = pd.get_dummies(train_y)
        test_y = pd.get_dummies(test_y)
        if validation_split > 0:
            val_y = pd.get_dummies(val_y)
    else:
        train_y = train_y.to_numpy(dtype="float32")
        test_y = test_y.to_numpy(dtype="float32")
        if validation_split > 0:
            val_y = val_y.to_numpy(dtype="float32")

    print("Train x:", train_x.shape)
    print("Train y:", train_y.shape)
    print(np.unique(train_y, return_counts=True))
    print("Test x:", test_x.shape)
    print("Test y:", test_y.shape)
    print(np.unique(test_y, return_counts=True))

    if validation_split > 0:
        print("Validation x:", val_x.shape)
        print("Validation y:", val_y.shape)
        print(np.unique(val_y, return_counts=True))
        result = [train_x, train_y, val_x, val_y, test_x, test_y]
    else:
        result = [train_x, train_y, test_x, test_y]

    return result


def data_split(data, labels, id_col, test_split, validation_split):
    # labels = labels.groupby('idp').tail(1).Mortality
    idx = np.unique(id_col)
    np.random.shuffle(idx)
    idx = list(idx)
    test_size = int(round(len(idx) * test_split))
    train_size = int(round(len(idx) * (1 - (test_split + validation_split))))
    validation_size = int(round(len(idx) * validation_split))

    test_ids = idx[0:test_size]
    train_ids = idx[test_size:(test_size + train_size)]
    validation_ids = idx[(test_size + train_size):]

    data = np.array(data, dtype="float32")
    test_x = data[id_col.isin(test_ids), :, :]
    train_x = data[id_col.isin(train_ids), :, :]
    val_x = data[id_col.isin(validation_ids), :, :]

    test_y = labels[id_col.isin(test_ids)]
    train_y = labels[id_col.isin(train_ids)]
    val_y = labels[id_col.isin(validation_ids)]
    print("Train x:", train_x.shape)
    print("Train y:", train_y.shape)
    print(np.unique(train_y, return_counts=True))
    print("Validation x:", val_x.shape)
    print("Validation y:", val_y.shape)
    print(np.unique(val_y, return_counts=True))
    print("Test x:", test_x.shape)
    print("Test y:", test_y.shape)
    print(np.unique(test_y, return_counts=True))

    return train_x, train_y.to_numpy(dtype="float32"), val_x, val_y.to_numpy(dtype="float32"), test_x, test_y.to_numpy(
        dtype="float32")


def create_subset(data, n):
    import numpy as np
    idx = np.unique(data['idp'])
    np.random.seed(13)
    np.random.shuffle(idx)
    data = data.copy()
    data.set_index('idp', inplace=True)
    sample_ids = idx[0:n]
    data = data.loc[sample_ids]
    data['idp'] = data.index
    return data


def oversampler(data, labels, proportion):
    pos_features = data[labels == 0, :, :]
    pos_labels = labels[labels == 0]
    ids = pos_features.shape[0]
    choices = np.random.choice(ids, round(proportion * sum(labels == 1)))
    pos_features = pos_features[choices, :, :]
    pos_labels = pos_labels.reset_index(drop=True)[choices]
    oversampled_data = np.concatenate([pos_features, data[labels == 1, :, :]], axis=0)
    oversampled_labels = np.concatenate([pos_labels, labels[labels == 1]], axis=0)
    order = np.arange(len(oversampled_labels))
    np.random.shuffle(order)
    oversampled_data = oversampled_data[order, :, :]
    oversampled_labels = oversampled_labels[order]

    return oversampled_data, oversampled_labels


def create_mortality_labels(data):
    situacio = data[['idp', 'situacio']].drop_duplicates()
    situacio["Mortality"] = (situacio.situacio == "D").astype(int).values
    situacio.set_index('idp', inplace=True)
    situacio["idp"] = situacio.index
    situacio = situacio[["idp", "Mortality"]]
    return (situacio)


def create_labels(data, target):
    labels = data[['idp'] + target]
    labels.set_index('idp', inplace=True)
    labels = labels.groupby('idp').tail(1)
    labels["idp"] = labels.index
    data = data.drop(target, axis=1)
    return data, labels


def data_padder(data_toPad, situacio, cols, yearsToTarget):
    patients = list()
    labels = pd.DataFrame(columns=["idp", "Mortality"])
    i = 0
    temp = np.unique(data_toPad.index)
    for id in temp:
        if i % 1000 == 0:
            print(datetime.now(), 'i = {}'.format(i))
        i = i + 1
        # print(np.where(temp  == id))
        ages = []
        data_id = data_toPad.loc[id]
        data_id = data_id[cols]
        labels = pd.concat([labels, situacio.loc[situacio.index == id]], ignore_index=True, axis=0)
        # Quito el último año, para no utilizar los datos del mismo año de la predicción
        for age in np.unique(data_id.age)[:(len(data_id.age) - yearsToTarget)]:
            ages.append(data_id[data_id.age == age].values.flatten().tolist())
        patients.append(ages)

    n_timesteps = max([len(item) for item in patients])
    n_features = len(patients[0][0])
    n_samples = len(patients)
    padded = pad_sequences_multi(patients, padding='pre', value=-99,
                                 truncating='pre', maxlen=n_timesteps)

    data_padded = tf.reshape(padded, [n_samples, n_timesteps, n_features])

    return (data_padded, labels)


def data_padder_v2(data_toPad, labels, cols, yearsToTarget):
    patients = list()
    labels_clean = pd.DataFrame(columns=labels.columns)
    i = 0
    temp = np.unique(data_toPad.index)
    for id in temp:
        if i % 1000 == 0:
            print(datetime.now(), 'i = {}'.format(i))
        i = i + 1
        # print(np.where(temp  == id))
        ages = list()
        data_id = data_toPad.loc[id]
        data_id = data_id[cols]
        labels_clean = pd.concat([labels_clean, labels.loc[labels.index == id]], ignore_index=True, axis=0)
        for age in np.unique(data_id.age)[
                   :(
                           len(data_id.age) - yearsToTarget)]:  # Quito el último año, para no utilizar los datos del mismo año de la predicción
            ages.append(data_id[data_id.age == age].values.flatten().tolist())
        patients.append(ages)

    n_timesteps = max([len(item) for item in patients])
    n_features = len(patients[0][0])
    n_samples = len(patients)
    padded = pad_sequences_multi(patients, padding='pre', value=-99,
                                 truncating='pre', maxlen=n_timesteps)
    data_padded = tf.reshape(padded, [n_samples, n_timesteps, n_features])

    return (data_padded, labels_clean)


def cm_analysis(y_true, y_pred, labels, target_names, title, ymap=None, ax=None, figsize=(10, 10)):
    import seaborn as sns
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm_perc = pd.DataFrame(cm_perc, index=target_names, columns=target_names)
    if ax == None:
        print("Hola")
        fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_perc, annot=annot, fmt='', ax=ax, cmap="Blues")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title + '\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # ax.show()


def fitModel(model, train_x, train_y, val_x, val_y, cols, categoric_cols, numeric_cols, static_cols, epochs, batch_size,
             callbacks):
    import numpy as np
    static_cols_index = np.isin(cols, static_cols)
    train_statics = train_x[:, 0, static_cols_index]
    val_statics = val_x[:, 0, static_cols_index]
    history = model.fit(
        [train_x[:, :, np.isin(cols, categoric_cols)], train_x[:, :, np.isin(cols, numeric_cols)], train_statics],
        train_y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(
        [val_x[:, :, np.isin(cols, categoric_cols)], val_x[:, :, np.isin(cols, numeric_cols)], val_statics],
        val_y),
        callbacks=callbacks
    )
    return model, history


def two_loss_func(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) + 0.5 * dice_coef_loss(y_true, y_pred)


def plot_metrics(history):
    tpr = history.history['recall']
    val_tpr = history.history['val_recall']
    auc = history.history['auc']
    val_auc = history.history['val_auc']
    prec = history.history['precision']
    val_prec = history.history['val_precision']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(auc))
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, tpr, label='Training Recall')
    plt.plot(epochs_range, val_tpr, label='Validation Recall')
    plt.plot(epochs_range, auc, label='Training AUC')
    plt.plot(epochs_range, val_auc, label='Validation AUC')
    plt.plot(epochs_range, prec, label='Training Precision')
    plt.plot(epochs_range, val_prec, label='Validation Precision')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Metrics')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def plot_metrics_onlyTraining(history):
    tpr = history['recall']
    # val_tpr = history.history['val_recall']
    auc = history['auc']
    # val_auc = history.history['val_auc']
    prec = history['precision']
    # val_prec = history.history['val_precision']
    loss = history['loss']
    # val_loss = history.history['val_loss']
    epochs_range = range(len(auc))
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, tpr, label='Training Recall')
    # plt.plot(epochs_range, val_tpr, label='Validation Recall')
    plt.plot(epochs_range, auc, label='Training AUC')
    # plt.plot(epochs_range, val_auc, label='Validation AUC')
    plt.plot(epochs_range, prec, label='Training Precision')
    # plt.plot(epochs_range, val_prec, label='Validation Precision')
    plt.legend(loc='lower right')
    plt.title('Training metrics')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training  loss')
    plt.show()


def plot_metrics_multi(history):
    tpr = history.history['categorical_accuracy']
    val_tpr = history.history['categorical_accuracy']
    auc = history.history['cohen_kappa']
    val_auc = history.history['val_cohen_kappa']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(auc))
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, tpr, label='Training Cat. Accuracy')
    plt.plot(epochs_range, val_tpr, label='Validation Cat. Accuracy')
    plt.plot(epochs_range, auc, label="Training Cohen's Kappa")
    plt.plot(epochs_range, val_auc, label="Validation Cohen's Kappa")
    plt.legend(loc='lower right')
    plt.title('Training and Validation Metrics')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def get_bestThreshold(pred_y, real_y, plot=True):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(real_y, pred_y)
    # plot the roc curve for the model
    if plot:
        plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        plt.plot(fpr, tpr, marker='.', label='Model')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        # show the plot
        plt.show()

    gmeans = np.sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print('Best Threshold for validation=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    return thresholds[ix], gmeans[ix]


def plot_predictedProbs(train_y, train_y_pred, val_y, val_y_pred, test_y, test_y_pred, threshold):
    import seaborn as sns
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Probability distribution per real class')
    df = pd.DataFrame({'Real': train_y, 'Predicted': train_y_pred.flatten()})
    sns.histplot(ax=ax[0], data=df, x='Predicted', hue='Real', stat='count', edgecolor=None)
    ax[0].axvline(x=threshold)
    ax[0].set_title("Training")
    df = pd.DataFrame({'Real': val_y, 'Predicted': val_y_pred.flatten()})
    sns.histplot(ax=ax[1], data=df, x='Predicted', hue='Real', stat='count', edgecolor=None)
    ax[1].axvline(x=threshold)
    ax[1].set_title("Validation")
    df = pd.DataFrame({'Real': test_y, 'Predicted': test_y_pred.flatten()})
    sns.histplot(ax=ax[2], data=df, x='Predicted', hue='Real', stat='count', edgecolor=None)
    ax[2].axvline(x=threshold)
    ax[2].set_title("Test")
    plt.show()
    plt.show(block=True)

    fig.savefig("PredictedProbs.pdf", dpi=720)


def plot_predictedProbs_TestTrain(train_y, train_y_pred, test_y, test_y_pred, threshold):
    import seaborn as sns
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Probability distribution per real class')
    df = pd.DataFrame({'Real': train_y, 'Predicted': train_y_pred.flatten()})
    sns.histplot(ax=ax[0], data=df, x='Predicted', hue='Real', stat='count', edgecolor=None)
    ax[0].axvline(x=threshold)
    ax[0].set_title("Training")
    # df = pd.DataFrame({'Real': val_y, 'Predicted': val_y_pred.flatten()})
    # sns.histplot(ax=ax[1], data=df, x='Predicted', hue='Real', stat='count', edgecolor=None)
    # ax[1].axvline(x=threshold)
    # ax[1].set_title("Validation")
    df = pd.DataFrame({'Real': test_y, 'Predicted': test_y_pred.flatten()})
    sns.histplot(ax=ax[1], data=df, x='Predicted', hue='Real', stat='count', edgecolor=None)
    ax[1].axvline(x=threshold)
    ax[1].set_title("Test")
    plt.show()
    plt.show(block=True)

    fig.savefig("PredictedProbs.pdf", dpi=720)


def plot_predictedProbs_Test(test_y, test_y_pred, threshold, name):
    import seaborn as sns
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.suptitle('Calibration plot - test data')
    df = pd.DataFrame({'Real': test_y, 'Predicted': test_y_pred.flatten()})
    sns.histplot(ax=ax, data=df, x='Predicted', hue='Real', stat='count', edgecolor=None)
    ax.axvline(x=threshold)
    plt.show()
    plt.show(block=True)
    fig.savefig(name + ".pdf", dpi=720)


def calibration_plot(test_y, test_y_pred, title_plot, name_file):
    from sklearn.calibration import calibration_curve
    import matplotlib.lines as mlines
    plot_y, plot_x = calibration_curve(test_y, test_y_pred, n_bins=20)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.hist(test_y_pred[test_y == 0], weights=np.ones_like(test_y_pred[test_y == 0]) / len(test_y[test_y == 0]),
            alpha=0.4, label="Negative", bins=50)
    ax.hist(test_y_pred[test_y == 1], weights=np.ones_like(test_y_pred[test_y == 1]) / len(test_y[test_y == 1]),
            alpha=0.4, label="Positive", bins=50)
    ax.plot(plot_x, plot_y, marker='o', linewidth=1, label="Model")
    line = mlines.Line2D([0, 1], [0, 1], color="blue", label="Ideally calibrated")
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    fig.suptitle(title_plot)
    ax.set_xlabel("Predicted positive probability")
    ax.set_ylabel("Observed positive proportion")
    plt.legend(loc="upper left")
    plt.show()
    fig.savefig(name_file + ".pdf", dpi=720)


# https://towardsdatascience.com/get-confidence-intervals-for-any-model-performance-metrics-in-machine-learning-f9e72a3becb2
def one_boot(*data_args):
    """Usage: (t, p) = one_boot(true, pred) with true, pred, t, p arrays of same length
    """
    length = len(data_args[0])
    index = np.random.randint(0, length, size=length)  # apply same sampled index to all args:
    return [pd.Series(arg.values[index], name=arg.name)  # .reset_index() is slower
            if isinstance(arg, pd.Series) else arg[index] for arg in data_args
            ]


import re


def calc_metrics(metrics, *data_args):
    """Return a list of calculated values for each metric applied to *data_args
    where metrics is a metric func or iterable of funcs e.g. [m1, m2, m3, m4]
    """
    metrics = _fix_metrics(metrics)
    mname = metrics.__name__ if hasattr(metrics, '__name__') else "Metric"
    return pd.Series \
        ([m(*data_args) for m in metrics], index=[_metric_name(m) for m in metrics], name=mname)


def _metric_name(metric):  # use its prettified __name__
    name = re.sub(' score$', '', metric.__name__.replace('_', ' ').strip())
    return name.title() if name.lower() == name else name


def _fix_metrics(metrics_):  # allow for single metric func or any iterable of metric funcs
    if callable(metrics_): metrics_ = [metrics_]  # single metric func to list of one
    return pd.Series(metrics_)  # in case iterable metrics_ is generator, generate & store


def raw_metric_samples(metrics, *data_args, nboots):
    """Return dataframe containing metric(s) for nboots boot sample datasets
    where metrics is a metric func or iterable of funcs e.g. [m1, m2, m3]
    """
    metrics = _fix_metrics(metrics)
    cols = [calc_metrics(metrics, *boot_data) for boot_data in _boot_generator \
        (*data_args, nboots=nboots) if len(np.unique(boot_data[0])) > 1  # >1 for log Loss, ROC
            ]  # end of list comprehension
    return pd.DataFrame \
        ({iboot: col for iboot, col in enumerate(cols)}  # end of dict comprehension
         ).rename_axis("Boot", axis="columns").rename_axis(cols[0].name)


def _boot_generator(*data_args, nboots):  # return Gener of boot sampl datasets, not huge list!
    return (one_boot(*data_args) for _ in range(nboots))  # generator expression


def ci_auto(metrics, *data_args, alpha=0.05, nboots=None, Sample, CV):
    """Return Pandas data frame of bootstrap confidence intervals.
    PARAMETERS:
    metrics : a metric func or iterable of funcs e.g. [m1, m2, m3]
    data_args : 1+ (often 2, e.g. ytrue,ypred) iterables for metric
    alpha: = 1 - confidence level; default=0.05 i.e. confidence=0.95
    nboots (optional!): # boots drawn from data; dflt None ==> calc. from alpha
    """
    # alpha, nboots = _get_alpha_nboots(alpha, nboots)
    metrics = _fix_metrics(metrics)
    result = raw_metric_samples(metrics, *data_args, nboots=nboots)
    nb = result.shape[1]  # num boots we ended up with
    if nb < nboots:
        t = f'Note: {nboots - nb} boot sample datasets dropped\n'
        print(t + f'(out of {nboots}) because all vals were same in 1st data arg.')
    result = result.apply(lambda row: row.quantile([0.5 * alpha, 1 - 0.5 * alpha]), axis=1)
    result.columns = [f'{x * 100:.4g}%ile' for x in (0.5 * alpha, 1 - 0.5 * alpha)]
    result.insert(0, "Observed", calc_metrics(metrics, *data_args))  # col for obs (point estim)
    result = result.rename_axis(f"%ile for {nb} Boots", axis="columns")
    result["Sample"] = Sample
    result["CV"] = CV
    return result


def kappa_loss(y_pred, y_true, y_pow=2, eps=1e-10, N=5, bsize=256, name='kappa'):
    """A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            bsize: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss."""

    with tf.name_scope(name):
        y_true = tf.to_float(y_true)
        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)

        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))

        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(y_true, 0)

        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                              tf.to_float(bsize))

        return nom / (denom + eps)
