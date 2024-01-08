import os
import warnings

import keras as k
import matplotlib.pyplot as plt
from keras import Sequential
from keras.src.utils.version_utils import callbacks
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV


# Function that finds the best hyperparameter search through CV on the given classifier.
# Retrains on the whole set at the end.
def grid_search(classifier, X, y, parameters, folds=5, print_res=True):
    cvs = GridSearchCV(classifier, parameters, refit=True, cv=folds, scoring='accuracy',
                       n_jobs=max(os.cpu_count() - 2, 1), verbose=4)
    # Suppress ConvergenceWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        cvs.fit(X, y)
    if print_res:
        print("Best parameters set found:")
        print(cvs.best_params_)
        print("Best accuracy score found:")
        print(cvs.best_score_)

    return cvs.best_score_, cvs.best_estimator_


def plot_roc_curve(y_true, y_probs, label='ROC curve'):
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    # Compute ROC AUC score
    roc_auc = roc_auc_score(y_true, y_probs)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='{} (AUC = {:.2f})'.format(label, roc_auc))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(label + ' ROC curve')
    plt.legend(loc='lower right')
    plt.show()


def plot_roc_curves(y_true, score_label):
    # Plot ROC curve
    plt.figure(figsize=(8, 6))

    for y_probs, label in score_label:
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)

        # Compute ROC AUC score
        roc_auc = roc_auc_score(y_true, y_probs)

        plt.plot(fpr, tpr, lw=2, label='{} (AUC = {:.2f})'.format(label, roc_auc))

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

def keras_mlp(train_X, train_y, val_X, val_y, optimizer, activation, dropout, epochs):
    neural_net = Sequential(
        [
            k.Input(shape=13),
            k.layers.Dense(units=50, activation=activation),
            k.layers.Dense(units=25, activation=activation),
            k.layers.Dense(units=25, activation=activation),
            k.layers.Dropout(dropout),
            k.layers.Dense(1, "sigmoid")
        ]
    )
    callback = [
        callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=7, cooldown=2, verbose=1, factor=0.25,
                                    min_lr=1e-7,
                                    min_delta=1e-5),
        callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1, min_delta=1e-5,
                                restore_best_weights=True)
    ]
    neural_net.compile(optimizer=optimizer, loss=k.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return neural_net.fit(train_X, train_y, batch_size=320, epochs=epochs, verbose=2, validation_data=(val_X, val_y),
                          callbacks=callback, use_multiprocessing=True, workers=10), neural_net
