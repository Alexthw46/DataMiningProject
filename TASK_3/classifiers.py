import os
import warnings

import keras as k
from keras import Sequential
from keras.src.utils.version_utils import callbacks
from sklearn.exceptions import ConvergenceWarning
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