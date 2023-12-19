import keras as k
from keras import Sequential
from sklearn.model_selection import GridSearchCV


# Function that finds the best hyperparameter search through CV on the given classifier.
# Retrains on the whole set at the end.
def grid_search(classifier, X, y, parameters, folds=None, print_res=True):
    cvs = GridSearchCV(classifier, parameters, refit=True, cv=folds, scoring='accuracy',
                       n_jobs=10, verbose=10)
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
            k.layers.Dropout(dropout),
            k.layers.Dense(1, "sigmoid")
        ]
    )
    neural_net.compile(optimizer=optimizer, loss=k.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return neural_net.fit(train_X, train_y, epochs=epochs, verbose=2, validation_data=(val_X, val_y))
