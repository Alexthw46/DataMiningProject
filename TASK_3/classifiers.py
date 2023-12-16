import multiprocessing

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
