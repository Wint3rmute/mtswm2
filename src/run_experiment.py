import itertools
import warnings

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier

import parse_stroke_data_file

# TODO: MANAGE WITH THIS SOMEHOW
warnings.filterwarnings('ignore') 

def get_mlp_classifier(
    solver="lbfgs",
    activation="relu",
    alpha=1e-5,
    hidden_layer_sizes=(5, 2),
    random_state=1,
    max_iter=100,
):
    return MLPClassifier(
        solver=solver,
        activation=activation,
        alpha=alpha,
        hidden_layer_sizes=hidden_layer_sizes,
        random_state=random_state,
        max_iter=max_iter,
    )


def calculate_score_with_kfold(clf, X, Y, n_splits=5, random_state=123):
    scores = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        scores.append(accuracy_score(y_test, predict))

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    return mean_score, std_score


if __name__ == "__main__":
    X, y = parse_stroke_data_file.get_dataset_x_y()
    # print(X.shape, y.shape)

    # Tweakables
    solver = ["lbfgs", "sgd", "adam"]
    activation = ["identity", "logistic", "tanh", "relu"]
    alpha = [1e-4, 1e-5, 1e-6]
    # hidden_layer_sizes = list(itertools.product(*[ list(range(1,10)) , list(range(1,10))] ))
    max_iter = list(np.arange(100, 10000, 100))

    all_tweakables = [solver, activation, alpha, max_iter]
    all_possible_tweakables_values = itertools.product(*all_tweakables)

    # print(len(list(all_possible_tweakables_values)))
    # exit()

    print("solver, activation, alpha, max_iter, mean_score, std_score")
        
    for solver, activation, alpha, max_iter in all_possible_tweakables_values:
        clf = get_mlp_classifier(solver, activation, alpha, hidden_layer_sizes=(5,2), max_iter=max_iter)
        mean_score, std_score = calculate_score_with_kfold(clf, X, y)
        # print("Accuracy score: %.3f (%.3f)" % (mean_score, std_score))
        print(f'{solver}, {activation}, {alpha}, {max_iter}, {mean_score}, {std_score}')
