"""
    Nie byliśmy pewni, jakie podejście przyjąć, żeby
    zbudować sensowne środowisko eksperymentalne.

    Postąpiliśmy następująco:
        - Napisaliśmy funkcje parsujące otrzymany dataset do formatu kompatybilnego ze scikitem
        - Wykorzystujemy k-krotną walidację krzyżową do oceny klasyfikatora
        - Napisaliśmy skrypt sprawdzający wiele mozliwych ustawień klasyfikatora MLP
        - Wyniki zapisywane są do pliku CSV, zamierzamy je później przeanalizować i znaleźć interesujące zakresy parametrów
"""

import itertools
import warnings

import numpy as np
from sklearn import svm
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier

import parse_stroke_data_file

# TODO: MANAGE WITH THIS SOMEHOW
warnings.filterwarnings("ignore")

# CONFIG

"""
Ewaluacja wykorzystanego klasyfikatora
z wykorzystaniem 5 razy powtarzanej metody
2-krotnej walidacji krzyżowej
"""
N_SPLITS = 2
N_REPEATS = 5

"""
Badania należy przeprowadzić dla różnej
liczby cech (poczynając od jednej - najlepszej
wg. wyznaczonego rankingu, a następnie dokładać
kolejno po jednej
"""
FEATURES_RANGE = range(1, 3)  # range(1,60)

"""
Sieć jednokierunkowa z 1 warstwą ukrytą dla
3 różnych liczb neuronów w warstwie ukrytej
oraz dla uczenia metodą propagacji wstecznej
z momentum i bez momentum
"""
HIDDEN_LAYER_SIZES = [2, 5, 7]  # [20, 30, 40, 50, 60, 70, 80, 90, 100]
MOMENTUM_VALUES = [0.0, 0.9]

# END OF CONFIG

# def get_classifiers():

if __name__ == "__main__":
    classifiers = {}
    X, y = parse_stroke_data_file.get_dataset_x_y()

    for num_of_features in FEATURES_RANGE:
        for hidden_layer_size in HIDDEN_LAYER_SIZES:
            for momentum_value in MOMENTUM_VALUES:
                new_classifier = MLPClassifier(
                    hidden_layer_sizes=(hidden_layer_size,), momentum=momentum_value
                )

                new_classifier.num_of_features = num_of_features

                classifiers[
                    f"features_{num_of_features}__hidden_{hidden_layer_size}__momentum_{momentum_value}"
                ] = new_classifier

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42
    )  # haha śmieszna liczba 42 haha

    scores = np.zeros((len(classifiers), N_SPLITS * N_REPEATS))

    for clf_id, clf_name in enumerate(classifiers):
        X_new = SelectKBest(
            chi2, k=classifiers[clf_name].num_of_features
        ).fit_transform(X, y)

        for fold_id, (train, test) in enumerate(rskf.split(X_new, y)):
            clf = clone(classifiers[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)

    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)

    for clf_id, clf_name in enumerate(classifiers):
        print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))

    np.save("results", scores)
