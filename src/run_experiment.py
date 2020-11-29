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
from sklearn.feature_selection import chi2, SelectKBest 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier

import parse_stroke_data_file

# TODO: MANAGE WITH THIS SOMEHOW
warnings.filterwarnings("ignore")

def evaluate(clf, X, Y, n_splits=2, n_of_repeats=5, random_state=123):
    '''
    5  razy  powtarzana metody 2-krotna walidacja krzyżowa
    '''
    scores = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for i in range(n_of_repeats):
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            predict = clf.predict(X_test)
            scores.append(accuracy_score(y_test, predict))

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    return mean_score, std_score


FEATURES_RANGE = range(1,60)
HIDDEN_LAYER_SIZES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
MOMENTUM_VALUES = [0.0, 0.9]

if __name__ == "__main__":
    X, y = parse_stroke_data_file.get_dataset_x_y()

    print('num_of_features,hidden_layer_size,momentum,mean_score,std_score')
    '''
    Badania należy przeprowadzić dla różnej
    liczby cech (poczynając od jednej - najlepszej
    wg. wyznaczonego rankingu, a następnie dokładać
    kolejno po jednej
    '''
    for num_of_features in FEATURES_RANGE:
        X_new = SelectKBest(chi2, k=num_of_features).fit_transform(X, y)

        '''
        sieć jednokierunkowa z 1 warstwą ukrytą 
        dla 3  różnych  liczb neuronów w warstwie
        ukrytej  oraz  dla uczenia metodą propagacji
        wstecznej z momentum i bez momentum
        '''
        for hidden_layer_size in HIDDEN_LAYER_SIZES: # Mniej więcej pomiędzy rozmiarem liczby inputów i liczby outputów
            for momentum in MOMENTUM_VALUES: # 0.9 to domyślna wartość z dokumentacji scikita

                clf = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), momentum=momentum)

                mean_score, std_score = evaluate(clf, X_new, y)
                #       num_of_features             , hidden_layer_size, momentum, mean_score, std_score')
                print(f'{num_of_features},{hidden_layer_size},{momentum},{mean_score:.8},{std_score:.8}')
