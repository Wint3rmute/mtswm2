import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import parse_stroke_data_file

if __name__ == "__main__":
    X, y = parse_stroke_data_file.get_dataset_x_y()
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0
    )

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # clf = svm.SVC(kernel="linear", C=5).fit(X_train, y_train)
    clf = MLPClassifier(
        solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(50, 20), random_state=1, max_iter=4000
    ).fit(X_train, y_train)

    print(clf.score(X_test, y_test))
