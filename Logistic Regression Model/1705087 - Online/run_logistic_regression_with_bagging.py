"""
main code that you will run
"""

from data_handler import load_dataset, split_dataset
from ensemble import BaggingClassifier
from linear_model import LogisticRegression
from metrics import accuracy, f1_score, precision_score, recall_score

if __name__ == '__main__':
    #My name and id
    print("\n"+"*"*50)
    print("Name: Fahmid Al Rifat")
    print("Student ID: 1705087")
    print("*"*50+"\n")

    # data load
    X, y = load_dataset()

    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(X, y, 0.2,True)

    # training
    params = dict()
    params = {'learning_rate': 0.01, 'n_iters': 2500}

    base_estimator = LogisticRegression(params)
    classifier = BaggingClassifier(base_estimator=base_estimator, n_estimator=9)
    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)

    # performance on test set
    print('Accuracy ', accuracy(y_true=y_test, y_pred=y_pred))
    print('Recall score ', recall_score(y_true=y_test, y_pred=y_pred))
    print('Precision score ', precision_score(y_true=y_test, y_pred=y_pred))
    print('F1 score ', f1_score(y_true=y_test, y_pred=y_pred))
    print("*"*50+"\n")
