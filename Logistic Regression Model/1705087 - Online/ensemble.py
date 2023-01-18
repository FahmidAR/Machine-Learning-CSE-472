from data_handler import bagging_sampler
import numpy as np


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # todo: implement

        print("base_estimator", base_estimator)
        print("n_estimator", n_estimator)
        print("*"*50+"\n")

        self.base_estimator = base_estimator
        self.n_estimator = n_estimator

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement

        self.estimators = []

        for _ in range(self.n_estimator):
            X_sample, y_sample = bagging_sampler(X, y)
            self.estimators.append(self.base_estimator.fit(X_sample, y_sample))

        return self

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement

        predict=np.zeros(len(X))
        
        for estimator in self.estimators:
            predict+=estimator.predict(X)

        return np.round(predict/self.n_estimator)
