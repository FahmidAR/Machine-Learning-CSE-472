import numpy as np

class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        #self.params = params

        print(params)
        print("*"*50+"\n")

        self.learning_rate = params['learning_rate']
        self.n_iters = params['n_iters']



    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = 1 / (1 + np.exp( - (np.dot(X, self.weights) + self.bias ) ))
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement

        y_predicted = 1 / (1 + np.exp( - (np.dot(X, self.weights) + self.bias ) ))   
        y_predicted = np.where( y_predicted > 0.5, 1, 0)
        return y_predicted
