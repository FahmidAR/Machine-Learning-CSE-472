"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""

import numpy as np


def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement

    # print("y_true: ", y_true)
    # print("y_pred: ", y_pred)

    return np.sum(y_true == y_pred) / len(y_true)
    

def precision_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    # print("tp: ", tp)
    # print("fp: ", fp)


    return tp / (tp + fp)


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))


    # print("tp: ", tp)
    # print("fn: ", fn)

    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return 2 * (precision * recall) / (precision + recall)
