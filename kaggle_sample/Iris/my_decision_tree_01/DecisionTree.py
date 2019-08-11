import numpy as np
from typing import List
from kaggle_sample.Iris.my_decision_tree_01._Node import _Node


class DecisionTree:
    """ 分類木学習 """

    def __init__(self, criterion: float=0.1):
        """

        :param criterion: 剪定条件
        """
        self.root: _Node = None
        self.criterion: float = criterion

    def fit(self, data, target):
        """
        学習を行う
        :param data:
        :param target:
        :return:
        """
        self.root = _Node()
        self.root.build(data, target)
        self.root.prune(self.criterion, self.root.numdata)

    def predict(self, data):
        """
        予測を行う
        :param data:
        :return:
        """
        answer: List = []
        for d in data:
            answer.append(self.root.predict(d))

        return np.array(answer)

    def print_tree(self):
        """ 木出力 """
        self.root.print_tree(0, "")
