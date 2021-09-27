# Iris data is the famous R. A. Fisher set:  https://en.wikipedia.org/wiki/Iris_flower_data_set
# X:  150 x 4.  Each of the 4 is a continuous variable.
# y:  150 x 1.  Categorical:  0, 1, 2;   30 each
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import logger
from DecTree import DecTree
from Knn import Knn
from LogReg import LogReg
from RandForest import RandForest

NUM_ITERS = 5


class DataWrangler:

    def __init__(self):
        self.log = logger.getLogger()
        X, self.y = load_iris(return_X_y=True)
        # print(f"{X}")
        # print(f"{y}")
        # print(f"{len(X)}")

        # Optional:  Scale the data, so has mean 0 and variance 1.
        # Doesn't do much with this data
        scaler = preprocessing.StandardScaler().fit(X)
        self.X = scaler.transform(X)

    def create_train_and_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2)

    def do_logistic_regression(self):
        lr = LogReg(self.X_train, self.X_test, self.y_train, self.y_test)
        lr.score_multi_class()

    def do_decision_tree(self):
        dt = DecTree(self.X_train, self.X_test, self.y_train, self.y_test)
        dt.score()

    def do_random_forest(self):
        dt = RandForest(self.X_train, self.X_test, self.y_train, self.y_test)
        dt.score()

    def do_knn(self):
        dt = Knn(self.X_train, self.X_test, self.y_train, self.y_test)
        dt.score()


if __name__ == "__main__":
    dw = DataWrangler()

    for ii in range(0, NUM_ITERS):
        dw.create_train_and_test()

        # dw.do_logistic_regression()
        # dw.do_decision_tree()
        # dw.do_random_forest()
        dw.do_knn()
