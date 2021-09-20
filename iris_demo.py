# Iris data is the famous R. A. Fisher set:  https://en.wikipedia.org/wiki/Iris_flower_data_set
# X:  150 x 4.  Each of the 4 is a continuous variable.
# y:  150 x 1.  Categorical:  0, 1, 2;   30 each
import logging

import logger

logger.set_logging()

from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from LogReg import LogReg

NUM_ITERS = 5


class DataWrangler:

    def __init__(self):
        X, self.y = load_iris(return_X_y=True)
        # print(f"{X}")
        # print(f"{y}")
        # print(f"{len(X)}")

        # Optional:  Scale the data, so has mean 0 and variance 1.
        # Doesn't do much with this data
        scaler = preprocessing.StandardScaler().fit(X)
        self.X = scaler.transform(X)

    def create_train_test_and_fit(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        lr = LogReg(X_train, X_test, y_train, y_test)
        lr.score_multi_class()


if __name__ == "__main__":
    dw = DataWrangler()
    for ii in range(0, NUM_ITERS):
        dw.create_train_test_and_fit()
        logging.info("\n")
