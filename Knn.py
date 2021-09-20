#
# Helper class for K nearest neighbors regressor
#
import logging

from sklearn.neighbors import KNeighborsRegressor

import logger

logger.set_logging()


class Knn:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        clf = KNeighborsRegressor(n_neighbors=10)
        self.clf = clf.fit(X_train, y_train)

    def score(self):
        score = self.clf.score(self.X_test, self.y_test)
        logging.info(f"Test:  {score}")
        return score
