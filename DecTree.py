#
# Helper class for Decision Tree classifier
#
import logging

from sklearn import tree
from sklearn.metrics import roc_auc_score

import logger

logger.set_logging()


class DecTree:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        clf = tree.DecisionTreeClassifier()
        self.clf = clf.fit(X_train, y_train)

    def show_tree(self):
        tree.plot_tree(self.clf)

    def score(self):
        score = self.clf.score(self.X_test, self.y_test)
        logging.info(f"Test:  {score}")
        return score

        # AUC score
        # prob_y = self.clf.predict_proba(self.X_test)
        # auc_score = roc_auc_score(self.y_test, prob_y, multi_class='ovr')
        # logging.info(f"Test:  {auc_score}")
