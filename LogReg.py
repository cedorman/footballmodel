#
# Simple wrapper for Logistic Regression
#

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

import logger

class LogReg:

    def __init__(self, X_train, X_test, y_train, y_test):
        """Run a 'standard' LogReg process, use CV to optimize, then print results on
        train and test."""
        self.log = logger.getLogger()

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # without cross validation
        # logistic_regression = LogisticRegression(random_state=0, max_iter=2000).fit(X_train, y_train)
        # With cross validation.
        self.logistic_regression = LogisticRegressionCV(random_state=0, max_iter=2000).fit(X_train, y_train)

    def score(self):
        """ Score, where the y_train / y_test is binary.  """

        x_shape = self.X_train.shape
        y_shape = self.y_train.shape

        if x_shape[0] != y_shape[0]:
            self.log.warning(f"Problem with shape of x/y {x_shape} {y_shape}")
        if len(y_shape) != 1:
            self.log.warning(f"Problem with shape of x/y {x_shape} {y_shape}")

        # ----------------------------------
        # Training data

        # Score on training data.
        # Note that this uses the 'natural' scoring for this sort of classifier,
        # which is accuracy_score from _classification.py, which is simply
        # the % that match.
        score = self.logistic_regression.score(self.X_train, self.y_train)
        self.log.info(f"Train: {score}")

        # AUC score
        prediction = self.logistic_regression.predict_proba(self.X_train)[:, 1]
        auc_score = roc_auc_score(self.y_train, prediction, multi_class='ovr')
        self.log.info(f"Train: {auc_score}")

        # ----------------------------------
        # Test data
        score = self.logistic_regression.score(self.X_test, self.y_test)
        self.log.info(f"Test:  {score}")

        # AUC score
        prediction = self.logistic_regression.predict_proba(self.X_test)[:, 1]
        auc_score = roc_auc_score(self.y_test, prediction, multi_class='ovr')
        self.log.info(f"Test:  {auc_score}")

    def score_multi_class(self):
        """ Score, where the y_train / y_test is multiclass.  """

        x_shape = self.X_train.shape
        y_shape = self.y_train.shape

        if x_shape[0] != y_shape[0]:
            self.log.warning(f"Problem with shape of x/y {x_shape} {y_shape}")
        if len(y_shape) > 1:
            self.log.warning(f"Problem with shape of x/y {x_shape} {y_shape}")

        # ----------------------------------
        # Training data

        # Score on training data.
        # Note that this uses the 'natural' scoring for this sort of classifier,
        # which is accuracy_score from _classification.py, which is simply
        # the % that match.
        score = self.logistic_regression.score(self.X_train, self.y_train)
        self.log.info(f"Train: {score}")

        # AUC score
        prediction = self.logistic_regression.predict_proba(self.X_train)
        auc_score = roc_auc_score(self.y_train, prediction, multi_class='ovr')
        self.log.info(f"Train: {auc_score}")

        # ----------------------------------
        # Test data
        score = self.logistic_regression.score(self.X_test, self.y_test)
        self.log.info(f"Test:  {score}")

        # AUC score
        prediction = self.logistic_regression.predict_proba(self.X_test)
        auc_score = roc_auc_score(self.y_test, prediction, multi_class='ovr')
        self.log.info(f"Test:  {auc_score}")
