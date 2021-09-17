#
# Simple wrapper for Logistic Regression
#

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

NUM_TO_TEST = 20


class LogReg:

    def __init__(self, X_train, X_test, y_train, y_test):
        # without cross validation
        # logistic_regression = LogisticRegression(random_state=0, max_iter=2000).fit(X_train, y_train)
        # With cross validation.
        logistic_regression = LogisticRegressionCV(random_state=0, max_iter=2000).fit(X_train, y_train)

        # ----------------------------------
        # Training data

        # First NUM_TO_TEST rows of x, all columns.
        # print(f" {X[:NUM_TO_TEST, :]}")

        # predict on those, categorical (0, 1, 2)
        prediction = logistic_regression.predict(X_train[:NUM_TO_TEST, :])
        # print(f" {y_train[:20]}  {prediction}")

        # Predict probability of being in each category, so 3 x continuous [0,1]
        probs = logistic_regression.predict_proba(X_train[:NUM_TO_TEST, :])
        # print(f" {y_train[:20]}  {probs}")

        # Score on training data.
        # Note that this uses the 'natural' scoring for this sort of classifier,
        # which is accuracy_score from _classification.py, which is simply
        # the % that match.
        score = logistic_regression.score(X_train, y_train)
        print(f"Train: {score}")  # 0.98 with CV, 0.973 without

        # AUC score
        auc_score = roc_auc_score(y_train, logistic_regression.predict(X_train), multi_class='ovr')
        print(f"Train: {auc_score}")  # 0.9989 with no-scaling;  0.99906 with scaling

        # ----------------------------------
        # Test data
        score = logistic_regression.score(X_test, y_test)
        print(f"Test: {score}")  # 0.98 with CV, 0.973 without

        # AUC score
        auc_score = roc_auc_score(y_test, logistic_regression.predict(X_test), multi_class='ovr')
        print(f"Test: {auc_score}")  # 0.9989 with no-scaling;  0.99906 with scaling
