#
# Helper class for Random Forest classifier
#
from sklearn.ensemble import RandomForestClassifier
import logger

class RandForest:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.log = logger.getLogger()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        clf = RandomForestClassifier()
        self.clf = clf.fit(X_train, y_train)

    def score(self):
        score = self.clf.score(self.X_test, self.y_test)
        self.log.info(f"Test:  {score}")
        return score
