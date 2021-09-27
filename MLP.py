#
# Helper class for K nearest neighbors regressor
#
import logging

from sklearn.neural_network import MLPClassifier

import logger

logger.set_logging()


class MLP:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def score(self, layer1=5, layer2=2):
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
        clf = MLPClassifier(solver='adam', alpha=1e-5,
                            hidden_layer_sizes=(layer1, layer2),
                            random_state=1)
        self.clf = clf.fit(self.X_train, self.y_train)
        score = self.clf.score(self.X_test, self.y_test)
        logging.info(f"Test:  {score}")
        return score

    def param_search(self):
        for layer1 in range(3, 10):
            for layer2 in range(2, 5):
                logging.info(f"Layer1:   {layer1}   Layer2: {layer2}")
                self.score(layer1, layer2)
