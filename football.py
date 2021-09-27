import logging

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import logger
from DecTree import DecTree
from Knn import Knn
from LogReg import LogReg
from MLP import MLP
from RandForest import RandForest
from data.football_data import FootballData

logger.set_logging()


class Football:

    def __init__(self):
        self.data = FootballData()

    @staticmethod
    def get_columns_to_use():
        columns_to_use = [
            ["Down"],
            ["ToGo"],
            ["YardLine"],
            ["Down", "ToGo"],
            ["Down", "ToGo", "YardLine"],
            ["Down", "ToGo", "YardLineFixed", "SeriesFirstDown"],
            ["Down", "ToGo", "YardLineFixed", "SeriesFirstDown", "Quarter"],
            ["Down", "ToGo", "YardLineFixed", "SeriesFirstDown", "Quarter", "SeasonYear"],
            ["Down", "ToGo", "YardLineFixed", "SeriesFirstDown", "Quarter", "SeasonYear", "OffenseTeam"],
            ["Down", "ToGo", "YardLineFixed", "SeriesFirstDown", "Quarter", "SeasonYear", "OffenseTeam", "DefenseTeam"]
        ]
        return columns_to_use

    def model_logit(self):
        for column_list in self.get_columns_to_use():
            logging.info(f"--- Column List: {column_list} --- ")
            array_x, array_y = self.data.get_simplified_data(column_list)
            X_train, X_test, y_train, y_test = train_test_split(array_x, array_y, test_size=0.2)
            lr = LogReg(X_train, X_test, y_train, y_test)
            lr.score()

    def model_dectree(self):
        for column_list in self.get_columns_to_use():
            avg = 0
            # Noisy, so do multiple times
            for ii in range(0, 10):
                logging.info(f"--- Column List: {column_list} --- ")
                array_x, array_y = self.data.get_simplified_data(column_list)
                X_train, X_test, y_train, y_test = train_test_split(array_x, array_y, test_size=0.2)
                dt = DecTree(X_train, X_test, y_train, y_test)
                avg += dt.score()

            avg /= 10.
            print(f"Average: {avg}")

    def model_random_forest(self):
        for column_list in self.get_columns_to_use():
            avg = 0
            # Noisy, so do multiple times
            for ii in range(0, 10):
                logging.info(f"--- Column List: {column_list} --- ")
                array_x, array_y = self.data.get_simplified_data(column_list)
                X_train, X_test, y_train, y_test = train_test_split(array_x, array_y, test_size=0.2)
                dt = RandForest(X_train, X_test, y_train, y_test)
                avg += dt.score()

            avg /= 10.
            print(f"Average: {avg}")

    def model_knn(self):
        for column_list in self.get_columns_to_use():
            avg = 0
            # Noisy, so do multiple times
            for ii in range(0, 10):
                logging.info(f"--- Column List: {column_list} --- ")
                array_x, array_y = self.data.get_simplified_data(column_list)
                X_train, X_test, y_train, y_test = train_test_split(array_x, array_y, test_size=0.2)
                dt = Knn(X_train, X_test, y_train, y_test)
                avg += dt.score()

            avg /= 10.
            print(f"Average: {avg}")

    def model_mlp(self):
        for column_list in self.get_columns_to_use():
            avg = 0
            # Noisy, so do multiple times
            for ii in range(0, 10):
                logging.info(f"--- Column List: {column_list} --- ")
                array_x, array_y = self.data.get_simplified_data(column_list)
                X_train, X_test, y_train, y_test = train_test_split(array_x, array_y, test_size=0.2)
                dt = MLP(X_train, X_test, y_train, y_test)
                avg += dt.score()

            avg /= 10.
            print(f"Average: {avg}")

    def simple_model(self):
        """ Trivial estimate, where we pass unless it is 4th and short, then run"""
        array_x, array_y = self.data.get_simplified_data(["Down", "ToGo"], False)

        # Pass is a '1'.  So make all ones
        predict_y = np.ones(array_y.shape[0])
        score = accuracy_score(array_y, predict_y)
        logging.info(f"Simple score (all pass): {score}")

        # Set short yardage to zeros
        predict_y = np.ones(array_y.shape[0])
        for ii in range(0, array_y.shape[0]):
            if array_x[ii][1] < 4:
                predict_y[ii] = 0
            # if array_x[ii][0] == 4 and array_x[ii][1] < 4:
            # print(f"{ii} {array_x[ii][0]} {array_x[ii][1]} {predict_y[ii]}")

        score = accuracy_score(array_y, predict_y)
        logging.info(f"Simple score (all pass except short yardage): {score}")
        logging.info(f"Number of zeros:  {np.count_nonzero(predict_y == 0)}")

        # Set late and short yardage to zeros
        predict_y = np.ones(array_y.shape[0])
        for ii in range(0, array_y.shape[0]):
            if array_x[ii][0] >= 3 and array_x[ii][1] < 4:
                predict_y[ii] = 0

        score = accuracy_score(array_y, predict_y)
        logging.info(f"Simple score (all pass except short yardage and late down): {score}")
        logging.info(f"Number of zeros:  {np.count_nonzero(predict_y == 0)}")

    def print_stats(self):
        self.data.print_stats()


if __name__ == "__main__":
    football = Football()

    football.simple_model()
    football.print_stats()
    logging.info("------------------- logit ----------------------")
    football.model_logit()
    logging.info("------------------- dectree ----------------------")
    football.model_dectree()
    logging.info("------------------- random forest ----------------------")
    football.model_random_forest()
    logging.info("------------------- knn ----------------------")
    football.model_knn()
    logging.info("------------------- mlp ----------------------")
    football.model_mlp()
