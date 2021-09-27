import io
import os

import numpy as np
import pandas
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from DecTree import DecTree
from Knn import Knn
from LogReg import LogReg
from MLP import MLP
from RandForest import RandForest
from data.foodball_data import FootballData


import logging
import logger

logger.set_logging()


class Football:

    def __init__():
        self.data =  FootballData().read_data()
  
    def get_columns_to_use(self):
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
            array_x, array_y = self.get_simplified_data(column_list)
            X_train, X_test, y_train, y_test = train_test_split(array_x, array_y, test_size=0.2)
            lr = LogReg(X_train, X_test, y_train, y_test)
            lr.score()

    def model_dectree(self):
        for column_list in self.get_columns_to_use():
            avg = 0
            # Noisy, so do multiple times
            for ii in range(0, 10):
                logging.info(f"--- Column List: {column_list} --- ")
                array_x, array_y = self.get_simplified_data(column_list)
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
                array_x, array_y = self.get_simplified_data(column_list)
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
                array_x, array_y = self.get_simplified_data(column_list)
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
                array_x, array_y = self.get_simplified_data(column_list)
                X_train, X_test, y_train, y_test = train_test_split(array_x, array_y, test_size=0.2)
                dt = MLP(X_train, X_test, y_train, y_test)
                avg += dt.score()

            avg /= 10.
            print(f"Average: {avg}")

    def simple_model(self):
        """ Trivial estimate, where we pass unless it is 4th and short, then run"""
        array_x, array_y = self.get_simplified_data(["Down", "ToGo"], False)

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

    def print_stats(self, frame):

        # Num lines
        logging.info(f"Total:   {len(frame)}")

        # Overall probability of run, pass, or 'other'
        non_zeros = np.count_nonzero(frame, axis=0)

        logging.info("--- Rushing and Passing ---")
        logging.info(f"Rushing: {non_zeros[FootballHeader.IsRush]}")
        logging.info(f"Passing: {non_zeros[FootballHeader.IsPass]}")

        # Count the types of plays
        logging.info("\n--- Types of Plays ---")
        counts = frame[FootballHeader.PlayType.name].value_counts()
        logging.info(f"{counts}")

    def print_yearly_stats(self):
        for ii in range(BEGIN_YEAR, END_YEAR + 1):
            logging.info(f"\n--- {ii} ---")
            self.print_stats(self.football_yearly[str(ii)])

    def print_overall_stats(self):
        self.print_stats(self.football_data)


if __name__ == "__main__":
    football = Football()

    football.simple_model()
    # football.print_overall_stats()
    # football.print_yearly_stats()
    # logging.info("------------------- logit ----------------------")
    # football.model_logit()
    # logging.info("------------------- dectree ----------------------")
    # football.model_dectree()
    # logging.info("------------------- random forest ----------------------")
    # football.model_random_forest()
    # logging.info("------------------- knn ----------------------")
    # football.model_knn()
    # logging.info("------------------- mlp ----------------------")
    # football.model_mlp()
