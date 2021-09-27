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
from data.nflsavant_format import FootballHeader

DATA_DIR = "./data/"
PBP = "pbp-"
SUFFIX = ".csv"

# BEGIN_YEAR = 2013
BEGIN_YEAR = 2020
END_YEAR = 2020

import logging
import logger

logger.set_logging()


class Football:

    def __init__(self):

        self.football_data = None
        self.football_yearly = {}

    def read_data(self):
        logging.info("Starting read data")
        for ii in range(BEGIN_YEAR, END_YEAR + 1):
            filename = PBP + str(ii) + SUFFIX
            self.football_yearly[str(ii)] = self.read_datafile(filename)
            if self.football_data is None:
                self.football_data = self.football_yearly[str(ii)]
            else:
                self.football_data = self.football_data.append(self.football_yearly[str(ii)])
        logging.info("Done reading data")
        logging.debug(f"{self.football_data}")

    def read_datafile(self, filename: str):
        try:
            logging.info(f"Reading file:  {filename}")
            with io.open(os.path.join(DATA_DIR, filename), mode='r', encoding='utf-8-sig') as data_file:

                # The data is complicated because it has mixed types (categorical, continuous,
                # text-strings with commas, etc.).  so use pandas
                frame = pandas.read_csv(data_file, low_memory=False)
                # on_bad_lines ='skip',
                # dtype=FootballHeader.get_dtypes(),
                # error_bad_lines = False, warn_bad_lines=True)
                # low_memory = False)
                return frame

        except Exception as e:
            logging.warning("Error reading", exc_info=True)

    def get_simplified_data(self, columns_to_use, scale=True):
        """Turn the data into clean(er) numpy data"""

        logging.debug(f"Size of the original data: {self.football_data.shape}\n")

        # Get data where PlayType is Rush or Pass; not kickoff, punt, field goal, kneel, etc.  
        rush_or_pass = self.football_data.loc[self.football_data['PlayType'].isin(['RUSH', 'PASS'])]
        logging.debug("Rush or Pass Data")
        logging.debug(f"{rush_or_pass}\n")

        # Only columns that we want to use
        subset_x = rush_or_pass[columns_to_use]

        # convert some columns to factors
        for column in columns_to_use:
            if column == 'OffenseTeam' or \
                    column == 'DefenseTeam':
                subset_x[column] = subset_x[column].factorize()[0]

        # convert to numpy
        array_x_unscaled = subset_x.to_numpy()
        if scale:
            # scale, using numpy, to [-1,1]
            scaler = preprocessing.StandardScaler().fit(array_x_unscaled)
            array_x = scaler.transform(array_x_unscaled)
        else:
            array_x = array_x_unscaled
        logging.debug("Scaled numpy version of X")
        logging.debug(f"{array_x}")

        # get y as 0==RUSH 1==PASS
        subset_y = (rush_or_pass['PlayType'] == 'PASS').astype(int)
        logging.debug("Subset Y")
        logging.debug(f"{subset_y}")

        # Turn the 2d array into 1d array
        array_y = subset_y.to_numpy().ravel()
        logging.debug(f"array y \n{array_y} \n  {array_y.shape}")
        return array_x, array_y

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

    def model_mlp_search_params(self):
        cols = ["Down", "ToGo", "YardLineFixed", "SeriesFirstDown", "Quarter", "SeasonYear", "OffenseTeam",
                "DefenseTeam"]
        array_x, array_y = self.get_simplified_data(cols)
        X_train, X_test, y_train, y_test = train_test_split(array_x, array_y, test_size=0.2)
        dt = MLP(X_train, X_test, y_train, y_test)
        dt.param_search()

    def simple_model(self):
        """ Trivial estimate, where we pass unless it is 4th and short, then run"""
        array_x, array_y = self.get_simplified_data(["Down", "ToGo"], False)

        # Pass is a '1'.  So make all ones
        predict_y = np.ones(array_y.shape[0])

        # If ALWAYS pass, what is the score?
        score = accuracy_score(array_y, predict_y)
        logging.info(f"Simple score (all pass): {score}")

        # Set short yardage to zeros
        predict_y = np.ones(array_y.shape[0])
        for ii in range(0, array_y.shape[0]):
            if array_x[ii][1] < 4:
                predict_y[ii] = 0

        score = accuracy_score(array_y, predict_y)
        logging.info(f"Simple score (all pass except short yardage): {score}")
        logging.info(f"Number of zeros:  {np.count_nonzero(predict_y == 0)}")

        # Set late AND short yardage to zeros
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
    football.read_data()

    # football.simple_model()
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
    logging.info("------------------- mlp ----------------------")
    # football.model_mlp()
    football.model_mlp_search_params()
