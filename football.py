import io
import logging
import os

import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from LogReg import LogReg

from data.nflsavant_format import FootballHeader

DATA_DIR = "./data/"
PBP = "pbp-"
SUFFIX = ".csv"

# BEGIN_YEAR = 2013
BEGIN_YEAR = 2020
END_YEAR = 2020

LOG_LEVEL = logging.INFO
LOG_LEVEL = logging.DEBUG

logging.getLogger().setLevel(LOG_LEVEL)

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

    def get_simplified_data(self):

        # Get data where PlayType is Rush or Pass; not kickoff, punt, field goal, kneel, etc.  
        rush_or_pass = self.football_data.loc[self.football_data['PlayType'].isin(['RUSH','PASS'])]
        print("Rush or Pass")
        print(f"{rush_or_pass}")

        subset_x = rush_or_pass[["Down", "ToGo", "YardLineFixed", "SeriesFirstDown", "SeasonYear", "OffenseTeam"]]     
        # subset_x = rush_or_pass[["Down", "ToGo", "YardLineFixed", "SeriesFirstDown", "SeasonYear"]]        
        # subset_x = rush_or_pass[["Down", "ToGo", "YardLineFixed", "SeriesFirstDown"]]
        # subset_x = rush_or_pass[["Down", "ToGo", "YardLine"]]        
        # subset_x = rush_or_pass[["Down", "ToGo"]]
        # subset_x = rush_or_pass[["Down"]]
        # subset_x = rush_or_pass[["ToGo"]]
        print("Subset X ")
        print(f"{subset_x}")

        array_x_unscaled = subset_x.to_numpy()

        scaler = preprocessing.StandardScaler().fit(array_x_unscaled)
        array_x = scaler.transform(array_x_unscaled)
        print("Scaled numpy version of X")
        print(f"{array_x}")

        subset_y = rush_or_pass[["PlayType"]]
        subset_y['PlayType'] = (subset_y['PlayType'] == 'PASS').astype(int)
        print("Subset Y")
        print(f"{subset_y}")

        array_y = subset_y.to_numpy().ravel()
        print(f"array y \n{array_y} \n  {array_y.shape}")
        return array_x, array_y

    def model_logit(self):
        array_x, array_y = self.get_simplified_data()
        X_train, X_test, y_train, y_test = train_test_split(array_x, array_y, test_size=0.2)
        lr = LogReg(X_train, X_test, y_train, y_test)


    def print_stats(self, frame):

        # Num lines
        print(f"Total:   {len(frame)}")

        # Overall probability of run, pass, or 'other'
        non_zeros = np.count_nonzero(frame, axis=0)

        print("--- Rushing and Passing ---")
        print(f"Rushing: {non_zeros[FootballHeader.IsRush]}")
        print(f"Passing: {non_zeros[FootballHeader.IsPass]}")

        # Count the types of plays
        print("\n--- Types of Plays ---")
        counts = frame[FootballHeader.PlayType.name].value_counts()
        print(f"{counts}")

    def print_yearly_stats(self):
        for ii in range(BEGIN_YEAR, END_YEAR + 1):
            print(f"\n--- {ii} ---")
            self.print_stats(self.football_yearly[str(ii)])

    def print_overall_stats(self):
        self.print_stats(self.football_data)


if __name__ == "__main__":
    football = Football()
    football.read_data()

    # football.print_overall_stats()
    # football.print_yearly_stats()
    football.model_logit()
