import io
import logging
import os

import numpy as np
import pandas

from data.football_format import FootballHeader

DATA_DIR = "./data/"
PBP = "pbp-"
SUFFIX = ".csv"

BEGIN_YEAR = 2013
END_YEAR = 2020

LOG_LEVEL = logging.INFO


class Football:

    def __init__(self):
        logging.getLogger().setLevel(LOG_LEVEL)
        self.read_data()

    def read_data(self):
        self.football_data = None
        self.football_yearly = {}
        for ii in range(BEGIN_YEAR, END_YEAR + 1):
            filename = PBP + str(ii) + SUFFIX
            self.football_yearly[str(ii)] = self.read_datafile(filename)
            if self.football_data is None:
                self.football_data = self.football_yearly[str(ii)]
            else:
                self.football_data = self.football_data.append(self.football_yearly[str(ii)])

    def read_datafile(self, filename: str):
        try:
            print(f"Reading file:  {filename}")
            with io.open(os.path.join(DATA_DIR, filename), mode='r', encoding='utf-8-sig') as data_file:

                # The data is complicated because it has mixed types (categorical, continuous,
                # text-strings with commas, etc.).  so use pandas
                frame = pandas.read_csv(data_file)
                return frame


        except Exception as e:
            logging.warning("Error reading", exc_info=True)

    def model_logit(self):
        subset_x = self.football_data[["Down", "ToGo"]]
        print("Subset X ")
        print(f"{subset_x}")

        subset_y = self.football_data[["IsPass"]]
        print("Subset Y")
        print(f"{subset_y}")
        # X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        # lr = LogReg(X_train, X_test, y_train, y_test)

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
    # football.print_overall_stats()
    # football.print_yearly_stats()
    football.model_logit()
