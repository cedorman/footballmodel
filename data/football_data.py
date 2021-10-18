#
# FootballData is a class to handle reading in, cleaning up, and
# serving data from the nflsavant site.
#
# It will return the array_x data according to the columns that are passed,
# see nflsavant_format for the names of the columns.
#
# array_y is rush or pass:    RUSH==0, PASS == 1


import io
import os

import pandas
from sklearn import preprocessing

import logger
from data.nflsavant_format import FootballHeader

# Change this to get a smaller data set
BEGIN_YEAR = 2013
# BEGIN_YEAR = 2020
END_YEAR = 2020

DATA_DIR = "./data/"
PBP = "pbp-"
SUFFIX = ".csv"


class FootballData:

    def __init__(self):
        self.log = logger.getLogger()
        self.football_data = None
        self.football_yearly = {}
        self.read_data()

    def get_yearly(self):
        return self.football_yearly

    def get_data(self):
        return self.football_data

    def read_data(self):
        self.log.info("Starting read data")
        for ii in range(BEGIN_YEAR, END_YEAR + 1):
            ii_str = str(ii)
            filename = PBP + ii_str + SUFFIX
            self.football_yearly[ii_str] = self.read_datafile(filename)
            self.log.info(f"Year {ii_str}  Data: {len(self.football_yearly[ii_str])}")
            if self.football_data is None:
                self.football_data = self.football_yearly[ii_str]
            else:
                self.football_data = self.football_data.append(self.football_yearly[ii_str])
        self.log.info("Done reading data")
        self.log.debug(f"{self.football_data}")

    def read_datafile(self, filename: str):
        try:
            self.log.info(f"Reading file:  {filename}")
            with io.open(os.path.join(DATA_DIR, filename), mode='r', encoding='utf-8-sig') as data_file:

                # The data is complicated because it has mixed types (categorical, continuous,
                # text-strings with commas, etc.).  so use pandas
                frame = pandas.read_csv(data_file, low_memory=False)
                return frame

        except Exception:
            self.log.warning("Error reading", exc_info=True)

    def get_simplified_data(self, columns_to_use, scale=True):
        """Turn the data into clean(er) numpy data"""

        self.log.debug(f"Size of the original data: {self.football_data.shape}\n")

        # Get data where PlayType is Rush or Pass; not kickoff, punt, field goal, kneel, etc.  
        rush_or_pass = self.football_data.loc[self.football_data['PlayType'].isin(['RUSH', 'PASS'])]
        self.log.debug("Rush or Pass Data")
        self.log.debug(f"{rush_or_pass}\n")

        # Only columns that we want to use
        subset_x = rush_or_pass[columns_to_use]

        # convert some columns to factors
        for column in columns_to_use:
            if column == 'OffenseTeam' or column == 'DefenseTeam':
                subset_x[column] = subset_x[column].factorize()[0]

        # convert to numpy
        array_x_unscaled = subset_x.to_numpy()
        if scale:
            # scale, using numpy, to [-1,1]
            scaler = preprocessing.StandardScaler()
            scaler = scaler.fit(array_x_unscaled)
            array_x = scaler.transform(array_x_unscaled)
        else:
            array_x = array_x_unscaled
        self.log.debug("Scaled numpy version of X")
        self.log.debug(f"{array_x}")

        # get y as 0==RUSH 1==PASS
        subset_y = (rush_or_pass['PlayType'] == 'PASS').astype(int)
        self.log.debug("Subset Y")
        self.log.debug(f"{subset_y}")

        # Turn the 2d array into 1d array
        array_y = subset_y.to_numpy().ravel()
        self.log.debug(f"array y \n{array_y} \n  {array_y.shape}")
        return array_x, array_y

    def print_stats(self):
        self.log.info("-------- Overall stats ---------")
        self.print_sub_stats(self.football_data)

        for ii in range(BEGIN_YEAR, END_YEAR + 1):
            self.log.info(f"--- {ii} ---")
            self.print_sub_stats(self.football_yearly[str(ii)])

    def print_sub_stats(self, stats_to_print):
        # Num lines
        self.log.info(f"Total lines:   {len(stats_to_print)}")

        # Is rush or pass
        sub_data = stats_to_print.loc[stats_to_print['PlayType'].isin(['RUSH', 'PASS'])]

        self.log.info("--- Rushing and Passing ---")
        self.log.info(f"Rushing: {len(sub_data[sub_data['IsRush'] == 1])}")
        self.log.info(f"Passing: {len(sub_data[sub_data['IsPass'] == 1])}")

        # Count the types of plays
        self.log.info("\n--- All Types of Plays ---")
        counts = stats_to_print[FootballHeader.PlayType.name].value_counts()
        self.log.info(f"{counts}")
