from nflsavant_format import FootballHeader

# Change this to get a smaller data set
# BEGIN_YEAR = 2013
BEGIN_YEAR = 2020
END_YEAR = 2020


DATA_DIR = "./data/"
PBP = "pbp-"
SUFFIX = ".csv"


class FootballData:

    def __init__(self):

        self.football_data = None
        self.football_yearly = {}
        self.read_data()


    def get_yearly(self):
        return self.football_yearly

    def get_data(self):
        return football_data

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
