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
from hygene.cue import Cue
from hygene.hygene import Hygene


class Football:

    def __init__(self):
        self.log = logger.getLogger()
        self.data = FootballData()

    def run_all(self):
        self.simple_model()
        self.print_stats()
        self.log.info("------------------- logit ----------------------")
        self.model_logit()
        self.log.info("------------------- dectree ----------------------")
        self.model_dectree()
        self.log.info("------------------- random forest ----------------------")
        self.model_random_forest()
        self.log.info("------------------- knn ----------------------")
        self.model_knn()
        self.log.info("------------------- mlp ----------------------")
        self.model_mlp_search_params()

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
            self.log.info(f"--- Column List: {column_list} --- ")
            array_x, array_y = self.data.get_simplified_data(column_list)
            X_train, X_test, y_train, y_test = train_test_split(array_x, array_y, test_size=0.2)
            lr = LogReg(X_train, X_test, y_train, y_test)
            lr.score()

    def model_dectree(self):
        for column_list in self.get_columns_to_use():
            avg = 0
            # Noisy, so do multiple times
            for ii in range(0, 10):
                self.log.info(f"--- Column List: {column_list} --- ")
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
                self.log.info(f"--- Column List: {column_list} --- ")
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
                self.log.info(f"--- Column List: {column_list} --- ")
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
                self.log.info(f"--- Column List: {column_list} --- ")
                array_x, array_y = self.data.get_simplified_data(column_list)
                X_train, X_test, y_train, y_test = train_test_split(array_x, array_y, test_size=0.2)
                dt = MLP(X_train, X_test, y_train, y_test)
                avg += dt.score()

            avg /= 10.
            print(f"Average: {avg}")

    def model_mlp_search_params(self):
        cols = ["Down", "ToGo", "YardLineFixed", "SeriesFirstDown", "Quarter", "SeasonYear", "OffenseTeam",
                "DefenseTeam"]
        array_x, array_y = self.data.get_simplified_data(cols)
        X_train, X_test, y_train, y_test = train_test_split(array_x, array_y, test_size=0.2)
        dt = MLP(X_train, X_test, y_train, y_test)
        dt.param_search()

    def simple_model(self):
        """ Trivial estimate, where we pass unless it is 4th and short, then run"""
        array_x, array_y = self.data.get_simplified_data(["Down", "ToGo"], False)

        # Pass is a '1'.  So make all ones
        predict_y = np.ones(array_y.shape[0])

        # If ALWAYS pass, what is the score?
        score = accuracy_score(array_y, predict_y)
        self.log.info(f"Simple score (all pass): {score}")

        # Set short yardage to zeros
        predict_y = np.ones(array_y.shape[0])
        for ii in range(0, array_y.shape[0]):
            if array_x[ii][1] < 4:
                predict_y[ii] = 0

        score = accuracy_score(array_y, predict_y)
        self.log.info(f"Simple score (all pass except short yardage): {score}")
        self.log.info(f"Number of zeros:  {np.count_nonzero(predict_y == 0)}")

        # Set late AND short yardage to zeros
        predict_y = np.ones(array_y.shape[0])
        for ii in range(0, array_y.shape[0]):
            if array_x[ii][0] >= 3 and array_x[ii][1] < 4:
                predict_y[ii] = 0

        score = accuracy_score(array_y, predict_y)
        self.log.info(f"Simple score (all pass except short yardage and late down): {score}")
        self.log.info(f"Number of zeros:  {np.count_nonzero(predict_y == 0)}")

    def print_stats(self):
        self.data.print_stats()

    def run_hygene(self):
        self.log.info("---------------------------------------------- Football Data -----------------------------------------")
        array_x, array_y = self.data.get_simplified_data(["Down"], False)

        # What should this be??
        length = 12
        learning_rate = 0.95

        self.log.info("---------------------------------------------- Setting hygene data -----------------------------------------")

        # Convert Down to a Cue.
        cues = []
        for index, x_val in enumerate(array_x):
            # TODO:  Generalize this to other types (ToGo, yardline, etc.)
            down = np.full((length), -1)
            if x_val[0] == 1:
                down[length - 3:length] = 1
            if x_val[0] == 2:
                down[length - 6:length] = 1
            if x_val[0] == 3:
                down[length - 9:length] = 1
            if x_val[0] == 4:
                down[length - 12:length] = 1
            # print(f"x_val {x_val}  down {down}")

            # TODO:  Add other values

            # TODO:  Should there only be 2 hypotheses??
            if array_y[index] == 0:
                hypo = np.full((length), 1)
                event = 1
            else:
                hypo = np.full((length), -1)
                event = 2

            cue = Cue.with_np(down, hypo, event)

            # Apply learning.
            # TODO:  Figure out what learning should be.  Should
            #  it be higher for older?
            cue.apply_learning_rate(learning_rate)

            cues.append(cue)

        # Make some of them Semantic memory.
        # TODO:  Use grouping / clustering
        num_sem = int(.05 * len(cues))
        sem = []
        for ii in range(num_sem):
            rand_index = np.random.randint(0, len(cues))
            sem.append(cues.pop(rand_index))

        # Test on the last .1 of them
        probes = []
        num_test = -2   # -1 * int(.1 * len(cues))
        probes = cues[num_test:len(cues)]

        # Print sizes
        self.log.info(f"Size of cues: {len(cues)}")
        self.log.info(f"Size of semantic memory: {len(sem)}")
        self.log.info(f"Num probes: {len(probes)}")

        hygene = Hygene()
        hygene.set_traces(cues)
        hygene.set_semantic_memory(sem)

        for probe in probes:
            self.log.info("---------------------------------------------- New Probe -----------------------------------------")

            hygene.set_probe(probe)
            hygene.compute_activations()
            hygene.calculate_content_vectors()
            hygene.get_unspecified_probe()
            hygene.get_semantic_activations()
            hygene.sample_hypotheses()
            hygene.get_echo_intensities()
            probs = hygene.get_probabilities()
            highest_event = hygene.get_highest_prob_event()
            print(f"probe {probe}. GT: {probe.event}  probs: {probs}  HighestEvent: {highest_event}")






if __name__ == "__main__":
    football = Football()
    # football.run_all()
    football.run_hygene()
