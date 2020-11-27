import sys
sys.path.append('..')

import glob
import json
import os
import hashlib
import pandas as pd
import numpy as np
import joblib

from functools import reduce
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from aiflib.config import Config
from aiflib.logger import Logger

class DataManager():
    def __init__(self, directory):
        self.config = Config()
        self.logger = Logger(__name__)
        self.logger.info(f"Loading data from {directory}...")
        self.target_column_name = self.config.target_column
        self.feature_column_names = None
        self.is_single_file = self.config.csv_name is not None
        
        self.raw_data = self.read_all_data(directory) 
        if self.raw_data is None: return
        self.logger.info(f"Done read [{len(self.raw_data)}] points.")

    def read_all_data(self, directory):
        dataframe_from_csv = self.read_all_csv(directory)

        if self.is_single_file:
            if dataframe_from_csv is None:
                self.logger.info(f"Not able to read any valid csv data in [{os.path.join(directory, self.config.csv_name)}]")
                return None
            return dataframe_from_csv
        else:
            if dataframe_from_csv is None:
                self.logger.info(f"Unable to read any valid data from *.csv files in [{directory}]")
                return None
        
        return dataframe_from_csv

    def read_all_csv(self, directory):

        help_string = " The csv file must contain a header, a target column and and at least one feature column." \
            " The target column name is set by the <input_column> variable of this run. The default value is 'target'."

        paths = []
        if self.is_single_file:
            paths = [os.path.join(directory, self.config.csv_name)]
        else:
            paths = glob.glob(os.path.join(directory, "*.csv"), recursive=True)

        frames = []
        for path in paths:
            try:
                self.logger.verbose(f"Attempting to read data from csv [{path}]"
                                    f" with delimiter [{self.config.delimiter}]")
                frame = pd.read_csv(path, error_bad_lines = False,
                                    delimiter = self.config.delimiter,
                                    encoding = self.config.encoding
                                    )
            except Exception as e:
                self.logger.info(f"Failed to read csv [{path}] exception:\n{e}")
                continue

            if self.target_column_name not in frame.columns:
                self.logger.info(f"File [{path}] does not have name [{self.target_column_name}] in header"
                                 f"{list(frame.columns)}', skipping this file." + help_string)
                continue

            frames.append(frame)
            self.logger.verbose(f"Read [{len(frame)}] data points from [{path}]\n")

        if len(frames) == 0: return None
        coalesced = reduce(lambda a,b: a.append(b), frames[1:], frames[0])
        
        return coalesced

    def validate(self, for_train = True):
        if self.raw_data is None: 
            return False
        elif not for_train: 
            return True
        else:
            return True

    @staticmethod
    def write_dataframe(frame, name, directory = None):
        config = Config()
        if directory is None:
            path = os.path.join(config.artifacts_directory, f"{name}.csv")
        else:
            path = os.path.join(directory, f"{name}.csv")
        frame.to_csv(path, index=False, sep = config.delimiter)
        checksum = DataManager.checksum(path)
        return checksum

    @staticmethod
    def checksum(path):
        hasher = hashlib.md5()
        with open(path, 'rb') as infile:
            hasher.update(infile.read())
        return hasher.hexdigest()

    def get_data(self):
        return self.raw_data
        
    def get_target_column(self):
        return self.target_column_name
    
    def get_feature_columns(self):
        self.feature_column_names = list(self.raw_data.drop(self.target_column_name, axis=1).columns)
        return self.feature_column_names