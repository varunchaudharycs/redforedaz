import logging
import os
import sys
from pathlib import Path
import json
import configparser
import pandas as pd
import numpy as np
import uuid
from sqlalchemy import create_engine


class Warehousing:

    def __init__(self):
        print('Warehousing data ...')

    # Takes user input for file paths
    def user_inputs(self):
        input_path = input('Enter the input data sample file path:')
        if not Path(input_path).is_file():
            print('Invalid file path- File does not exist. Please enter again.')
            self.user_inputs()
        return input_path

    # Reads data
    def read_file(self, input_path):
        print('Reading data...')
        try:
            data_frame = pd.read_csv(input_path, delimiter = ';')
        except Exception as e:
            print('An exception occurred while reading file- ', input_path)
            self.safe_exit(e)
        print('Reading data... DONE.')
        return data_frame

    # Flattens nested JSON
    @staticmethod
    def process(data_frame):
        print('Processing data...')
        # Converting column names to lower case
        data_frame.columns = map(str.lower, data_frame.columns)
        # Filtering columns according to schema
        cols_raw = ["id", "username", "text", "retweets", "favorites", "date", "geo", "mentions", "hashtags",
                    "permalink"]
        data_frame = data_frame.reindex(columns=cols_raw)
        print('Processing data... DONE.')
        return data_frame

    # Saves results
    def write_to_db(self, data_frame):
        print('Writing to DB...')
        # TODO: store respective columns in user, vehicle, event, raw. Update insert query
        sql_connector = create_engine('mysql+pymysql://root:0404@localhost/redforedaz')
        try:
            # Inserting data into tables
            data_frame.to_sql('tweet_info', con=sql_connector, if_exists='append', chunksize=1000, index=False)
        except Exception as e:
            print('An exception occurred while writing to db')
            self.safe_exit(e)
        print('Writing to DB... DONE.')

    @staticmethod
    def safe_exit(e):
        print('Exception-\n{}\nExiting application.'.format(e))
        sys.exit(1)


# Stores tweets data in DB
def run():
    """
The Analytics team would like to be able the query the information inside these messages with sql.
The department head has asked you to capture and warehouse this data for easy querying.
The JSON messages are in the file “data_sample.txt“.
Warehouse this data in a set of sql tables that you feel best represents the data.
Feel free to use any relevant packages for your answer.
    """
    obj = Warehousing()
    # Checking file paths in config file
    input_path = '/home/varun/PycharmProjects/redForEdAZ_sentiment/input/redforedaz_2019.csv'
    df_raw = obj.read_file(input_path)
    df_processed = obj.process(df_raw)
    obj.write_to_db(df_processed)
    print('Warehousing data ...DONE.')


if __name__ == "__main__":
    run()