# -*- coding: utf-8 -*-

import os
import click
import logging
import pandas as pd
from pathlib import Path

@click.command()
@click.argument('input_filename')
@click.argument('output_filename')
def process_raw_df(input_filename, output_filename):

    FROM_DATA_DIR = "data/raw/"
    TO_DATA_DIR = "data/processed/"

    # load dataset
    df= pd.read_csv(FROM_DATA_DIR + input_filename).replace("male",0).replace("female",1)

    # treat missing data
    df["Age"].fillna(df.Age.median(), inplace=True)

    # make variables
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df2 = df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

    df2.to_csv(TO_DATA_DIR + output_filename, index=None)

    # print(os.getcwd())
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


def main():
    process_raw_df()


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    main()
