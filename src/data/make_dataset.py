# -*- coding: utf-8 -*-

import os
import click
import logging
import pandas as pd
from pathlib import Path

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def process_raw_df(input_filepath, output_filepath):


    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    # load dataset
    df= pd.read_csv(input_filepath).replace("male",0).replace("female",1)

    # treat missing data
    df["Age"].fillna(df.Age.median(), inplace=True)

    # make variables
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df2 = df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

    df2.to_csv(output_filepath)

    print(os.getcwd())
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


def main():
    process_raw_df()


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    main()
