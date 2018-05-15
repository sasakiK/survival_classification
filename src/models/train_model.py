# -*- coding: utf-8 -*-

import os
import click
import logging
import pandas as pd
from pathlib import Path
from sklearn import svm
from sklearn import tree
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import csv

@click.command()
@click.argument('input_filename')
def train_models(input_filename):

    FROM_DATA_DIR = "data/processed/"
    TO_DATA_DIR = "data/processed/"

    # load data
    df = pd.read_csv(FROM_DATA_DIR + input_filename)

    # define train dataset
    train_data = df.values
    xs = train_data[0:800, 2:] # features
    y  = train_data[0:800, 1]  # binary variables
    # define test datasets
    xs_test = train_data[801:train_data.shape[0], 2:] # features
    y_test  = train_data[801:train_data.shape[0], 1]

    # build ml models -------------------------

    # train svm
    clf_svm = svm.SVC()
    clf_svm = clf_svm.fit(xs, y)

    # train RandomForest
    clf_rf = RandomForestClassifier(n_estimators = 100)
    clf_rf = clf_rf.fit(xs, y)

    # train desicion tree
    clf_dt = tree.DecisionTreeClassifier()
    clf_dt = clf_dt.fit(xs, y)

    # train LogisticRegression
    clf_rl = LogisticRegression(random_state=1)
    clf_rl = clf_rl.fit(xs, y)

    # score to train data
    svm_score = clf_svm.score(xs, y)
    rf_score  = clf_rf.score(xs, y)
    dt_score  = clf_dt.score(xs, y)
    rl_score  = clf_rl.score(xs, y)

    # serialize models
    joblib.dump(clf_svm, 'models/clf_svm.pkl')
    joblib.dump(clf_rf, 'models/clf_rf.pkl')
    joblib.dump(clf_dt, 'models/clf_dt.pkl')
    joblib.dump(clf_rl, 'models/clf_rl.pkl')

    # print score
    print('svm score : %.3f' % svm_score)
    print('rf score  : %.3f' % rf_score)
    print('dt score  : %.3f' % dt_score)
    print('rl score  : %.3f' % rl_score)

    logger = logging.getLogger(__name__)
    logger.info('trained models were saved.')


def main():
    train_models()


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    main()
