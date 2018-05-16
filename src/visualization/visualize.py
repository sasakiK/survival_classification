import os
import click
import logging
import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def score_result(actual, predict, model_name):

    print("----------------- Result of {} -----------------".format(model_name))
    print("{0:10} .... %.3f".format("Accuracy") % accuracy_score(actual, predict))
    print("{0:10} .... %.3f".format("Precision") % precision_score(actual, predict))
    print("{0:10} .... %.3f".format("Recall") % recall_score(actual, predict))
    print("{0:10} .... %.3f".format("F-measure") % f1_score(actual, predict))


@click.command()
@click.argument('input_filename')
def visualize_result(input_filename):

    FROM_DATA_DIR = "data/processed/"

    df = pd.read_csv(FROM_DATA_DIR + input_filename)

    # svm
    score_result(df["origin"], df["svm"], "SVM")
    score_result(df["origin"], df["rf"], "RandomForest")
    score_result(df["origin"], df["dt"], "DecisionTree")
    score_result(df["origin"], df["rl"], "LogisticRegression")

    logger = logging.getLogger(__name__)
    logger.info('result of prediction.')


def main():
    visualize_result()

if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    main()
