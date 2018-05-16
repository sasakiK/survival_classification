
import click
import logging
import pandas as pd
from sklearn import datasets
from datetime import datetime
from sklearn.externals import joblib

@click.command()
@click.argument('input_filename')
def predict_models(input_filename):

    FROM_DATA_DIR = "data/processed/"

    # prepare tets data
    df = pd.read_csv(FROM_DATA_DIR + input_filename)

    # define train dataset
    train_data = df.values
    xs_test = train_data[801:train_data.shape[0], 2:] # features
    y_test  = train_data[801:train_data.shape[0], 1]  # target

    # 予測モデルを復元
    clf_svm = joblib.load('models/clf_svm.pkl')
    clf_rf = joblib.load('models/clf_rf.pkl')
    clf_dt = joblib.load('models/clf_dt.pkl')
    clf_rl = joblib.load('models/clf_rl.pkl')

    # check sroce to test data
    svm_score = clf_svm.score(xs_test, y_test)
    rf_score  = clf_rf.score(xs_test, y_test)
    dt_score  = clf_dt.score(xs_test, y_test)
    rl_score  = clf_rl.score(xs_test, y_test)

    # output predict result
    result_df = pd.concat([pd.DataFrame({"origin" : y_test}),
                           pd.DataFrame({"svm": clf_svm.predict(xs_test),
                                         "rf" : clf_rf.predict(xs_test),
                                         "dt" : clf_dt.predict(xs_test),
                                         "rl" : clf_rl.predict(xs_test)})], axis=1)
    result_df.to_csv("data/processed/result_{}.csv".format(datetime.now().strftime('%m%d')), index=None)

    # print score
    print('svm score : %.3f' % svm_score)
    print('rf score  : %.3f' % rf_score)
    print('dt score  : %.3f' % dt_score)
    print('rl score  : %.3f' % rl_score)

    logger = logging.getLogger(__name__)
    logger.info('result to test data(81 observation). output.csv was saved in /data/processed/.')


def main():
    predict_models()


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    main()
