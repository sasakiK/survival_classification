import click
import unittest
import pandas as pd
from click.testing import CliRunner
from .train_model import train_models
from .predict_model import predict_models
from pandas.util.testing import assert_frame_equal


class TrainModel(unittest.TestCase):
    """test class of train_models.py
    """

    def test_train_models(self):
        """test method for process_raw_df
           test about row and column number of processed data.
        """

        expected_score =  0.7

        runner = CliRunner()
        actual_score = runner.invoke(train_models, ['processed_df.csv']).output

        # get each score
        spl = actual_score.split("\n")
        output_res = []
        for output in spl[0:4]:
            output_res.append(float(output.split(":")[1]))

        # test that output result score is all > 0.8
        self.assertTrue(all(elem > expected_score for elem in output_res))


class PredictModel(unittest.TestCase):
    """test class of predict_model.py
    """

    def test_predict_models(self):
        """test method for process_raw_df
           test about row and column number of processed data.
        """

        expected_score =  0.7

        runner = CliRunner()
        actual_score = runner.invoke(predict_models, ['processed_df.csv']).output

        # get each score
        spl = actual_score.split("\n")
        output_res = []
        for output in spl[0:4]:
            output_res.append(float(output.split(":")[1]))

        # test that output result score is all > 0.8
        self.assertTrue(all(elem > expected_score for elem in output_res))


if __name__ == "__main__":
    unittest.main()
