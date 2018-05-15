import click
import unittest
import pandas as pd
from click.testing import CliRunner
from .make_dataset import process_raw_df
from pandas.util.testing import assert_frame_equal


class MakeDataset(unittest.TestCase):
    """test class of make_dataset.py
    """

    def test_process_raw_df(self):
        """test method for process_raw_df
           test about row and column number of processed data.
        """

        expected_shape = "(891, 6)\n"

        runner = CliRunner()
        actual_shape = runner.invoke(process_raw_df, ['train.csv', "processed_df.csv"]).output

        self.assertEqual(expected_shape, actual_shape)


if __name__ == "__main__":
    unittest.main()
