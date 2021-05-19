import __main__

from os import listdir
import pandas as pd


def load_dataset(path_to_dir):
    filenames = find_csv_filenames(path_to_dir)
    dataframe = pd.read_csv(os.path.join(path_to_dir, filenames[0]))

    return dataframe


def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)

    return [filename for filename in filenames if filename.endswith(suffix)]
