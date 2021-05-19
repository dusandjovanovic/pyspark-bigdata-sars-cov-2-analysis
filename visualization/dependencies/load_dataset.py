from os import listdir, path
import pandas as pd


def load_dataset(path_to_dir, subdir):
    path_output = path_to_dir + "/" + subdir + "/"
    filenames = find_csv_filenames(path_output)
    dataframe = pd.read_json(path.join(path_output, filenames[0]), lines=True)

    return dataframe


def find_csv_filenames(path_to_dir, suffix=".json"):
    filenames = listdir(path_to_dir)

    return [filename for filename in filenames if filename.endswith(suffix)]
