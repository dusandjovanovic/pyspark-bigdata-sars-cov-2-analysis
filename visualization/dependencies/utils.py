import random
import dependencies.colors as color_scheme
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


def generate_custom_color(word, font_size, position, orientation, random_state=None, **kwargs):
    return random.choice(
        [color_scheme.color_50, color_scheme.color_100, color_scheme.color_200, color_scheme.color_300,
         color_scheme.color_400, color_scheme.color_500, color_scheme.color_600, color_scheme.color_700,
         color_scheme.color_800, color_scheme.color_900])
