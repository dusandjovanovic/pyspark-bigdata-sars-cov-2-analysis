import random
import dependencies.colors as color_scheme


def get_options(analysis):
    dict_list = []
    for i in analysis:
        dict_list.append({'label': i, 'value': i})

    return dict_list


def generate_custom_color(word, font_size, position, orientation, random_state=None, **kwargs):
    return random.choice(
        [color_scheme.color_50, color_scheme.color_100, color_scheme.color_200, color_scheme.color_300,
         color_scheme.color_400, color_scheme.color_500, color_scheme.color_600, color_scheme.color_700,
         color_scheme.color_800, color_scheme.color_900])
