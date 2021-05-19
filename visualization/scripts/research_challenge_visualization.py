import sys
import random

from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import plotly.figure_factory as ff

from dependencies.load_dataset import load_dataset
import dependencies.colors as color_scheme


def main():
    visualize_abstracts_words()

    return None


def visualize_abstracts_words():
    dataframe_pd = load_dataset(sys.argv[1], "paper_abstracts")

    fig = ff.create_distplot([dataframe_pd["sentiment_abstract"]], ["sentiment_abstract"],
                             colors=[color_scheme.color_400])
    fig.show()

    text = dataframe_pd["clean_abstract"].values
    stopwords = set(STOPWORDS)
    word_cloud = WordCloud(width=1000, height=500, stopwords=stopwords, background_color="white",
                           max_words=25).generate(
        str(text))
    fig = px.imshow(word_cloud.recolor(color_func=generate_custom_color, random_state=3), binary_compression_level=1,
                    title="Most commonly used words in abstracts")
    fig.show()

    return None


def generate_custom_color(word, font_size, position, orientation, random_state=None, **kwargs):
    return random.choice(
        [color_scheme.color_50, color_scheme.color_100, color_scheme.color_200, color_scheme.color_300,
         color_scheme.color_400, color_scheme.color_500, color_scheme.color_600, color_scheme.color_700,
         color_scheme.color_800, color_scheme.color_900])


if __name__ == '__main__':
    main()
