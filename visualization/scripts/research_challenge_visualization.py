from dependencies.utils import generate_custom_color, load_dataset
from dependencies.dash import app, dash_content, dash_sidebar, dash_graph, dash_error
import dependencies.colors as color_scheme
from meta.research_challenge import analysis_name, analysis_description, analysis_options

import sys
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import plotly.figure_factory as ff

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dash_sidebar(analysis_name, analysis_description, analysis_options),
        dash_content()
    ]
)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return visualize_sentiment_analysis()
    elif pathname == "/words":
        return visualize_most_commonly_used_words()
    return dash_error()


def visualize_sentiment_analysis():
    dataframe_pd = load_dataset(sys.argv[1], "paper_abstracts")

    figure = ff.create_distplot(
        [dataframe_pd["sentiment_abstract"]],
        ["sentiment_abstract"],
        colors=[color_scheme.color_400]
    )

    figure.update_layout(title='Sentiment value distribution based on all article excerpts')

    return dash_graph(figure)


def visualize_most_commonly_used_words():
    dataframe_pd = load_dataset(sys.argv[1], "paper_abstracts")
    text = dataframe_pd["clean_abstract"].values
    stopwords = set(STOPWORDS)

    word_cloud = WordCloud(
        width=1000,
        height=500,
        stopwords=stopwords,
        background_color="white",
        max_words=25
    ).generate(str(text))

    figure = px.imshow(
        word_cloud.recolor(
            color_func=generate_custom_color,
            random_state=3
        ),
        binary_compression_level=1,
        title="Most commonly used words in abstracts"
    )

    return dash_graph(figure)


if __name__ == '__main__':
    app.run_server()
