from dependencies.load_dataset import load_dataset
from dependencies.utils import get_options, generate_custom_color
import dependencies.colors as color_scheme

import sys
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import plotly.figure_factory as ff

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

analysis_options = ["Sentiment analysis", "Most commonly used words"]

analysis_name = "CORD-19 Open Research Challenge"

analysis_description = "In response to the COVID-19 pandemic, the White House and a coalition of leading research groups " \
                       "have prepared the COVID-19 Open Research Dataset (CORD-19). CORD-19 is a resource of over 500," \
                       "000 scholarly articles about COVID-19, SARS-CoV-2, and related coronaviruses. "

app.layout = html.Div(
    children=[
        dbc.Row(
            className='justify-content-center',
            align="center",
            children=[
                dbc.Col(md=4,
                        children=[
                            html.H3(analysis_name, className="d-flex pt-8 pl-4 pb-4"),
                            html.P(analysis_description, className="d-flex pl-4 pb-2"),
                            html.Div(
                                children=[
                                    dcc.Dropdown(
                                        id='analysis_selection',
                                        options=get_options(analysis_options),
                                        value=analysis_options[0],
                                        multi=False, clearable=False,
                                        className="d-flex pl-4 pb-2"
                                    ),
                                ],
                                style={'color': '#1E1E1E'})
                        ]
                        ),
                dbc.Col(md=8,
                        children=[
                            dcc.Graph(
                                id='analysis_graph',
                                config={'displayModeBar': False},
                                animate=True
                            )
                        ])
            ])
    ]
)


@app.callback(Output('analysis_graph', 'figure'),
              [Input('analysis_selection', 'value')])
def update_graph(selected_dropdown_value):
    if selected_dropdown_value == analysis_options[0]:
        return visualize_sentiment_analysis()
    elif selected_dropdown_value == analysis_options[1]:
        return visualize_most_commonly_used_words()

    return None


def visualize_sentiment_analysis():
    dataframe_pd = load_dataset(sys.argv[1], "paper_abstracts")

    return ff.create_distplot([dataframe_pd["sentiment_abstract"]], ["sentiment_abstract"],
                              colors=[color_scheme.color_400])


def visualize_most_commonly_used_words():
    dataframe_pd = load_dataset(sys.argv[1], "paper_abstracts")
    text = dataframe_pd["clean_abstract"].values
    stopwords = set(STOPWORDS)

    word_cloud = WordCloud(width=1000, height=500, stopwords=stopwords, background_color="white",
                           max_words=25).generate(str(text))

    return px.imshow(word_cloud.recolor(color_func=generate_custom_color, random_state=3), binary_compression_level=1,
                     title="Most commonly used words in abstracts")


if __name__ == '__main__':
    app.run_server(debug=True)
