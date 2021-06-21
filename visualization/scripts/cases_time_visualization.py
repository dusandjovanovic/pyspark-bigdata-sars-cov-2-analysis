from dependencies.utils import load_dataset
from dependencies.dash import app, dash_content, dash_sidebar, dash_graph, dash_error
import dependencies.colors as color_scheme
from meta.cases_time_analysis import analysis_name, analysis_description, analysis_options

import sys
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

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
        return visualize_confirmed_cases_globally()
    elif pathname == "/deaths":
        return visualize_deaths_globally()
    elif pathname == "/countries":
        return visualize_confirmed_cases_countries()
    elif pathname == "/europe_heat_map":
        return visualize_confirmed_cases_europe_heat_map()
    elif pathname == "/europe_countries":
        return visualize_confirmed_cases_europe_countries()
    elif pathname == "/comparison":
        return visualize_confirmed_cases_comparison()
    elif pathname == "/mortality_rates":
        return visualize_confirmed_cases_mortality_rates()
    elif pathname == "/recovery_rates":
        return visualize_confirmed_cases_recovery_rates()
    elif pathname == "/future_predictions":
        return visualize_future_predictions()
    elif pathname == "/future_forecasting":
        return visualize_future_forecasting()
    return dash_error()


def visualize_confirmed_cases_globally():
    dataframe_pd = load_dataset(sys.argv[1], "confirmed_cases_and_deaths_globally")

    figure = px.line(
        dataframe_pd,
        x="date",
        y="sum(confirmed)",
        title="Confirmed cases over time (logarithmic)",
        log_y=True,
        color_discrete_sequence=[color_scheme.color_400]
    )

    return dash_graph(figure)


def visualize_deaths_globally():
    dataframe_pd = load_dataset(sys.argv[1], "confirmed_cases_and_deaths_globally")

    figure = px.line(
        dataframe_pd,
        x="date",
        y="sum(deaths)",
        title="Death cases over time (logarithmic)",
        log_y=True,
        color_discrete_sequence=[color_scheme.color_900]
    )

    return dash_graph(figure)


def visualize_confirmed_cases_countries():
    df_serbia_grouped = load_dataset(sys.argv[1], "confirmed_cases_serbia")
    df_china_grouped = load_dataset(sys.argv[1], "confirmed_cases_china")
    df_italy_grouped = load_dataset(sys.argv[1], "confirmed_cases_italy")
    df_norway_grouped = load_dataset(sys.argv[1], "confirmed_cases_norway")

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=df_serbia_grouped['date'],
            y=df_serbia_grouped['sum(confirmed)'],
            name="Serbia",
            line_color=color_scheme.color_500,
            opacity=0.8
        )
    )

    figure.add_trace(
        go.Scatter(
            x=df_china_grouped['date'],
            y=df_china_grouped['sum(confirmed)'],
            name="China",
            line_color=color_scheme.color_700,
            opacity=0.8
        )
    )

    figure.add_trace(
        go.Scatter(
            x=df_italy_grouped['date'],
            y=df_italy_grouped['sum(confirmed)'],
            name="Italy",
            line_color=color_scheme.color_900,
            opacity=0.8
        )
    )

    figure.add_trace(
        go.Scatter(
            x=df_norway_grouped['date'],
            y=df_norway_grouped['sum(confirmed)'],
            name="Norway",
            line_color=color_scheme.color_300,
            opacity=0.8
        )
    )

    figure.update_layout(title_text="Overview of case growth in Serbia, China, Italy and Norway")

    return dash_graph(figure)


def visualize_confirmed_cases_europe_heat_map():
    dataframe_pd = load_dataset(sys.argv[1], "confirmed_cases_europe")

    figure = px.choropleth(
        dataframe_pd,
        locations="country",
        locationmode='country names',
        color="sum(confirmed)",
        hover_name="country",
        range_color=[1, 1000000],
        color_continuous_scale='portland',
        title='European countries with confirmed cases',
        scope='europe',
        height=800
    )

    return dash_graph(figure)


def visualize_confirmed_cases_europe_countries():
    dataframe_pd = load_dataset(sys.argv[1], "confirmed_cases_europe")
    dataframe_sorted = dataframe_pd.sort_values('sum(confirmed)', ascending=False)[:10][::-1];

    figure = px.bar(
        dataframe_sorted,
        x='sum(confirmed)',
        y='country',
        color_discrete_sequence=[color_scheme.color_400],
        title='Confirmed cases in Europe (top-10 countries)',
        text='sum(confirmed)',
        orientation='h'
    )

    return dash_graph(figure)


def visualize_confirmed_cases_comparison():
    dataframe_pd = load_dataset(sys.argv[1], "confirmed_cases_comparison")
    dataframe_melted = dataframe_pd.melt(
        id_vars="date",
        value_vars=['sum(recovered)', 'sum(deaths)', 'sum(active)'],
        var_name='case', value_name='count'
    )

    figure = px.area(
        dataframe_melted,
        x="date",
        y="count",
        color='case',
        title='Cases over time',
        color_discrete_sequence=[color_scheme.color_200, color_scheme.color_400, color_scheme.color_800]
    )

    return dash_graph(figure)


def visualize_confirmed_cases_mortality_rates():
    dataframe_pd = load_dataset(sys.argv[1], "confirmed_cases_mortality_rates")

    figure = px.bar(
        dataframe_pd,
        x='mortalityRate',
        y='country',
        title='Deaths per 100 confirmed cases (top-10)',
        text='mortalityRate',
        height=800,
        orientation='h',
        color_discrete_sequence=[color_scheme.color_600]
    )

    return dash_graph(figure)


def visualize_confirmed_cases_recovery_rates():
    dataframe_pd = load_dataset(sys.argv[1], "confirmed_cases_recovery_rates")

    figure = px.bar(
        dataframe_pd,
        x='recoveryRate',
        y='country',
        title='Recoveries per 100 confirmed cases (top-10)',
        text='recoveryRate',
        height=800,
        orientation='h',
        color_discrete_sequence=[color_scheme.color_500]
    )

    return dash_graph(figure)


def visualize_future_predictions():
    dataframe_predictions = load_dataset(sys.argv[1], "future_predictions")
    dataframe_time_series = load_dataset(sys.argv[1], "time_series_test_data")
    dataframe_predictions = dataframe_predictions.set_index('ds')
    dataframe_time_series = dataframe_time_series.set_index('ds')

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=dataframe_time_series.index,
            y=dataframe_time_series.y,
            name="Actual Cases",
            line_color=color_scheme.color_400,
            mode='lines',
            opacity=0.8
        )
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_predictions.index,
            y=dataframe_predictions.yhat,
            name="Prediction",
            mode='lines',
            line_color=color_scheme.color_alt,
            opacity=0.8
        )
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_predictions.index,
            y=dataframe_predictions.yhat_lower,
            name="Prediction Lower Bound",
            mode='lines',
            line=dict(color=color_scheme.color_200, width=2, dash='dash'),
            opacity=0.8
        )
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_predictions.index,
            y=dataframe_predictions.yhat_upper,
            name="Prediction Upper Bound",
            mode='lines',
            line=dict(color=color_scheme.color_200, width=2, dash='dash'),
            opacity=0.8
        )
    )

    figure.update_layout(title_text="Model's Test Prediction", xaxis_title="Date", yaxis_title="Cases")

    return dash_graph(figure)


def visualize_future_forecasting():
    dataframe_forecasting = load_dataset(sys.argv[1], "future_forecasting")
    dataframe_time_series = load_dataset(sys.argv[1], "time_series_by_countries")
    dataframe_forecasting = dataframe_forecasting.set_index('ds')

    dataframe_forecasting_serbia = dataframe_forecasting.loc[
        dataframe_forecasting['country'] == "Serbia"
        ]

    dataframe_forecasting_croatia = dataframe_forecasting.loc[
        dataframe_forecasting['country'] == "Croatia"
        ]

    dataframe_forecasting_slovenia = dataframe_forecasting.loc[
        dataframe_forecasting['country'] == "Slovenia"
        ]

    dataframe_forecasting_montenegro = dataframe_forecasting.loc[
        dataframe_forecasting['country'] == "Montenegro"
        ]

    dataframe_time_series_serbia = dataframe_time_series.loc[
        dataframe_time_series['country'] == "Serbia"
        ]

    dataframe_time_series_croatia = dataframe_time_series.loc[
        dataframe_time_series['country'] == "Croatia"
        ]

    dataframe_time_series_slovenia = dataframe_time_series.loc[
        dataframe_time_series['country'] == "Slovenia"
        ]

    dataframe_time_series_montenegro = dataframe_time_series.loc[
        dataframe_time_series['country'] == "Montenegro"
        ]

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=dataframe_time_series_serbia.date,
            y=dataframe_time_series_serbia.confirmed,
            name="Actual (Serbia)",
            line_color=color_scheme.color_200,
            opacity=0.8
        )
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_forecasting_serbia.index,
            y=dataframe_forecasting_serbia.yhat,
            name="Prediction (Serbia)",
            line_color=color_scheme.color_600,
            opacity=0.8
        )
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_time_series_croatia.date,
            y=dataframe_time_series_croatia.confirmed,
            name="Actual (Croatia)",
            line_color=color_scheme.color_200,
            opacity=0.8
        )
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_forecasting_croatia.index,
            y=dataframe_forecasting_croatia.yhat,
            name="Prediction (Croatia)",
            line_color=color_scheme.color_600,
            opacity=0.8
        )
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_time_series_slovenia.date,
            y=dataframe_time_series_slovenia.confirmed,
            name="Actual (Slovenia)",
            line_color=color_scheme.secondary_200,
            opacity=0.8
        )
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_forecasting_slovenia.index,
            y=dataframe_forecasting_slovenia.yhat,
            name="Prediction (Slovenia)",
            line_color=color_scheme.secondary_600,
            opacity=0.8
        )
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_time_series_montenegro.date,
            y=dataframe_time_series_montenegro.confirmed,
            name="Actual (Montenegro)",
            line_color=color_scheme.tertiary_200,
            opacity=0.8
        )
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_forecasting_montenegro.index,
            y=dataframe_forecasting_montenegro.yhat,
            name="Prediction (Montenegro)",
            line_color=color_scheme.tertiary_600,
            opacity=0.8
        )
    )

    figure.update_layout(title_text="Model Forecasting", xaxis_title="Date", yaxis_title="Cases")

    return dash_graph(figure)


if __name__ == '__main__':
    app.run_server()
