from dependencies.utils import load_dataset
from dependencies.dash import app, dash_content, dash_sidebar, dash_graph, dash_error
import dependencies.colors as color_scheme
from meta.cases_clinical_spectrum import analysis_name, analysis_description, analysis_options

import sys
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        return visualize_red_blood_cells_values()
    elif pathname == "/hemoglobin":
        return visualize_hemoglobin_values()
    elif pathname == "/aggregate":
        return visualize_aggregate_age_result()
    elif pathname == "/age_relations":
        return visualize_age_relations()
    elif pathname == "/regular_care":
        return visualize_care_relations_regular_care()
    elif pathname == "/intensive_care":
        return visualize_care_relations_intensive_care()
    elif pathname == "/missing_values":
        return visualize_predictions_missing_values()
    elif pathname == "/predictions_distribution":
        return visualize_predictions_value_distribution()
    elif pathname == "/predictions_test_distribution":
        return visualize_predictions_test_result_distribution()
    elif pathname == "/predictions":
        return visualize_predictions()
    return dash_error()


def visualize_red_blood_cells_values():
    dataframe_pd = load_dataset(sys.argv[1], "red_blood_cells_values")

    figure = px.histogram(
        dataframe_pd,
        x="Red blood Cells",
        title="Red blood Cells distribution",
        opacity=0.8,
        marginal="rug",
        color_discrete_sequence=[color_scheme.color_300]
    )

    return dash_graph(figure)


def visualize_hemoglobin_values():
    dataframe_pd = load_dataset(sys.argv[1], "hemoglobin_values")

    figure = px.histogram(
        dataframe_pd,
        x="Hemoglobin",
        title="Hemoglobin distribution",
        opacity=0.8,
        marginal="rug",
        color_discrete_sequence=[color_scheme.color_500]
    )

    return dash_graph(figure)


def visualize_aggregate_age_result():
    dataframe_pd = load_dataset(sys.argv[1], "aggregate_age_result")

    figure = px.line(
        dataframe_pd,
        x="result",
        y="avg(age)",
        log_y=True,
        title="Average age/result distribution",
        color_discrete_sequence=[color_scheme.color_400]
    )

    return dash_graph(figure)


def visualize_age_relations():
    dataframe_pd = load_dataset(sys.argv[1], "age_relations")
    dataframe_positive = dataframe_pd["positive"]
    dataframe_negative = dataframe_pd["negative"]
    dataframe_age = dataframe_pd["age"]

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Positive test/age coefficient correlation",
            "Negative test/age coefficient correlation"
        )
    )

    figure.add_trace(
        go.Box(
            x=dataframe_positive,
            y=dataframe_age,
            name="Positive",
            marker_color=color_scheme.color_300
        ),
        row=1,
        col=1
    )

    figure.add_trace(
        go.Box(
            x=dataframe_negative,
            y=dataframe_age,
            name="Negative",
            marker_color=color_scheme.color_700
        ),
        row=1,
        col=2
    )

    figure.update_traces(boxpoints='all')
    figure.update_layout(title_text="Subplots of age in relation a positive/negative test result")

    return dash_graph(figure)


def visualize_care_relations_regular_care():
    dataframe_pd = load_dataset(sys.argv[1], "care_relations")

    figure = px.bar(
        dataframe_pd,
        y="result",
        x="Patient addmited to regular ward (1=yes, 0=no)",
        title="Positive patients admitted to regular care",
        color_discrete_sequence=[color_scheme.color_400, color_scheme.color_500]
    )

    return dash_graph(figure)


def visualize_care_relations_intensive_care():
    dataframe_pd = load_dataset(sys.argv[1], "care_relations")

    figure = px.bar(
        dataframe_pd,
        y="result",
        x="Patient addmited to intensive care unit (1=yes, 0=no)",
        title="Positive patients admitted to intensive care",
        color_discrete_sequence=[color_scheme.color_900, color_scheme.color_500]
    )

    return dash_graph(figure)


def visualize_predictions_missing_values():
    dataframe_pd = load_dataset(sys.argv[1], "predictions_missing_values")
    dataframe_pd = dataframe_pd.rename(index={0: 'count'}).T.sort_values("count", ascending=False)

    figure = px.bar(
        dataframe_pd,
        y="count",
        title="Statistics of missing (null/nan) values across columns",
        color_discrete_sequence=[color_scheme.color_400, color_scheme.color_500]
    )

    return dash_graph(figure)


def visualize_predictions_value_distribution():
    dataframe_pd = load_dataset(sys.argv[1], "predictions_value_distribution")

    figure = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "Hemoglobin/Exam Result", "Platelets/Exam Result", "Eosinophils/Exam Result", "Red blood Cells/Exam Result",
            "Lymphocytes/Exam Result", "Leukocytes/Exam Result", "Basophils/Exam Result", "Monocytes/Exam Result",
            "Hematocrit/Exam Result"
        )
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_pd['SARS-Cov-2 exam result'],
            y=dataframe_pd["Hemoglobin"],
            mode='markers',
            marker=dict(color=color_scheme.color_900)
        ), row=1,
        col=1
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_pd['SARS-Cov-2 exam result'],
            y=dataframe_pd["Platelets"],
            mode='markers',
            marker=dict(color=color_scheme.color_800)
        ),
        row=1,
        col=2
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_pd['SARS-Cov-2 exam result'],
            y=dataframe_pd["Eosinophils"],
            mode='markers',
            marker=dict(color=color_scheme.color_700)
        ),
        row=1,
        col=3
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_pd['SARS-Cov-2 exam result'],
            y=dataframe_pd["Red blood Cells"],
            mode='markers',
            marker=dict(color=color_scheme.color_600)
        ),
        row=2,
        col=1
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_pd['SARS-Cov-2 exam result'],
            y=dataframe_pd["Lymphocytes"],
            mode='markers',
            marker=dict(color=color_scheme.color_500)
        ),
        row=2,
        col=2
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_pd['SARS-Cov-2 exam result'],
            y=dataframe_pd["Leukocytes"],
            mode='markers',
            marker=dict(color=color_scheme.color_400)
        ),
        row=2,
        col=3
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_pd['SARS-Cov-2 exam result'],
            y=dataframe_pd["Basophils"],
            mode='markers',
            marker=dict(color=color_scheme.color_300)
        ),
        row=3,
        col=1
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_pd['SARS-Cov-2 exam result'],
            y=dataframe_pd["Monocytes"],
            mode='markers',
            marker=dict(color=color_scheme.color_200)
        ),
        row=3,
        col=2
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_pd['SARS-Cov-2 exam result'],
            y=dataframe_pd["Hematocrit"],
            mode='markers',
            marker=dict(color=color_scheme.color_100)
        ),
        row=3,
        col=3
    )

    return dash_graph(figure)


def visualize_predictions_test_result_distribution():
    dataframe_pd = load_dataset(sys.argv[1], "predictions_test_result_distribution")

    figure = px.pie(
        dataframe_pd,
        values='count',
        names='result',
        title="Statistics of test result distribution",
        color_discrete_sequence=[color_scheme.color_100, color_scheme.color_400]
    )

    return dash_graph(figure)


def visualize_predictions():
    dataframe_pd = load_dataset(sys.argv[1], "predictions")

    figure = go.Figure(
        data=[
            go.Bar(
                y=dataframe_pd['value'],
                x=[
                    'Random Forest classifier Accuracy',
                    'Decision Tree Accuracy',
                    'Logistic Regression Accuracy',
                    'Gradient-boosted Trees Accuracy'
                ]
            )
        ]
    )

    figure.update_traces(
        opacity=0.6,
        marker_line_width=1.5,
        marker_color=color_scheme.color_200,
        marker_line_color=color_scheme.color_600,

    )

    figure.update_layout(title_text='Comparison of classifier accuracy reports')

    return dash_graph(figure)


if __name__ == '__main__':
    app.run_server()
