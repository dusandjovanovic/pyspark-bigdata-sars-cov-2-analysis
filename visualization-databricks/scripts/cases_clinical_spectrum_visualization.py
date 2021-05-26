import sys

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dependencies.load_dataset import load_dataset
from dependencies.load_spark import load_spark, get_db_utils
import dependencies.colors as color_scheme


def main():
    spark = load_spark()
    dbutils = get_db_utils(spark)

    visualize_hemoglobin_values(dbutils)
    visualize_red_blood_cells_values(dbutils)
    visualize_aggregate_age_result(dbutils)
    visualize_age_relations(dbutils)
    visualize_care_relations(dbutils)

    visualize_predictions_missing_values(dbutils)
    visualize_predictions_value_distribution(dbutils)
    visualize_predictions_test_result_distribution(dbutils)
    visualize_predictions(dbutils)

    return None


def visualize_hemoglobin_values(dbutils):
    dataframe_pd = load_dataset(sys.argv[1], "hemoglobin_values", dbutils)

    fig = px.histogram(dataframe_pd, x="Hemoglobin", title="Hemoglobin distribution",
                       color_discrete_sequence=[color_scheme.color_500], opacity=0.8, marginal="rug")
    fig.show()

    return None


def visualize_red_blood_cells_values(dbutils):
    dataframe_pd = load_dataset(sys.argv[1], "red_blood_cells_values", dbutils)

    fig = px.histogram(dataframe_pd, x="Red blood Cells", title="Red blood Cells distribution",
                       color_discrete_sequence=[color_scheme.color_300], opacity=0.8, marginal="rug")
    fig.show()

    return None


def visualize_aggregate_age_result(dbutils):
    dataframe_pd = load_dataset(sys.argv[1], "aggregate_age_result", dbutils)

    fig = px.line(dataframe_pd, x="result", y="avg(age)", title="Average age/result distribution",
                  log_y=True, color_discrete_sequence=[color_scheme.color_400])
    fig.show()

    return None


def visualize_age_relations(dbutils):
    dataframe_pd = load_dataset(sys.argv[1], "age_relations", dbutils)

    display_positive = dataframe_pd["positive"]
    display_negative = dataframe_pd["negative"]
    display_age = dataframe_pd["age"]

    rec_age_fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Positive test/age coefficient correlation",
                                                "Negative test/age coefficient correlation"))
    rec_age_fig.add_trace(
        go.Box(x=display_positive, y=display_age, name="Positive", marker_color=color_scheme.color_300), row=1, col=1)
    rec_age_fig.add_trace(
        go.Box(x=display_negative, y=display_age, name="Negative", marker_color=color_scheme.color_700), row=1, col=2)
    rec_age_fig.update_traces(boxpoints='all')
    rec_age_fig.update_layout(title_text="Subplots of age in relation a positive/negative test result")
    rec_age_fig.show()

    return None


def visualize_care_relations(dbutils):
    dataframe_pd = load_dataset(sys.argv[1], "care_relations", dbutils)

    fig = px.bar(dataframe_pd, y="result", x="Patient addmited to regular ward (1=yes, 0=no)",
                 color_discrete_sequence=[color_scheme.color_400, color_scheme.color_500],
                 title="Positive patients admitted to regular care")
    fig.show()

    fig_intensive = px.bar(dataframe_pd, y="result",
                           x="Patient addmited to intensive care unit (1=yes, 0=no)",
                           color_discrete_sequence=[color_scheme.color_900, color_scheme.color_500],
                           title="Positive patients admitted to intensive care")
    fig_intensive.show()

    return None


def visualize_predictions_missing_values(dbutils):
    dataframe_pd = load_dataset(sys.argv[1], "predictions_missing_values", dbutils)
    dataframe_pd = dataframe_pd.rename(index={0: 'count'}).T.sort_values("count", ascending=False)

    fig = px.bar(dataframe_pd, y="count",
                 color_discrete_sequence=[color_scheme.color_400, color_scheme.color_500],
                 title="Statistics of missing (null/nan) values across columns")
    fig.show()

    return None


def visualize_predictions_value_distribution(dbutils):
    dataframe_pd = load_dataset(sys.argv[1], "predictions_value_distribution", dbutils)

    fig = make_subplots(rows=3, cols=3, subplot_titles=(
        "Hemoglobin/Exam Result", "Platelets/Exam Result", "Eosinophils/Exam Result", "Red blood Cells/Exam Result",
        "Lymphocytes/Exam Result", "Leukocytes/Exam Result", "Basophils/Exam Result", "Monocytes/Exam Result",
        "Hematocrit/Exam Result"))

    fig.add_trace(
        go.Scatter(x=dataframe_pd['SARS-Cov-2 exam result'], y=dataframe_pd["Hemoglobin"], mode='markers',
                   marker=dict(color=color_scheme.color_900)), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=dataframe_pd['SARS-Cov-2 exam result'], y=dataframe_pd["Platelets"], mode='markers',
                   marker=dict(color=color_scheme.color_800)), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=dataframe_pd['SARS-Cov-2 exam result'], y=dataframe_pd["Eosinophils"], mode='markers',
                   marker=dict(color=color_scheme.color_700)), row=1, col=3)
    fig.add_trace(
        go.Scatter(x=dataframe_pd['SARS-Cov-2 exam result'], y=dataframe_pd["Red blood Cells"], mode='markers',
                   marker=dict(color=color_scheme.color_600)), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=dataframe_pd['SARS-Cov-2 exam result'], y=dataframe_pd["Lymphocytes"], mode='markers',
                   marker=dict(color=color_scheme.color_500)), row=2, col=2)
    fig.add_trace(
        go.Scatter(x=dataframe_pd['SARS-Cov-2 exam result'], y=dataframe_pd["Leukocytes"], mode='markers',
                   marker=dict(color=color_scheme.color_400)), row=2, col=3)
    fig.add_trace(
        go.Scatter(x=dataframe_pd['SARS-Cov-2 exam result'], y=dataframe_pd["Basophils"], mode='markers',
                   marker=dict(color=color_scheme.color_300)), row=3, col=1)
    fig.add_trace(
        go.Scatter(x=dataframe_pd['SARS-Cov-2 exam result'], y=dataframe_pd["Monocytes"], mode='markers',
                   marker=dict(color=color_scheme.color_200)), row=3, col=2)
    fig.add_trace(
        go.Scatter(x=dataframe_pd['SARS-Cov-2 exam result'], y=dataframe_pd["Hematocrit"], mode='markers',
                   marker=dict(color=color_scheme.color_100)), row=3, col=3)

    fig.show()

    return None


def visualize_predictions_test_result_distribution(dbutils):
    dataframe_pd = load_dataset(sys.argv[1], "predictions_test_result_distribution", dbutils)

    fig = px.pie(dataframe_pd, values='count', names='result',
                 title="Statistics of test result distribution",
                 color_discrete_sequence=[color_scheme.color_100, color_scheme.color_400])
    fig.show()

    return None


def visualize_predictions(dbutils):
    dataframe_pd = load_dataset(sys.argv[1], "predictions", dbutils)

    fig = go.Figure(data=[go.Bar(y=dataframe_pd['value'],
                                 x=['Random Forest classifier Accuracy', 'Decision Tree Accuracy',
                                    'Logistic Regression Accuracy', 'Gradient-boosted Trees Accuracy'])])
    fig.update_traces(marker_color=color_scheme.color_200, marker_line_color=color_scheme.color_600,
                      marker_line_width=1.5, opacity=0.6)
    fig.update_layout(title_text='Comparison of classifier accuracy reports')
    fig.show()

    return None


if __name__ == '__main__':
    main()
