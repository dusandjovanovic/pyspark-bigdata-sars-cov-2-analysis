from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
import pyspark.sql.functions as func
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import re

from dependencies.spark import start_spark


def main():
    spark, sql_context, log, config = start_spark(
        app_name='cases_patient_level_analysis',
        files=['configs/cases_patient_level_config.json'])

    log.warn('Running cases_patient_level analysis...')

    # extracting and transforming the dataset
    data = extract_data(spark)
    data_transformed = transform_data(data, sql_context)

    # gender value analysis
    data_transformed = transform_gender_values(data_transformed)
    load_data(data_transformed, "gender_distribution")

    # frequency of symptoms
    data_transformed = transform_symptoms_frequency(data_transformed)
    load_data(data_transformed, "symptoms_frequency")

    # relations between age and recovery/death
    data_transformed = transform_relations_age_recovery_death(data_transformed)
    load_data(data_transformed, "relations_age_recovery_death")

    log.warn('Terminating cases_patient_level analysis...')

    spark.stop()
    return None


def extract_data(spark):
    dataframe = spark.read.csv("../data/covid-19-patient-level-data/DXY.cn patient level data - Line-list.csv",
                               header=True)

    return dataframe


def transform_data(frame, sql_context):
    dt_transformed = frame
    dt_transformed = dt_transformed.fillna("NA")

    return dt_transformed


def transform_gender_values(dataframe):
    df_gender = dataframe.select("gender").toPandas()
    fig = px.histogram(x=df_gender["gender"], title="Gender distribution")
    fig.show()

    return dataframe


def transform_symptoms_frequency(dataframe):
    df_grouped = dataframe.groupBy("symptom").count().sort(func.col("count").desc())
    df_symptoms = df_grouped.take(17)[1:]

    symptoms = pd.DataFrame(df_symptoms)
    words = symptoms[0]
    weights = symptoms[1]
    word_cloud_data = go.Scatter(x=[4, 2, 2, 3, 1.5, 5, 4, 4, 0],
                                 y=[2, 2, 3, 3, 1, 5, 1, 3, 0],
                                 mode='text',
                                 text=words,
                                 marker={'opacity': 0.5},
                                 textfont={'size': weights,
                                           'color': ["red", "green", "blue", "purple", "black", "orange", "blue",
                                                     "black"]})
    layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                        'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})
    word_cloud = go.Figure(data=[word_cloud_data], layout=layout)
    word_cloud.update_layout(title_text='Word cloud of most common symptoms by frequency')
    word_cloud.show()

    return dataframe


def transform_relations_age_recovery_death(dataframe):
    udf_function = udf(is_date, StringType())

    dataframe = dataframe.withColumn("clean_recovered", udf_function("recovered"))
    dataframe = dataframe.withColumn("clean_death", udf_function("death"))

    clean_recovered = dataframe.select("clean_recovered").toPandas()["clean_recovered"]
    clean_death = dataframe.select("clean_death").toPandas()["clean_death"]
    age = dataframe.select("age").toPandas()["age"]

    rec_age_fig = make_subplots(rows=1, cols=2, subplot_titles=("Age vs. Recovered", "Age vs. Death"))
    rec_age_fig.add_trace(go.Box(x=clean_recovered, y=age, name="Recovered"), row=1, col=1)
    rec_age_fig.add_trace(go.Box(x=clean_death, y=age, name="Death"), row=1, col=2)
    rec_age_fig.update_traces(boxpoints='all')
    rec_age_fig.update_layout(title_text="Subplots of age in relation to recovery and death")
    rec_age_fig.show()

    return dataframe


def load_data(dataframe, name):
    (dataframe
     .coalesce(1)
     .write
     .csv(name, mode='overwrite', header=True))
    return None


def is_date(value):
    regex = re.compile(r'\d{1,2}/\d{1,2}/\d{4}')
    if bool(regex.match(value)):
        return '1'
    else:
        return value


if __name__ == '__main__':
    main()
