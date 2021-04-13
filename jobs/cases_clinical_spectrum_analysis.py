from plotly.subplots import make_subplots
from pyspark.sql.types import IntegerType, StringType
import pyspark.sql.functions as func
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import shared_modules

from dependencies.spark import start_spark


def main():
    spark, sql_context, log, config = start_spark(
        app_name='cases_clinical_spectrum_analysis',
        files=['configs/cases_clinical_spectrum_config.json'])

    log.warn('Running cases_clinical_spectrum analysis...')

    # extracting and transforming the dataset
    data = extract_data()
    data_transformed = transform_data(data, sql_context)

    # hemoglobin/eosinophils values analysis
    data_transformed = transform_hemoglobin_red_blood_cells_values(data_transformed)
    load_data(data_transformed, "hemoglobin_red_blood_cells_values")

    # average age/result aggregated distribution analysis
    data_transformed = transform_aggregate(data_transformed, sql_context)
    load_data(data_transformed, "age_result_distribution")

    # age relations to test outcomes
    data_transformed = transform_age_relations(data_transformed, sql_context)
    load_data(data_transformed, "age_relations")

    # care type admition relations of positive patients
    data_transformed = transform_care_relations(data_transformed, sql_context)
    load_data(data_transformed, "care_relations")

    log.warn('Terminating cases_clinical_spectrum analysis...')

    spark.stop()
    return None


def extract_data():
    dataframe = pd.read_excel('../data/covid-19-clinical-spectrum/dataset.xlsx')

    return dataframe


def transform_data(frame, sql_context):
    dt_transformed = frame

    dt_transformed['Respiratory Syncytial Virus'] = dt_transformed['Respiratory Syncytial Virus'].astype(str)
    dt_transformed['Influenza A'] = dt_transformed['Influenza A'].astype(str)
    dt_transformed['Influenza B'] = dt_transformed['Influenza B'].astype(str)
    dt_transformed['Parainfluenza 1'] = dt_transformed['Parainfluenza 1'].astype(str)
    dt_transformed['CoronavirusNL63'] = dt_transformed['CoronavirusNL63'].astype(str)
    dt_transformed['Rhinovirus/Enterovirus'] = dt_transformed['Rhinovirus/Enterovirus'].astype(str)
    dt_transformed['Coronavirus HKU1'] = dt_transformed['Coronavirus HKU1'].astype(str)

    for column in dt_transformed.columns:
        dt_transformed[column] = dt_transformed[column].astype(str)

    dataframe = sql_context.createDataFrame(dt_transformed)

    dataframe = dataframe.fillna(0)
    dataframe = dataframe.replace("nan", "0")
    dataframe = dataframe.withColumn("Hemoglobin", dataframe["Hemoglobin"].cast(IntegerType()))

    return dataframe


def transform_hemoglobin_red_blood_cells_values(dataframe):
    df_hemoglobin = dataframe.select("Hemoglobin").toPandas()
    df_eosinophils = dataframe.select("Red blood Cells").withColumn("Red blood Cells",
                                                                    func.round(dataframe["Red blood Cells"],
                                                                               2)).toPandas()

    fig = px.histogram(df_hemoglobin, x="Hemoglobin", title="Hemoglobin distribution",
                       color_discrete_sequence=[shared_modules.color_500], opacity=0.8, marginal="rug")
    fig.show()

    fig = px.histogram(df_eosinophils, x="Red blood Cells", title="Red blood Cells distribution",
                       color_discrete_sequence=[shared_modules.color_300], opacity=0.8, marginal="rug")
    fig.show()

    return dataframe


def transform_aggregate(dataframe, sql_context):
    df_age_select = dataframe.select(func.col("SARS-Cov-2 exam result").alias("result"),
                                     func.col('Patient age quantile').alias('age'))

    df_age_select.write.mode('overwrite').option("header", "true").save("temporary.parquet",
                                                                        format="parquet")

    df_sql = sql_context.sql("SELECT * FROM parquet.`./temporary.parquet`")
    df_aggregate = df_sql.groupBy("result").agg(func.max("age"), func.avg("age")).toPandas()
    fig = px.line(df_aggregate, x="result", y="avg(age)", title="Average age/result distribution",
                  log_y=True, color_discrete_sequence=[shared_modules.color_400])
    fig.show()

    return dataframe


def transform_age_relations(dataframe, sql_context):
    udf_function_positive = func.udf(is_positive, StringType())
    udf_function_negative = func.udf(is_negative, StringType())

    df_age = dataframe.select(func.col("SARS-Cov-2 exam result").alias("result"),
                              func.col('Patient age quantile').alias('age'))

    df_age_positive = df_age.withColumn("positive", udf_function_positive("result"))
    df_age_negative = df_age.withColumn("negative", udf_function_negative("result"))

    display_positive = df_age_positive.select("positive").toPandas()["positive"]
    display_negative = df_age_negative.select("negative").toPandas()["negative"]
    display_age = df_age.select("age").toPandas()["age"]

    rec_age_fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Age and Positive/Negative correlation", "Positive/Negative"))
    rec_age_fig.add_trace(go.Box(x=display_positive, y=display_age, name="Positive"), row=1, col=1)
    rec_age_fig.add_trace(go.Box(x=display_negative, y=display_age, name="Negative"), row=1, col=2)
    rec_age_fig.update_traces(boxpoints='all')
    rec_age_fig.update_layout(title_text="Subplots of age in relation a positive/negative test result")
    rec_age_fig.show()

    return dataframe


def transform_care_relations(dataframe, sql_context):
    udf_function_to_numeric = func.udf(negative_positive_to_numeric, IntegerType())

    df_transformed_numeric = dataframe.withColumn("result", udf_function_to_numeric("SARS-Cov-2 exam result"))
    df_transformed_positive = df_transformed_numeric.filter(df_transformed_numeric.result == 1)
    df_transformed_positive_display = df_transformed_positive.toPandas()

    fig = px.bar(df_transformed_positive_display, y="result", x="Patient addmited to regular ward (1=yes, 0=no)",
                 color_discrete_sequence=[shared_modules.color_400, shared_modules.color_500],
                 title="Positive patients admited to regular care")
    fig.show()

    fig_intensive = px.bar(df_transformed_positive_display, y="result",
                           x="Patient addmited to intensive care unit (1=yes, 0=no)",
                           color_discrete_sequence=[shared_modules.color_900, shared_modules.color_500],
                           title="Positive patients admited to intensive care")
    fig_intensive.show()

    return dataframe


def is_positive(value):
    if value == 'positive':
        return '1'
    else:
        return '0'


def is_negative(value):
    if value == 'negative':
        return '1'
    else:
        return '0'


def negative_positive_to_numeric(value):
    if value == 'negative':
        return 0
    else:
        return 1


def load_data(dataframe, name):
    (dataframe
     .coalesce(1)
     .write
     .csv("./outputs/cases_clinical_spectrum_analysis/" + name, mode='overwrite', header=True))
    return None


if __name__ == '__main__':
    main()
