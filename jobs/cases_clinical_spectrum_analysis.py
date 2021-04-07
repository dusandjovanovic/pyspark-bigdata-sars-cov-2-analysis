from IPython.core.display import display
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as func
import plotly.express as px
import pandas as pd

from dependencies.spark import start_spark


def main():
    spark, sql_context, log, config = start_spark(
        app_name='cases_clinical_spectrum_analysis',
        files=['configs/cases_clinical_spectrum_config.json'])

    log.warn('Running cases_clinical_spectrum analysis...')

    # extracting and transforming the dataset
    data = extract_data()
    data_transformed = transform_data(data, sql_context)

    # hemoglobin values analysis
    data_transformed = transform_hemoglobin_values(data_transformed)
    load_data(data_transformed, "hemoglobin_distribution")

    # average age/result aggregated distribution analysis
    data_transformed = transform_aggregate(data_transformed, sql_context)
    load_data(data_transformed, "age_result_distribution")

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


def transform_hemoglobin_values(dataframe):
    df_hemoglobin = dataframe.select("Hemoglobin").toPandas()
    fig = px.histogram(x=df_hemoglobin['Hemoglobin'], title="Hemoglobin distribution")
    fig.show()

    return dataframe


def transform_aggregate(dataframe, sql_context):
    df_age_select = dataframe.select(func.col("SARS-Cov-2 exam result").alias("result"),
                                     func.col('Patient age quantile').alias('age'))

    df_age_select.write.mode('overwrite').option("header", "true").save("result_age.parquet",
                                                                        format="parquet")

    df_sql = sql_context.sql("SELECT * FROM parquet.`./result_age.parquet`")
    df_aggregate = df_sql.groupBy("result").agg(func.max("age"), func.avg("age")).toPandas()
    fig = px.line(df_aggregate, x="result", y="avg(age)", title="Average age/result distribution",
                  log_y=True, color_discrete_sequence=['#F42272'])
    display(df_aggregate)
    fig.show()

    return dataframe


def load_data(dataframe, name):
    (dataframe
     .coalesce(1)
     .write
     .csv(name, mode='overwrite', header=True))
    return None


if __name__ == '__main__':
    main()
