import sys

from pyspark.sql import Window
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as func

from prophet import Prophet
import pandas as pd
import numpy as np

from dependencies.spark import start_spark


def main():
    spark, sql_context, log, config = start_spark(
        app_name='cases_time_analysis',
        files=['configs/cases_time_analysis_config.json'])

    log.warn('Running cases_time analysis...')

    # extracting and transforming the dataset
    data = extract_data(spark)
    data_initial = transform_data(data, sql_context)

    # confirmed cases and deaths globally
    data_transformed = transform_confirmed_cases_and_deaths_globally(data_initial)
    load_data(data_transformed, "confirmed_cases_and_deaths_globally")

    # confirmed cases by countries
    data_transformed = transform_confirmed_cases_serbia(data_initial)
    load_data(data_transformed, "confirmed_cases_serbia")

    data_transformed = transform_confirmed_cases_norway(data_initial)
    load_data(data_transformed, "confirmed_cases_norway")

    data_transformed = transform_confirmed_cases_italy(data_initial)
    load_data(data_transformed, "confirmed_cases_italy")

    data_transformed = transform_confirmed_cases_china(data_initial)
    load_data(data_transformed, "confirmed_cases_china")

    # confirmed cases in Europe
    data_transformed = transform_confirmed_cases_europe(data_initial)
    load_data(data_transformed, "confirmed_cases_europe")

    # confirmed cases comparison
    data_transformed = transform_confirmed_cases_comparison(data_initial)
    load_data(data_transformed, "confirmed_cases_comparison")

    # confirmed cases comparison by mortality rate
    data_transformed = transform_confirmed_cases_mortality_rates(data_initial)
    load_data(data_transformed, "confirmed_cases_mortality_rates")

    # confirmed cases comparison by recovery rate
    data_transformed = transform_confirmed_cases_recovery_rates(data_initial)
    load_data(data_transformed, "confirmed_cases_recovery_rates")

    # time series
    data_transformed = transform_time_series(data_initial)
    load_data(data_transformed, "time_series")

    # time series test data
    data_transformed = transform_time_series_test_data(data_initial, spark)
    load_data(data_transformed, "time_series_test_data")

    # time series predictions
    data_transformed = transform_time_series_predictions(data_initial, spark)
    load_data(data_transformed, "future_predictions")

    # time series forecasting
    data_transformed = transform_time_series_forecasting(data_initial, spark)
    load_data(data_transformed, "future_forecasting")

    log.warn('Terminating cases_time analysis...')

    spark.stop()
    return None


def extract_data(spark):
    dataframe = spark.read.csv(sys.argv[1], header=True, dateFormat="yyyy-MM-dd")

    return dataframe


def transform_data(frame, sql_context):
    dt_transformed = frame

    dt_transformed = dt_transformed.withColumnRenamed("ObservationDate", "date")
    dt_transformed = dt_transformed.withColumnRenamed("Province/State", "state")
    dt_transformed = dt_transformed.withColumnRenamed("Country/Region", "country")
    dt_transformed = dt_transformed.withColumnRenamed("Last Update", "last_updated")
    dt_transformed = dt_transformed.withColumnRenamed("Confirmed", "confirmed")
    dt_transformed = dt_transformed.withColumnRenamed("Deaths", "deaths")
    dt_transformed = dt_transformed.withColumnRenamed("Recovered", "recovered")
    dt_transformed = dt_transformed.withColumnRenamed("Date", "date")

    dt_transformed = dt_transformed.fillna('', subset=['state'])
    dt_transformed = dt_transformed.fillna(0, subset=['confirmed', 'deaths', 'recovered', 'active'])

    dt_transformed = dt_transformed.withColumn("active",
                                               dt_transformed["confirmed"] - dt_transformed["deaths"] - dt_transformed[
                                                   "recovered"])
    dt_transformed = dt_transformed.withColumn("country", func.regexp_replace('country', "Mainland China", "China"))

    dt_transformed = dt_transformed.withColumn("confirmed", dt_transformed["confirmed"].cast(IntegerType()))
    dt_transformed = dt_transformed.withColumn("deaths", dt_transformed["deaths"].cast(IntegerType()))
    dt_transformed = dt_transformed.withColumn("recovered", dt_transformed["recovered"].cast(IntegerType()))
    dt_transformed = dt_transformed.withColumn("active", dt_transformed["active"].cast(IntegerType()))

    return dt_transformed


def transform_confirmed_cases_and_deaths_globally(dataframe):
    df_globally = dataframe.groupBy("date").sum("confirmed", "deaths").orderBy("date")

    return df_globally


def transform_confirmed_cases_serbia(dataframe):
    df_filtered = dataframe.filter(dataframe.country == "Serbia")
    df_grouped = df_filtered.groupBy("date").sum("confirmed").orderBy("date")

    return df_grouped


def transform_confirmed_cases_norway(dataframe):
    df_filtered = dataframe.filter(dataframe.country == "Norway")
    df_grouped = df_filtered.groupBy("date").sum("confirmed").orderBy("date")

    return df_grouped


def transform_confirmed_cases_italy(dataframe):
    df_filtered = dataframe.filter(dataframe.country == "Italy")
    df_grouped = df_filtered.groupBy("date").sum("confirmed").orderBy("date")

    return df_grouped


def transform_confirmed_cases_china(dataframe):
    df_filtered = dataframe.filter(dataframe.country == "China")
    df_grouped = df_filtered.groupBy("date").sum("confirmed").orderBy("date")

    return df_grouped


def transform_confirmed_cases_europe(dataframe):
    df_temp = dataframe.select([c for c in dataframe.columns if c not in {"state"}])
    w = Window.partitionBy("country")
    df_latest = df_temp.withColumn("maxDate", func.max("date").over(w)).where(
        func.col("date") == func.col("maxDate"))

    df_grouped = df_latest.groupby("country").sum("confirmed")
    df_grouped_europe = df_grouped.filter(df_grouped.country.isin(europe))
    df_ordered_europe = df_grouped_europe.orderBy(func.desc("sum(confirmed)"))

    return df_ordered_europe


def transform_confirmed_cases_comparison(dataframe):
    df_grouped = dataframe.groupBy("date").sum("recovered", "deaths", "active").orderBy("date")

    return df_grouped


def transform_confirmed_cases_mortality_rates(dataframe):
    mortality_window = Window.partitionBy('country')
    df_mortality = dataframe.withColumn("maxDate", func.max("date").over(mortality_window)).where(
        func.col("date") == func.col("maxDate"))
    df_grouped_mortality = df_mortality.groupby("country").sum("confirmed", "deaths", "recovered", "active")

    df_grouped_ordered = df_grouped_mortality.withColumn("mortalityRate", func.round(
        df_grouped_mortality["sum(deaths)"] / df_grouped_mortality["sum(confirmed)"] * 100, 2)).orderBy(
        func.desc("mortalityRate")).limit(10).orderBy(func.asc("mortalityRate"))

    return df_grouped_ordered


def transform_confirmed_cases_recovery_rates(dataframe):
    recovery_window = Window.partitionBy('country')
    df_recovery = dataframe.withColumn("maxDate", func.max("date").over(recovery_window)).where(
        func.col("date") == func.col("maxDate"))
    df_grouped_recovery = df_recovery.groupby("country").sum("confirmed", "deaths", "recovered", "active")

    df_grouped_ordered = df_grouped_recovery.withColumn("recoveryRate", func.round(
        df_grouped_recovery["sum(recovered)"] / df_grouped_recovery["sum(confirmed)"] * 100, 2)).orderBy(
        func.desc("recoveryRate")).limit(10).orderBy(func.asc("recoveryRate"))

    return df_grouped_ordered


def transform_time_series(dataframe):
    time_series_data = dataframe.select(["date", "confirmed"]).groupby("date").sum().orderBy("date")
    time_series_data = time_series_data.withColumnRenamed("date", "ds")
    time_series_data = time_series_data.withColumnRenamed("sum(confirmed)", "y")

    return time_series_data


def transform_time_series_test_data(dataframe, spark):
    time_series_data = transform_time_series(dataframe).toPandas()

    train_range = np.random.rand(len(time_series_data)) < 0.8
    test_ts = time_series_data[~train_range]

    df_test = spark.createDataFrame(test_ts)

    return df_test


def transform_time_series_predictions(dataframe, spark):
    time_series_data = transform_time_series(dataframe).toPandas()

    train_range = np.random.rand(len(time_series_data)) < 0.8
    train_ts = time_series_data[train_range]
    test_ts = time_series_data[~train_range]
    test_ts = test_ts.set_index('ds')

    prophet_model = Prophet()
    prophet_model.fit(train_ts)

    future = pd.DataFrame(test_ts.index)
    predict = prophet_model.predict(future)
    forecast = predict[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    df_predictions = spark.createDataFrame(forecast)

    return df_predictions


def transform_time_series_forecasting(dataframe, spark):
    time_series_data = transform_time_series(dataframe).toPandas()

    prophet_model_full = Prophet()
    prophet_model_full.fit(time_series_data)
    future_full = prophet_model_full.make_future_dataframe(periods=150)
    forecast_full = prophet_model_full.predict(future_full)

    df_forecast = spark.createDataFrame(forecast_full)

    return df_forecast


def load_data(dataframe, name):
    (dataframe
     .coalesce(1)
     .write
     .json("./outputs/cases_time_analysis/" + name, mode='overwrite'))
    return None


europe = list(
    ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France',
     'Germany', 'Greece', 'Hungary', 'Ireland',
     'Italy', 'Latvia', 'Luxembourg', 'Lithuania', 'Malta', 'Norway', 'Netherlands', 'Poland', 'Portugal', 'Romania',
     'Slovakia', 'Slovenia',
     'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus',
     'Albania', 'Bosnia and Herzegovina', 'Kosovo', 'Moldova', 'Montenegro', 'North Macedonia'])

if __name__ == '__main__':
    main()
