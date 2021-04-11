from pyspark.sql import Window
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as func
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
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
    data_transformed = transform_data(data, sql_context)

    # confirmed cases and deaths globally
    data_transformed = transform_confirmed_cases_and_deaths_globally(data_transformed)
    load_data(data_transformed, "confirmed_cases_and_deaths_globally")

    # confirmed cases by countries
    data_transformed = transform_confirmed_cases_countries(data_transformed)
    load_data(data_transformed, "confirmed_cases_countries")

    # confirmed cases in Europe
    data_transformed = transform_confirmed_cases_europe(data_transformed)
    load_data(data_transformed, "confirmed_cases_europe")

    # confirmed cases comparison
    data_transformed = transform_confirmed_cases_comparison(data_transformed)
    load_data(data_transformed, "confirmed_cases_comparison")

    # confirmed cases comparison by countries
    data_transformed = transform_confirmed_cases_comparison_countries(data_transformed)
    load_data(data_transformed, "confirmed_cases_comparison_countries")

    # forecasting the future
    data_transformed = transform_future_forecasting(data_transformed)
    load_data(data_transformed, "future_forecasting")

    log.warn('Terminating cases_time analysis...')

    spark.stop()
    return None


def extract_data(spark):
    dataframe = spark.read.csv("../data/covid-19-dataset/covid_19_clean_complete.csv",
                               header=True, dateFormat="yyyy-MM-dd")

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
    df_globally = dataframe.groupBy("date").sum("confirmed", "deaths").orderBy("date").toPandas()

    fig = px.line(df_globally, x="date", y="sum(confirmed)",
                  title="Confirmed Cases (Logarithmic Scale) Over Time",
                  log_y=True)
    fig.show()

    fig = px.line(df_globally, x="date", y="sum(deaths)", title="Worldwide Deaths (Logarithmic Scale) Over Time",
                  log_y=True, color_discrete_sequence=['#F42272'])
    fig.show()

    return dataframe


def transform_confirmed_cases_countries(dataframe):
    df_serbia = dataframe.filter(dataframe.country == "Serbia")
    df_china = dataframe.filter(dataframe.country == "China")
    df_italy = dataframe.filter(dataframe.country == "Italy")
    df_norway = dataframe.filter(dataframe.country == "Norway")

    df_serbia_grouped = df_serbia.groupBy("date").sum("confirmed").orderBy("date").toPandas()
    df_china_grouped = df_china.groupBy("date").sum("confirmed").orderBy("date").toPandas()
    df_italy_grouped = df_italy.groupBy("date").sum("confirmed").orderBy("date").toPandas()
    df_norway_grouped = df_norway.groupBy("date").sum("confirmed").orderBy("date").toPandas()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_serbia_grouped['date'],
                             y=df_serbia_grouped['sum(confirmed)'],
                             name="Serbia",
                             line_color="deepskyblue",
                             opacity=0.8
                             ))
    fig.add_trace(go.Scatter(x=df_china_grouped['date'],
                             y=df_china_grouped['sum(confirmed)'],
                             name="China",
                             line_color="red",
                             opacity=0.8
                             ))
    fig.add_trace(go.Scatter(x=df_italy_grouped['date'],
                             y=df_italy_grouped['sum(confirmed)'],
                             name="Italy",
                             line_color="green",
                             opacity=0.8
                             ))
    fig.add_trace(go.Scatter(x=df_norway_grouped['date'],
                             y=df_norway_grouped['sum(confirmed)'],
                             name="Norway",
                             line_color="blue",
                             opacity=0.8
                             ))
    fig.update_layout(title_text="Overview of the case growth in Serbia, China, Italy and Norway")

    fig.show()

    return dataframe


def transform_confirmed_cases_europe(dataframe):
    df_temp = dataframe.select([c for c in dataframe.columns if c not in {"state"}])
    w = Window.partitionBy("country")
    df_latest = df_temp.withColumn("maxDate", func.max("date").over(w)).where(
        func.col("date") == func.col("maxDate"))
    df_latest_grouped = df_latest.groupby("country").sum("confirmed")
    df_latest_grouped_europe = df_latest_grouped.filter(df_latest_grouped.country.isin(europe)).toPandas()

    fig = px.choropleth(df_latest_grouped_europe, locations="country",
                        locationmode='country names', color="sum(confirmed)",
                        hover_name="country", range_color=[1, 1000000],
                        color_continuous_scale='portland',
                        title='European Countries with Confirmed Cases', scope='europe', height=800)
    fig.show()

    fig = px.bar(df_latest_grouped_europe.sort_values('sum(confirmed)', ascending=False)[:10][::-1],
                 x='sum(confirmed)', y='country', color_discrete_sequence=['#84DCC6'],
                 title='Confirmed Cases in Europe', text='sum(confirmed)', orientation='h')
    fig.show()

    return dataframe


def transform_confirmed_cases_comparison(dataframe):
    df_grouped = dataframe.groupBy("date").sum("recovered", "deaths", "active").orderBy("date").toPandas()
    df_melted = df_grouped.melt(id_vars="date", value_vars=['sum(recovered)', 'sum(deaths)', 'sum(active)'],
                                var_name='case', value_name='count')

    fig = px.area(df_melted, x="date", y="count", color='case',
                  title='Cases over time: Area Plot', color_discrete_sequence=['cyan', 'red', 'orange'])
    fig.show()

    return dataframe


def transform_confirmed_cases_comparison_countries(dataframe):
    w = Window.partitionBy('country')
    df_latest = dataframe.withColumn("maxDate", func.max("date").over(w)).where(
        func.col("date") == func.col("maxDate"))
    df_latest_grouped = df_latest.groupby("country").sum("confirmed", "deaths", "recovered", "active")

    df_latest_grouped_with_mortality_rate = df_latest_grouped.withColumn("mortalityRate", func.round(
        df_latest_grouped["sum(deaths)"] / df_latest_grouped["sum(confirmed)"] * 100, 2)).orderBy(
        "mortalityRate").toPandas()
    df_latest_grouped_with_recovery_rate = df_latest_grouped.withColumn("recoveryRate", func.round(
        df_latest_grouped["sum(recovered)"] / df_latest_grouped["sum(confirmed)"] * 100, 2)).orderBy(
        "recoveryRate").toPandas()

    fig = px.bar(df_latest_grouped_with_mortality_rate.sort_values(by="mortalityRate", ascending=False)[:10][::-1],
                 x='mortalityRate', y='country',
                 title='Deaths per 100 Confirmed Cases', text='mortalityRate', height=800, orientation='h',
                 color_discrete_sequence=['darkred']
                 )
    fig.show()

    fig = px.bar(df_latest_grouped_with_recovery_rate.sort_values(by="recoveryRate", ascending=False)[:10][::-1],
                 x='recoveryRate', y='country',
                 title='Recoveries per 100 Confirmed Cases', text='recoveryRate', height=800, orientation='h',
                 color_discrete_sequence=['#2ca02c']
                 )
    fig.show()

    return dataframe

def transform_future_forecasting(dataframe):
    time_series_data = dataframe.select(["date", "confirmed"]).groupby("date").sum().orderBy("date")
    time_series_data = time_series_data.withColumnRenamed("date", "ds")
    time_series_data = time_series_data.withColumnRenamed("sum(confirmed)", "y")

    time_series_data = time_series_data.toPandas()

    train_range = np.random.rand(len(time_series_data)) < 0.8
    train_ts = time_series_data[train_range]
    test_ts = time_series_data[~train_range]
    test_ts = test_ts.set_index('ds')

    prophet_model = Prophet()
    prophet_model.fit(train_ts)

    future = pd.DataFrame(test_ts.index)
    predict = prophet_model.predict(future)
    forecast = predict[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast = forecast.set_index('ds')

    test_fig = go.Figure()
    test_fig.add_trace(go.Scatter(
        x=test_ts.index,
        y=test_ts.y,
        name="Actual Cases",
        line_color="deepskyblue",
        mode='lines',
        opacity=0.8))
    test_fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.yhat,
        name="Prediction",
        mode='lines',
        line_color='red',
        opacity=0.8))
    test_fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.yhat_lower,
        name="Prediction Lower Bound",
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        opacity=0.8))
    test_fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.yhat_upper,
        name="Prediction Upper Bound",
        mode='lines',
        line=dict(color='royalblue', width=2, dash='dash'),
        opacity=0.8
    ))

    test_fig.update_layout(title_text="Prophet Model's Test Prediction",
                           xaxis_title="Date", yaxis_title="Cases", )

    test_fig.show()

    prophet_model_full = Prophet()
    prophet_model_full.fit(time_series_data)
    future_full = prophet_model_full.make_future_dataframe(periods=150)
    forecast_full = prophet_model_full.predict(future_full)
    forecast_full = forecast_full.set_index('ds')
    prediction_fig = go.Figure()
    prediction_fig.add_trace(go.Scatter(
        x=time_series_data.ds,
        y=time_series_data.y,
        name="Actual",
        line_color="red",
        opacity=0.8))
    prediction_fig.add_trace(go.Scatter(
        x=forecast_full.index,
        y=forecast_full.yhat,
        name="Prediction",
        line_color="deepskyblue",
        opacity=0.8))
    prediction_fig.update_layout(title_text="Prophet Model Forecasting",
                                 xaxis_title="Date", yaxis_title="Cases", )

    prediction_fig.show()

    return dataframe


def load_data(dataframe, name):
    (dataframe
     .coalesce(1)
     .write
     .csv(name, mode='overwrite', header=True))
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
