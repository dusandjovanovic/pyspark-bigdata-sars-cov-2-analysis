from pyspark.sql import Window
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as func
import plotly.graph_objects as go
import plotly.express as px
from visualisation.dependencies import color_scheme

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
                  title="Confirmed cases over time (logarithmic)",
                  log_y=True, color_discrete_sequence=[color_scheme.color_400])
    fig.show()

    fig = px.line(df_globally, x="date", y="sum(deaths)", title="Death cases over time (logarithmic)",
                  log_y=True, color_discrete_sequence=[color_scheme.color_900])
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
                             line_color=color_scheme.color_500,
                             opacity=0.8
                             ))
    fig.add_trace(go.Scatter(x=df_china_grouped['date'],
                             y=df_china_grouped['sum(confirmed)'],
                             name="China",
                             line_color=color_scheme.color_700,
                             opacity=0.8
                             ))
    fig.add_trace(go.Scatter(x=df_italy_grouped['date'],
                             y=df_italy_grouped['sum(confirmed)'],
                             name="Italy",
                             line_color=color_scheme.color_900,
                             opacity=0.8
                             ))
    fig.add_trace(go.Scatter(x=df_norway_grouped['date'],
                             y=df_norway_grouped['sum(confirmed)'],
                             name="Norway",
                             line_color=color_scheme.color_300,
                             opacity=0.8
                             ))
    fig.update_layout(title_text="Overview of case growth in Serbia, China, Italy and Norway")

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
                        title='European countries with confirmed cases', scope='europe', height=800)
    fig.show()

    fig = px.bar(df_latest_grouped_europe.sort_values('sum(confirmed)', ascending=False)[:10][::-1],
                 x='sum(confirmed)', y='country', color_discrete_sequence=[color_scheme.color_400],
                 title='Confirmed cases in Europe (top-10 countries)', text='sum(confirmed)', orientation='h')
    fig.show()

    return dataframe


def transform_confirmed_cases_comparison(dataframe):
    df_grouped = dataframe.groupBy("date").sum("recovered", "deaths", "active").orderBy("date").toPandas()
    df_melted = df_grouped.melt(id_vars="date", value_vars=['sum(recovered)', 'sum(deaths)', 'sum(active)'],
                                var_name='case', value_name='count')

    fig = px.area(df_melted, x="date", y="count", color='case',
                  title='Cases over time',
                  color_discrete_sequence=[color_scheme.color_200, color_scheme.color_400,
                                           color_scheme.color_800])
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
                 title='Deaths per 100 confirmed cases (top-10)', text='mortalityRate', height=800, orientation='h',
                 color_discrete_sequence=[color_scheme.color_600]
                 )
    fig.show()

    fig = px.bar(df_latest_grouped_with_recovery_rate.sort_values(by="recoveryRate", ascending=False)[:10][::-1],
                 x='recoveryRate', y='country',
                 title='Recoveries per 100 confirmed cases (top-10)', text='recoveryRate', height=800, orientation='h',
                 color_discrete_sequence=[color_scheme.color_500]
                 )
    fig.show()

    return dataframe


def load_data(dataframe, name):
    (dataframe
     .coalesce(1)
     .write
     .csv("./outputs/cases_time_analysis/" + name, mode='overwrite', header=True))
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
