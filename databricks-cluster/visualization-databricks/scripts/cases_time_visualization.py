import sys

import plotly.express as px
import plotly.graph_objects as go

from dependencies.load_dataset import load_dataset
from dependencies.load_spark import load_spark, get_db_utils
import dependencies.colors as color_scheme


def main():
    spark = load_spark()
    dbutils = get_db_utils(spark)

    visualize_confirmed_cases_and_deaths_globally(dbutils)
    visualize_confirmed_cases_countries(dbutils)
    visualize_confirmed_cases_europe(dbutils)
    visualize_confirmed_cases_comparison(dbutils)
    visualize_confirmed_cases_mortality_rates(dbutils)
    visualize_confirmed_cases_recovery_rates(dbutils)

    return None


def visualize_confirmed_cases_and_deaths_globally(dbutils):
    dataframe_pd = load_dataset(sys.argv[1], "confirmed_cases_and_deaths_globally", dbutils)

    fig = px.line(dataframe_pd, x="date", y="sum(confirmed)",
                  title="Confirmed cases over time (logarithmic)",
                  log_y=True, color_discrete_sequence=[color_scheme.color_400])
    fig.show()

    fig = px.line(dataframe_pd, x="date", y="sum(deaths)", title="Death cases over time (logarithmic)",
                  log_y=True, color_discrete_sequence=[color_scheme.color_900])
    fig.show()

    return None


def visualize_confirmed_cases_countries(dbutils):
    df_serbia_grouped = load_dataset(sys.argv[1], "confirmed_cases_serbia", dbutils)
    df_china_grouped = load_dataset(sys.argv[1], "confirmed_cases_china", dbutils)
    df_italy_grouped = load_dataset(sys.argv[1], "confirmed_cases_italy", dbutils)
    df_norway_grouped = load_dataset(sys.argv[1], "confirmed_cases_norway", dbutils)

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

    return None


def visualize_confirmed_cases_europe(dbutils):
    dataframe_pd = load_dataset(sys.argv[1], "confirmed_cases_europe", dbutils)

    fig = px.choropleth(dataframe_pd, locations="country",
                        locationmode='country names', color="sum(confirmed)",
                        hover_name="country", range_color=[1, 1000000],
                        color_continuous_scale='portland',
                        title='European countries with confirmed cases', scope='europe', height=800)
    fig.show()

    fig = px.bar(dataframe_pd.sort_values('sum(confirmed)', ascending=False)[:10][::-1],
                 x='sum(confirmed)', y='country', color_discrete_sequence=[color_scheme.color_400],
                 title='Confirmed cases in Europe (top-10 countries)', text='sum(confirmed)', orientation='h')
    fig.show()

    return None


def visualize_confirmed_cases_comparison(dbutils):
    dataframe_pd = load_dataset(sys.argv[1], "confirmed_cases_comparison", dbutils)

    df_melted = dataframe_pd.melt(id_vars="date", value_vars=['sum(recovered)', 'sum(deaths)', 'sum(active)'],
                                  var_name='case', value_name='count')

    fig = px.area(df_melted, x="date", y="count", color='case',
                  title='Cases over time',
                  color_discrete_sequence=[color_scheme.color_200, color_scheme.color_400,
                                           color_scheme.color_800])
    fig.show()

    return None


def visualize_confirmed_cases_mortality_rates(dbutils):
    dataframe_pd = load_dataset(sys.argv[1], "confirmed_cases_mortality_rates", dbutils)

    fig = px.bar(dataframe_pd, x='mortalityRate', y='country', title='Deaths per 100 confirmed cases (top-10)',
                 text='mortalityRate', height=800, orientation='h',
                 color_discrete_sequence=[color_scheme.color_600]
                 )
    fig.show()

    return None


def visualize_confirmed_cases_recovery_rates(dbutils):
    dataframe_pd = load_dataset(sys.argv[1], "confirmed_cases_recovery_rates", dbutils)

    fig = px.bar(dataframe_pd, x='recoveryRate', y='country', title='Recoveries per 100 confirmed cases (top-10)',
                 text='recoveryRate', height=800, orientation='h',
                 color_discrete_sequence=[color_scheme.color_500]
                 )
    fig.show()

    return None


if __name__ == '__main__':
    main()
