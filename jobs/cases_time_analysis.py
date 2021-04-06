import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from pathlib import Path
import numpy as np
from prophet import Prophet

pio.templates.default = 'plotly_white'
clean_data_dir = Path("../data/covid-19-dataset")

cleaned_data = pd.read_csv(clean_data_dir / 'covid_19_clean_complete.csv', parse_dates=['Date'])

cleaned_data.rename(columns={'ObservationDate': 'date',
                             'Province/State': 'state',
                             'Country/Region': 'country',
                             'Last Update': 'last_updated',
                             'Confirmed': 'confirmed',
                             'Deaths': 'deaths',
                             'Recovered': 'recovered'
                             }, inplace=True)

cases = ['confirmed', 'deaths', 'recovered', 'active']
cleaned_data['active'] = cleaned_data['confirmed'] - cleaned_data['deaths'] - cleaned_data['recovered']
cleaned_data['country'] = cleaned_data['country'].replace('Mainland China', 'China')
cleaned_data[['state']] = cleaned_data[['state']].fillna('')
cleaned_data[cases] = cleaned_data[cases].fillna(0)
cleaned_data.rename(columns={'Date': 'date'}, inplace=True)
data = cleaned_data

# confirmed cases globally over time
grouped = data.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()
fig = px.line(grouped, x="date", y="confirmed",
              title="Confirmed Cases (Logarithmic Scale) Over Time",
              log_y=True)
fig.show()

# confirmed deaths globally over time
fig = px.line(grouped, x="date", y="deaths", title="Worldwide Deaths (Logarithmic Scale) Over Time",
              log_y=True, color_discrete_sequence=['#F42272'])
fig.show()

# confirmed cases in serbia over time
grouped_serbia = data[data['country'] == "Serbia"].reset_index()
grouped_serbia_date = grouped_serbia.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_china = data[data['country'] == "China"].reset_index()
grouped_china_date = grouped_china.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_italy = data[data['country'] == "Italy"].reset_index()
grouped_italy_date = grouped_italy.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_norway = data[data['country'] == "Norway"].reset_index()
grouped_norway_date = grouped_norway.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=grouped_serbia_date['date'],
                         y=grouped_serbia_date['confirmed'],
                         name="Serbia",
                         line_color="deepskyblue",
                         opacity=0.8
                         ))
fig.add_trace(go.Scatter(x=grouped_china_date['date'],
                         y=grouped_china_date['confirmed'],
                         name="China",
                         line_color="red",
                         opacity=0.8
                         ))
fig.add_trace(go.Scatter(x=grouped_italy_date['date'],
                         y=grouped_italy_date['confirmed'],
                         name="Italy",
                         line_color="green",
                         opacity=0.8
                         ))
fig.add_trace(go.Scatter(x=grouped_norway_date['date'],
                         y=grouped_norway_date['confirmed'],
                         name="Norway",
                         line_color="blue",
                         opacity=0.8
                         ))
fig.update_layout(title_text="Overview of the case growth in Serbia, China, Italy and Norway")

fig.show()

# cases in europe
data['state'] = data['state'].fillna('')
temp = data[[col for col in data.columns if col != 'state']]

latest = temp[temp['date'] == max(temp['date'])].reset_index()
latest_grouped = latest.groupby('country')['confirmed', 'deaths'].sum().reset_index()
europe = list(
    ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France',
     'Germany', 'Greece', 'Hungary', 'Ireland',
     'Italy', 'Latvia', 'Luxembourg', 'Lithuania', 'Malta', 'Norway', 'Netherlands', 'Poland', 'Portugal', 'Romania',
     'Slovakia', 'Slovenia',
     'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus',
     'Albania', 'Bosnia and Herzegovina', 'Kosovo', 'Moldova', 'Montenegro', 'North Macedonia'])

europe_grouped_latest = latest_grouped[latest_grouped['country'].isin(europe)]
fig = px.choropleth(europe_grouped_latest, locations="country",
                    locationmode='country names', color="confirmed",
                    hover_name="country", range_color=[1, 1000000],
                    color_continuous_scale='portland',
                    title='European Countries with Confirmed Cases', scope='europe', height=800)
fig.show()

fig = px.bar(europe_grouped_latest.sort_values('confirmed', ascending=False)[:10][::-1],
             x='confirmed', y='country', color_discrete_sequence=['#84DCC6'],
             title='Confirmed Cases in Europe', text='confirmed', orientation='h')
fig.show()

# comparisons of cases

temp = cleaned_data.groupby('date')['recovered', 'deaths', 'active'].sum().reset_index()
temp = temp.melt(id_vars="date", value_vars=['recovered', 'deaths', 'active'],
                 var_name='case', value_name='count')

fig = px.area(temp, x="date", y="count", color='case',
              title='Cases over time: Area Plot', color_discrete_sequence=['cyan', 'red', 'orange'])
fig.show()

# comparisons of countries
cleaned_latest = cleaned_data[cleaned_data['date'] == max(cleaned_data['date'])]
flg = cleaned_latest.groupby('country')['confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()

flg['mortalityRate'] = round((flg['deaths'] / flg['confirmed']) * 100, 2)
temp = flg[flg['confirmed'] > 100]
temp = temp.sort_values('mortalityRate', ascending=False)

fig = px.bar(temp.sort_values(by="mortalityRate", ascending=False)[:10][::-1],
             x='mortalityRate', y='country',
             title='Deaths per 100 Confirmed Cases', text='mortalityRate', height=800, orientation='h',
             color_discrete_sequence=['darkred']
             )
fig.show()

flg['recoveryRate'] = round((flg['recovered'] / flg['confirmed']) * 100, 2)
temp = flg[flg['confirmed'] > 100]
temp = temp.sort_values('recoveryRate', ascending=False)

fig = px.bar(temp.sort_values(by="recoveryRate", ascending=False)[:10][::-1],
             x='recoveryRate', y='country',
             title='Recoveries per 100 Confirmed Cases', text='recoveryRate', height=800, orientation='h',
             color_discrete_sequence=['#2ca02c']
             )
fig.show()

# time formatted
formated_gdf = cleaned_data.groupby(['date', 'country'])['confirmed', 'deaths', 'active', 'recovered'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])
formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['confirmed'].pow(0.3) * 5

fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names',
                     color="confirmed", size='size', hover_name="country",
                     range_color=[0, 5000],
                     projection="natural earth", animation_frame="date", scope="europe",
                     title='COVID-19: Spread Over Time in EUROPE', color_continuous_scale="portland", height=800)
fig.show()

# forecasting the future
time_series_data = cleaned_data[['date', 'confirmed']].groupby('date', as_index=False).sum()
time_series_data.columns = ['ds', 'y']
time_series_data.ds = pd.to_datetime(time_series_data.ds)

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
