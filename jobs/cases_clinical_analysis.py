import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
from datetime import datetime
import re
from prophet import Prophet


# create a new cleaned feature for recovered/death data in order to plot
def is_date(value):
    regex = re.compile(r'\d{1,2}/\d{1,2}/\d{4}')
    return bool(regex.match(value))


patient_info = pd.read_csv("../data/covid-19-patient-level-data/DXY.cn patient level data - Line-list.csv").fillna("NA")

# gender distribution
gender_fig = px.histogram(patient_info, x="gender")
gender_fig.show()

# symptoms
symptoms = pd.DataFrame(data=patient_info['symptom'].value_counts().head(17)[1:])
words = symptoms.index
weights = symptoms.symptom
word_cloud_data = go.Scatter(x=[4, 2, 2, 3, 1.5, 5, 4, 4, 0],
                             y=[2, 2, 3, 3, 1, 5, 1, 3, 0],
                             mode='text',
                             text=words,
                             marker={'opacity': 0.5},
                             textfont={'size': weights,
                                       'color': ["red", "green", "blue", "purple", "black", "orange", "blue", "black"]})
layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                    'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})
word_cloud = go.Figure(data=[word_cloud_data], layout=layout)
word_cloud.update_layout(title_text='Word cloud of most common symptoms by frequency')
word_cloud.show()

# relation between age, recovery and death
patient_info['clean_recovered'] = patient_info['recovered'].apply(lambda x: '1' if is_date(x) else x)
patient_info['clean_recovered'] = patient_info['clean_recovered'].astype('category')
patient_info['clean_death'] = patient_info['death'].apply(lambda x: '1' if is_date(x) else x)
patient_info['clean_death'] = patient_info['clean_death'].astype('category')
rec_age_fig = make_subplots(rows=1, cols=2, subplot_titles=("Age vs. Recovered", "Age vs. Death"))

rec_age_fig.add_trace(go.Box(x=patient_info['clean_recovered'], y=patient_info['age'], name="Recovered"),
              row=1, col=1)
rec_age_fig.add_trace(go.Box(x=patient_info['clean_death'], y=patient_info['age'], name = "Death"),
              row=1, col=2)
rec_age_fig.update_traces(boxpoints='all')
rec_age_fig.update_layout(title_text="Subplots of age in relation to recovery and death")
rec_age_fig.show()

