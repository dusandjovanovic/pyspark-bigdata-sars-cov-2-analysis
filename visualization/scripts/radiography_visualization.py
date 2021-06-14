import sys
import cv2

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from dependencies.load_dataset import load_dataset
import dependencies.colors as color_scheme


def main():
    visualize_sample_images()
    visualize_percentage_of_samples()
    visualize_colour_distribution()

    return None


def visualize_percentage_of_samples():
    dataframe_pd = load_dataset(sys.argv[1], "percentage_of_samples")

    fig = go.Figure(data=[go.Bar(x=marker_labels, y=dataframe_pd['percentage'].values, marker_color=marker_colours)])
    fig.show()

    return None


def visualize_colour_distribution():
    dataframe_pd = load_dataset(sys.argv[1], "colour_distribution")

    dataframe_normal = dataframe_pd.loc[dataframe_pd['label'] == DESCRIPTOR_NORMAL]
    dataframe_covid19 = dataframe_pd.loc[dataframe_pd['label'] == DESCRIPTOR_COVID19]
    dataframe_lung_opacity = dataframe_pd.loc[dataframe_pd['label'] == DESCRIPTOR_LUNG_OPACITY]
    dataframe_viral_pneumonia = dataframe_pd.loc[dataframe_pd['label'] == DESCRIPTOR_VIRAL_PNEUMONIA]

    fig = ff.create_distplot(
        [dataframe_normal['min'].values, dataframe_covid19['min'].values, dataframe_lung_opacity['min'].values,
         dataframe_viral_pneumonia['min'].values],
        marker_labels,
        curve_type="normal",
        colors=marker_colours, show_hist=False)
    fig.update_layout(title='Minimal value distribution by class')
    fig.show()

    fig = ff.create_distplot(
        [dataframe_normal['max'].values, dataframe_covid19['max'].values, dataframe_lung_opacity['max'].values,
         dataframe_viral_pneumonia['max'].values],
        marker_labels,
        curve_type="normal",
        colors=marker_colours, show_hist=False)
    fig.update_layout(title='Maximum value distribution by class')
    fig.show()

    fig = ff.create_distplot(
        [dataframe_normal['mean'].values, dataframe_covid19['mean'].values, dataframe_lung_opacity['mean'].values,
         dataframe_viral_pneumonia['mean'].values],
        marker_labels,
        curve_type="normal",
        colors=marker_colours, show_hist=False)
    fig.update_layout(title='Mean value distribution by class')
    fig.show()

    fig = ff.create_distplot(
        [dataframe_normal['standard_deviation'].values, dataframe_covid19['standard_deviation'].values,
         dataframe_lung_opacity['standard_deviation'].values,
         dataframe_viral_pneumonia['standard_deviation'].values],
        marker_labels,
        curve_type="normal",
        colors=marker_colours, show_hist=False)
    fig.update_layout(title='Standard deviation value distribution by class')
    fig.show()

    fig = make_subplots(rows=2, cols=2)

    fig.add_trace(
        go.Scatter(x=dataframe_normal['mean'].values, y=dataframe_normal['standard_deviation'].values,
                   mode='markers',
                   marker_color=marker_colours[0],
                   name='Normal'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=dataframe_covid19['mean'].values, y=dataframe_covid19['standard_deviation'].values,
                   mode='markers',
                   marker_color=marker_colours[1],
                   name='COVID-19'),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=dataframe_lung_opacity['mean'].values, y=dataframe_lung_opacity['standard_deviation'].values,
                   mode='markers',
                   marker_color=marker_colours[2],
                   name='Lung Opacity'),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=dataframe_viral_pneumonia['mean'].values, y=dataframe_viral_pneumonia['standard_deviation'].values,
                   mode='markers',
                   marker_color=marker_colours[3],
                   name='Viral Pneumonia'),
        row=2, col=2
    )

    fig.update_layout(title='Mean and Standard deviation of image samples')
    fig.show()

    return None


def visualize_sample_images():
    img_base = sys.argv[2]
    img_normal = cv2.imread(img_base + '/Normal/Normal-1.png')
    img_covid = cv2.imread(img_base + '/COVID/COVID-1.png')
    img_lung_opacity = cv2.imread(img_base + '/Lung_Opacity/Lung_Opacity-1.png')
    img_viral_pneumonia = cv2.imread(img_base + '/Viral_Pneumonia/Viral_Pneumonia-1.png')

    fig = make_subplots(rows=2, cols=2, subplot_titles=["Normal", "COVID-19", "Lung Opacity", "Viral Pneumonia"])

    fig.add_trace(
        px.imshow(img_normal).data[0],
        row=1, col=1
    )

    fig.add_trace(
        px.imshow(img_covid).data[0],
        row=1, col=2
    )

    fig.add_trace(
        px.imshow(img_lung_opacity).data[0],
        row=2, col=1
    )

    fig.add_trace(
        px.imshow(img_viral_pneumonia).data[0],
        row=2, col=2
    )

    fig.update_yaxes(matches=None, showticklabels=False, visible=False)
    fig.update_xaxes(matches=None, showticklabels=False, visible=False)
    fig.update_layout(height=800, width=800, title_text="Side By Side Radiography Images")
    fig.show()

    fig.add_trace(
        px.imshow(img_normal[:, :, 0]).data[0],
        row=1, col=1
    )

    fig.add_trace(
        px.imshow(img_covid[:, :, 0]).data[0],
        row=1, col=2
    )

    fig.add_trace(
        px.imshow(img_lung_opacity[:, :, 0]).data[0],
        row=2, col=1
    )

    fig.add_trace(
        px.imshow(img_viral_pneumonia[:, :, 0]).data[0],
        row=2, col=2
    )

    fig.update_layout(title_text="Side By Side Radiography Images (B-Channel)")
    fig.show()


marker_labels = ["Healthy", "COVID-19", "Lung Opacity", "Viral Pneumonia"]

marker_colours = [color_scheme.color_200,
                  color_scheme.color_400,
                  color_scheme.color_600,
                  color_scheme.color_800]

DESCRIPTOR_NORMAL = 0
DESCRIPTOR_COVID19 = 1
DESCRIPTOR_LUNG_OPACITY = 2
DESCRIPTOR_VIRAL_PNEUMONIA = 3

if __name__ == '__main__':
    main()
