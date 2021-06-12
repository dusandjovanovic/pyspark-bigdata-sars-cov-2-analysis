import sys
import imageio

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dependencies.load_dataset import load_dataset
import dependencies.colors as color_scheme


def main():
    visualize_percentage_of_samples()
    visualize_sample_images()

    return None


def visualize_percentage_of_samples():
    dataframe_pd = load_dataset(sys.argv[1], "percentage_of_samples")

    fig = go.Figure(
        data=[
            go.Bar(x=["Healthy", "COVID-19", "Lung Opacity", "Viral Pneumonia"], y=dataframe_pd['percentage'].values,
                   marker_color=[color_scheme.color_200, color_scheme.color_400, color_scheme.color_600,
                                 color_scheme.color_800])])
    fig.show()

    return None


def visualize_sample_images():
    img_base = sys.argv[2]
    img_normal = imageio.imread(img_base + '/Normal/Normal-1.png')
    img_covid = imageio.imread(img_base + '/COVID/COVID-1.png')
    img_lung_opacity = imageio.imread(img_base + '/Lung_Opacity/Lung_Opacity-1.png')
    img_viral_pneumonia = imageio.imread(img_base + '/Viral_Pneumonia/Viral_Pneumonia-1.png')

    fig = make_subplots(rows=2, cols=2)

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

    fig = make_subplots(1, 1)
    for channel, color in enumerate(['red', 'green', 'blue']):
        fig.add_trace(go.Histogram(x=img_normal[..., channel].ravel(), opacity=0.5,
                                   marker_color=color, name='%s channel' % color), 1, 1)
    fig.show()


if __name__ == '__main__':
    main()
