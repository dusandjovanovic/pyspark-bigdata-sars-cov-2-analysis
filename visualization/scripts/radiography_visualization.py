from dependencies.utils import load_dataset
from dependencies.dash import app, dash_content, dash_sidebar, dash_graph, dash_error
from meta.radiography import analysis_name, analysis_description, analysis_options, marker_labels, marker_colours, \
    DESCRIPTOR_NORMAL, DESCRIPTOR_COVID19, DESCRIPTOR_LUNG_OPACITY, DESCRIPTOR_VIRAL_PNEUMONIA, CLASSNAME_NORMAL, \
    CLASSNAME_COVID19, CLASSNAME_VIRAL_PNEUMONIA, CLASSNAME_LUNG_OPACITY

import sys
import cv2
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dash_sidebar(analysis_name, analysis_description, analysis_options),
        dash_content()
    ]
)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return visualize_percentage_of_samples()
    elif pathname == "/min":
        return visualize_colour_distribution_min()
    elif pathname == "/max":
        return visualize_colour_distribution_max()
    elif pathname == "/mean":
        return visualize_colour_distribution_mean()
    elif pathname == "/standard_deviation":
        return visualize_colour_distribution_standard_deviation()
    elif pathname == "/mean_standard_deviation":
        return visualize_colour_distribution_mean_standard_deviation()
    elif pathname == "/sample_images":
        return visualize_sample_images()
    elif pathname == "/sample_images_channel":
        return visualize_sample_images_channel()
    elif pathname == "/ml_classification":
        return visualize_ml_classification()

    return dash_error()


def visualize_percentage_of_samples():
    dataframe_pd = load_dataset(sys.argv[1], "percentage_of_samples")

    figure = go.Figure(
        data=[
            go.Bar(
                x=marker_labels,
                y=dataframe_pd['percentage'].values,
                marker_color=marker_colours
            )
        ]
    )

    figure.update_layout(title='Percentages of image samples across different classes')

    return dash_graph(figure)


def visualize_colour_distribution_min():
    [dataframe_normal,
     dataframe_covid19,
     dataframe_lung_opacity,
     dataframe_viral_pneumonia] = colour_distribution_dataframe()

    figure = ff.create_distplot(
        [
            dataframe_normal['min'].values,
            dataframe_covid19['min'].values,
            dataframe_lung_opacity['min'].values,
            dataframe_viral_pneumonia['min'].values
        ],
        marker_labels,
        curve_type="normal",
        colors=marker_colours,
        show_hist=False
    )

    figure.update_layout(title='Minimal value distribution by class')

    return dash_graph(figure)


def visualize_colour_distribution_max():
    [dataframe_normal,
     dataframe_covid19,
     dataframe_lung_opacity,
     dataframe_viral_pneumonia] = colour_distribution_dataframe()

    figure = ff.create_distplot(
        [
            dataframe_normal['max'].values,
            dataframe_covid19['max'].values,
            dataframe_lung_opacity['max'].values,
            dataframe_viral_pneumonia['max'].values
        ],
        marker_labels,
        curve_type="normal",
        colors=marker_colours,
        show_hist=False
    )

    figure.update_layout(title='Maximum value distribution by class')

    return dash_graph(figure)


def visualize_colour_distribution_mean():
    [dataframe_normal,
     dataframe_covid19,
     dataframe_lung_opacity,
     dataframe_viral_pneumonia] = colour_distribution_dataframe()

    figure = ff.create_distplot(
        [
            dataframe_normal['mean'].values,
            dataframe_covid19['mean'].values,
            dataframe_lung_opacity['mean'].values,
            dataframe_viral_pneumonia['mean'].values
        ],
        marker_labels,
        curve_type="normal",
        colors=marker_colours,
        show_hist=False
    )

    figure.update_layout(title='Mean value distribution by class')

    return dash_graph(figure)


def visualize_colour_distribution_standard_deviation():
    [dataframe_normal,
     dataframe_covid19,
     dataframe_lung_opacity,
     dataframe_viral_pneumonia] = colour_distribution_dataframe()

    figure = ff.create_distplot(
        [
            dataframe_normal['standard_deviation'].values,
            dataframe_covid19['standard_deviation'].values,
            dataframe_lung_opacity['standard_deviation'].values,
            dataframe_viral_pneumonia['standard_deviation'].values
        ],
        marker_labels,
        curve_type="normal",
        colors=marker_colours,
        show_hist=False
    )

    figure.update_layout(title='Standard deviation value distribution by class')

    return dash_graph(figure)


def visualize_colour_distribution_mean_standard_deviation():
    [dataframe_normal,
     dataframe_covid19,
     dataframe_lung_opacity,
     dataframe_viral_pneumonia] = colour_distribution_dataframe()

    figure = make_subplots(rows=2, cols=2)

    figure.add_trace(
        go.Scatter(
            x=dataframe_normal['mean'].values,
            y=dataframe_normal['standard_deviation'].values,
            mode='markers',
            marker_color=marker_colours[0],
            name='Normal'
        ),
        row=1,
        col=1
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_covid19['mean'].values,
            y=dataframe_covid19['standard_deviation'].values,
            mode='markers',
            marker_color=marker_colours[1],
            name='COVID-19'
        ),
        row=1,
        col=2
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_lung_opacity['mean'].values,
            y=dataframe_lung_opacity['standard_deviation'].values,
            mode='markers',
            marker_color=marker_colours[2],
            name='Lung Opacity'
        ),
        row=2,
        col=1
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe_viral_pneumonia['mean'].values,
            y=dataframe_viral_pneumonia['standard_deviation'].values,
            mode='markers',
            marker_color=marker_colours[3],
            name='Viral Pneumonia'
        ),
        row=2,
        col=2
    )

    figure.update_layout(title='Mean and Standard deviation of image samples')

    return dash_graph(figure)


def visualize_ml_classification():
    dataframe_pd = load_dataset(sys.argv[1], "ml_classification")

    accuracy = dataframe_pd.iloc[0].accuracy
    confusion_matrix = dataframe_pd.iloc[0].matrix
    x = [CLASSNAME_NORMAL, CLASSNAME_COVID19, CLASSNAME_LUNG_OPACITY, CLASSNAME_VIRAL_PNEUMONIA]
    y = [CLASSNAME_NORMAL, CLASSNAME_COVID19, CLASSNAME_LUNG_OPACITY, CLASSNAME_VIRAL_PNEUMONIA]

    z_text = [[str(y) for y in x] for x in confusion_matrix]

    figure = ff.create_annotated_heatmap(
        confusion_matrix,
        x=x,
        y=y,
        annotation_text=z_text,
        colorscale='balance'
    )

    figure.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Predicted value",
            xref="paper",
            yref="paper"
        )
    )

    figure.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=-0.35,
            y=0.5,
            showarrow=False,
            text="Real value",
            textangle=-90,
            xref="paper",
            yref="paper"
        )
    )

    figure.update_layout(margin=dict(t=50, l=200))
    figure['data'][0]['showscale'] = True

    figure.update_layout(
        title='Confusion Matrix with reported accuracy of ' + str(accuracy),
        margin=dict(
            l=200,
            r=200,
            b=50,
            t=200,
            pad=4
        )
    )

    figure.show()

    return dash_graph(figure)


def visualize_sample_images():
    [img_normal, img_covid, img_lung_opacity, img_viral_pneumonia] = sample_images()

    figure = make_subplots(rows=2, cols=2, subplot_titles=marker_labels)

    figure.add_trace(
        px.imshow(img_normal).data[0],
        row=1,
        col=1
    )

    figure.add_trace(
        px.imshow(img_covid).data[0],
        row=1,
        col=2
    )

    figure.add_trace(
        px.imshow(img_lung_opacity).data[0],
        row=2,
        col=1
    )

    figure.add_trace(
        px.imshow(img_viral_pneumonia).data[0],
        row=2,
        col=2
    )

    figure.update_yaxes(matches=None, showticklabels=False, visible=False)
    figure.update_xaxes(matches=None, showticklabels=False, visible=False)
    figure.update_layout(title_text="Side By Side Radiography Images")

    return dash_graph(figure)


def visualize_sample_images_channel():
    [img_normal, img_covid, img_lung_opacity, img_viral_pneumonia] = sample_images()

    figure = make_subplots(rows=2, cols=2, subplot_titles=marker_labels)

    figure.add_trace(
        px.imshow(img_normal[:, :, 0]).data[0],
        row=1,
        col=1
    )

    figure.add_trace(
        px.imshow(img_covid[:, :, 0]).data[0],
        row=1,
        col=2
    )

    figure.add_trace(
        px.imshow(img_lung_opacity[:, :, 0]).data[0],
        row=2,
        col=1
    )

    figure.add_trace(
        px.imshow(img_viral_pneumonia[:, :, 0]).data[0],
        row=2,
        col=2
    )

    figure.update_yaxes(matches=None, showticklabels=False, visible=False)
    figure.update_xaxes(matches=None, showticklabels=False, visible=False)
    figure.update_layout(title_text="Side By Side Radiography Images (R-Channel)")

    return dash_graph(figure)


def colour_distribution_dataframe():
    dataframe_pd = load_dataset(sys.argv[1], "colour_distribution")

    dataframe_normal = dataframe_pd.loc[dataframe_pd['label'] == DESCRIPTOR_NORMAL]
    dataframe_covid19 = dataframe_pd.loc[dataframe_pd['label'] == DESCRIPTOR_COVID19]
    dataframe_lung_opacity = dataframe_pd.loc[dataframe_pd['label'] == DESCRIPTOR_LUNG_OPACITY]
    dataframe_viral_pneumonia = dataframe_pd.loc[dataframe_pd['label'] == DESCRIPTOR_VIRAL_PNEUMONIA]

    return [dataframe_normal, dataframe_covid19, dataframe_lung_opacity, dataframe_viral_pneumonia]


def sample_images():
    dataframe_pd = load_dataset(sys.argv[1], "take_samples")

    img_normal = cv2.imread(
        (dataframe_pd.loc[dataframe_pd['class_name'] == CLASSNAME_NORMAL]).iloc[0].origin
    )

    img_covid = cv2.imread(
        (dataframe_pd.loc[dataframe_pd['class_name'] == CLASSNAME_COVID19]).iloc[0].origin
    )

    img_lung_opacity = cv2.imread(
        (dataframe_pd.loc[dataframe_pd['class_name'] == CLASSNAME_LUNG_OPACITY]).iloc[0].origin
    )

    img_viral_pneumonia = cv2.imread(
        (dataframe_pd.loc[dataframe_pd['class_name'] == CLASSNAME_VIRAL_PNEUMONIA]).iloc[0].origin
    )

    return [img_normal, img_covid, img_lung_opacity, img_viral_pneumonia]


if __name__ == '__main__':
    app.run_server()
