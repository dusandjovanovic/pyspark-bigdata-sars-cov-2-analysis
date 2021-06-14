import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True


def dash_sidebar_item(item):
    return dbc.NavLink(item["title"], href=item["href"], active="exact")


def dash_sidebar(title, description, items):
    return html.Div(
        [
            html.H4(title, className="display-6"),
            html.Hr(),
            html.P(description, className="lead pb-4"),
            dbc.Nav(
                list(map(dash_sidebar_item, items)),
                vertical=True,
                pills=True,
            ),
        ],
        style={
            "position": "fixed",
            "top": 0,
            "left": 0,
            "bottom": 0,
            "width": "32rem",
            "padding": "2rem 1rem",
            "backgroundColor": "#f8f9fa",
        },
    )


def dash_content():
    return html.Div(
        id="page-content",
        style={
            "marginLeft": "32rem",
            "marginRight": "2rem",
            "padding": "2rem 1rem",
            "height": "100vh",
            "overflow": "auto",
            "width": "calc(100% - 32rem)"
        }
    )


def dash_graph(figure):
    return dcc.Graph(
        config={'displayModeBar': False},
        figure=figure,
        style={
            "height": "100%"
        }
    )


def dash_error():
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"Something went wrong. Please try another path."),
        ]
    )
