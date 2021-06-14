import dash_bootstrap_components as dbc
import dash_html_components as html

sidebar_style = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "48tem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

content_style = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


def dash_sidebar_item(title, href):
    return dbc.NavLink(title, href=href, active="exact"),


def dash_sidebar(title, description, items):
    return html.Div(
        [
            html.H2(title, className="display-4"),
            html.Hr(),
            html.P(description, className="lead"),
            dbc.Nav(
                map(dash_sidebar_item, items),
                vertical=True,
                pills=True,
            ),
        ],
        style=sidebar_style,
    )


def dash_content():
    return html.Div(id="page-content", style=content_style)
