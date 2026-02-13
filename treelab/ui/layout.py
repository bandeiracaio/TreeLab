"""Main Dash layout for TreeLab."""

import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from treelab.core.state_manager import StateManager
from treelab.core.logger import SessionLogger


def create_app(state_manager: StateManager, logger: SessionLogger):
    """
    Create and configure the Dash application.

    Args:
        state_manager: StateManager instance
        logger: SessionLogger instance

    Returns:
        Configured Dash app
    """
    # Initialize Dash app with Bootstrap theme
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    # Store state manager and logger in app for callbacks
    app.state_manager = state_manager
    app.session_logger = logger

    # Build layout
    app.layout = create_layout()

    # Register callbacks
    from treelab.ui.callbacks import register_callbacks

    register_callbacks(app)

    return app


def create_layout():
    """Create the main layout structure."""

    layout = dbc.Container(
        [
            # Header
            create_header(),
            html.Hr(),
            # Main content area
            dbc.Row(
                [
                    # Left column: Action selector and history
                    dbc.Col(
                        [create_action_panel(), html.Br(), create_history_panel()],
                        width=3,
                    ),
                    # Right column: Tabs with visualizations
                    dbc.Col([create_tabs_panel()], width=9),
                ]
            ),
            # Hidden components for state management
            dcc.Store(id="current-mode", data="transformation"),
            dcc.Store(id="action-params", data={}),
            dcc.Store(id="refresh-trigger", data=0),
            dcc.Dropdown(
                id="dist-column-selector",
                options=[],
                value=None,
                style={"display": "none"},
            ),
        ],
        fluid=True,
        style={"padding": "20px"},
    )

    return layout


def create_header():
    """Create the header with title and mode switcher."""
    from treelab.utils.ascii_art import get_simple_banner

    return dbc.Row(
        [
            dbc.Col(
                [
                    html.Pre(
                        get_simple_banner(),
                        style={
                            "fontSize": "10px",
                            "lineHeight": "1.2",
                            "color": "#666",
                            "marginTop": "-10px",
                            "marginBottom": "0px",
                        },
                    ),
                ],
                width=3,
                style={"paddingTop": "5px"},
            ),
            dbc.Col(
                [
                    # Current mode indicator
                    html.Div(
                        id="mode-indicator",
                        style={
                            "padding": "15px",
                            "borderRadius": "8px",
                            "backgroundColor": "#e3f2fd",
                            "border": "2px solid #2196f3",
                            "marginTop": "10px",
                            "marginBottom": "10px",
                        },
                        children=[
                            html.Div(
                                "CURRENT MODE",
                                style={
                                    "fontSize": "10px",
                                    "color": "#666",
                                    "fontWeight": "bold",
                                },
                            ),
                            html.Div(
                                "TRANSFORMATION",
                                id="mode-display",
                                style={
                                    "fontSize": "20px",
                                    "fontWeight": "bold",
                                    "color": "#2196f3",
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Switch Mode:",
                                style={
                                    "fontWeight": "bold",
                                    "marginRight": "10px",
                                    "fontSize": "12px",
                                },
                            ),
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        "Transformation",
                                        id="mode-transform",
                                        color="primary",
                                        size="sm",
                                        active=True,
                                    ),
                                    dbc.Button(
                                        "Modeling",
                                        id="mode-model",
                                        color="success",
                                        size="sm",
                                        active=False,
                                    ),
                                ]
                            ),
                        ],
                        style={"textAlign": "right", "marginTop": "20px"},
                    ),
                ],
                width=9,
            ),
        ]
    )


def create_action_panel():
    """Create the action selection panel."""
    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Actions")),
            dbc.CardBody(
                [
                    # Action dropdown
                    html.Label("Select Action:", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="action-selector",
                        placeholder="Choose an action...",
                        clearable=False,
                    ),
                    html.Br(),
                    # Dynamic parameter form area
                    html.Div(id="action-params-form"),
                    html.Br(),
                    # Execute button
                    dbc.Button(
                        "Execute Action",
                        id="execute-button",
                        color="primary",
                        className="w-100",
                        size="lg",
                        disabled=True,
                    ),
                    # Status message
                    html.Div(id="action-status", style={"marginTop": "10px"}),
                ]
            ),
        ],
        style={"marginBottom": "20px"},
    )


def create_history_panel():
    """Create the history and checkpoint panel."""
    return dbc.Card(
        [
            dbc.CardHeader(html.H5("History")),
            dbc.CardBody(
                [
                    # Checkpoint controls
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Input(
                                        id="checkpoint-name",
                                        placeholder="Checkpoint name...",
                                        type="text",
                                        size="sm",
                                    )
                                ],
                                width=7,
                            ),
                            dbc.Col(
                                [
                                    dbc.Button(
                                        "Save",
                                        id="create-checkpoint-btn",
                                        color="success",
                                        size="sm",
                                        className="w-100",
                                    )
                                ],
                                width=5,
                            ),
                        ],
                        className="mb-3",
                    ),
                    # History list
                    html.Div(
                        id="history-list",
                        style={
                            "maxHeight": "400px",
                            "overflowY": "auto",
                            "fontSize": "14px",
                        },
                    ),
                    html.Hr(),
                    # Export button
                    dbc.Button(
                        "[>>] Export Python Script",
                        id="export-script-btn",
                        color="secondary",
                        size="sm",
                        className="w-100",
                    ),
                    dcc.Download(id="download-script"),
                ]
            ),
        ]
    )


def create_tabs_panel():
    """Create the tabbed visualization panel."""
    return dbc.Card(
        [
            dbc.CardHeader(
                dbc.Tabs(
                    [
                        dbc.Tab(label="[DATA] Data View", tab_id="tab-data"),
                        dbc.Tab(label="[STATS] Statistics", tab_id="tab-stats"),
                        dbc.Tab(label="[DIST] Distributions", tab_id="tab-dist"),
                        dbc.Tab(label="[CORR] Correlations", tab_id="tab-corr"),
                        dbc.Tab(
                            label="[MODEL] Model Results",
                            tab_id="tab-model",
                            disabled=True,
                            id="tab-model-link",
                        ),
                        dbc.Tab(label="[HELP] Help", tab_id="tab-help"),
                    ],
                    id="tabs",
                    active_tab="tab-data",
                )
            ),
            dbc.CardBody(html.Div(id="tab-content", style={"minHeight": "500px"})),
        ]
    )


def create_data_table(df):
    """Create a data table view with column distribution buttons."""
    from treelab.utils.column_analyzer import ColumnAnalyzer

    numeric_cols = ColumnAnalyzer.get_numeric_columns(df)

    # Create column header buttons for numeric columns
    column_buttons = []
    for col in df.columns:
        if col in numeric_cols:
            column_buttons.append(
                dbc.Button(
                    f"[DIST] {col}",
                    id={"type": "col-dist-btn", "column": col},
                    size="sm",
                    color="info",
                    outline=True,
                    style={"margin": "2px"},
                )
            )
        else:
            column_buttons.append(
                html.Span(f"{col}", style={"margin": "5px", "color": "gray"})
            )

    return html.Div(
        [
            html.H5(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns"),
            html.P(
                "Click [DIST] buttons to view column distributions",
                className="text-muted small",
            ),
            html.Div(column_buttons, style={"marginBottom": "15px"}),
            html.Hr(),
            dash_table.DataTable(
                data=df.head(100).to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
                page_size=20,
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "10px",
                    "minWidth": "100px",
                    "maxWidth": "300px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold",
                },
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgb(248, 248, 248)",
                    }
                ],
                sort_action="native",
            ),
        ]
    )
