"""Dash callbacks for TreeLab interactivity."""

from dash import Input, Output, State, html, dcc, callback_context, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

from treelab.core.action_registry import ActionRegistry
from treelab.utils.column_analyzer import ColumnAnalyzer


def register_callbacks(app):
    """Register all Dash callbacks."""

    def get_action_options(mode: str):
        options = []
        for name in ActionRegistry.get_action_names(mode):
            try:
                action_class = ActionRegistry.get_action_class(name)
                description = getattr(action_class, "description", "")
            except Exception:
                description = ""
            label = f"{name} — {description}" if description else name
            options.append({"label": label, "value": name})
        return options

    def build_tree_figure(model, feature_names, max_depth=3, y_train=None):
        if model is None:
            return None

        tree_model = None
        if hasattr(model, "tree_"):
            tree_model = model
        elif hasattr(model, "estimators_") and model.estimators_:
            tree_model = model.estimators_[0]

        if tree_model is None or not hasattr(tree_model, "tree_"):
            return None

        tree = tree_model.tree_

        total_samples = tree.n_node_samples[0]
        has_target = y_train is not None and len(y_train) > 0

        target_per_node = {}
        if has_target:
            try:
                classes = tree_model.classes_
                values = tree.value

                for node_id in range(tree.node_count):
                    node_values = values[node_id][0]
                    if len(classes) == 2:
                        if classes[0] == 0:
                            count = int(node_values[1])
                        else:
                            count = int(node_values[0])
                        total = int(tree.n_node_samples[node_id])
                        pct = (count / total * 100) if total > 0 else 0
                        target_per_node[node_id] = {
                            "count": count,
                            "total": total,
                            "pct": pct,
                        }
            except Exception:
                pass

        def layout(node_id, depth):
            left = tree.children_left[node_id]
            right = tree.children_right[node_id]

            is_leaf = depth >= max_depth or (left == -1 and right == -1)
            if is_leaf:
                return 1.0, {node_id: (0.5, -depth)}

            lw, lp = layout(left, depth + 1)
            rw, rp = layout(right, depth + 1)
            lw = max(lw, 1.0)
            rw = max(rw, 1.0)

            positions = {}
            for nid, (x, y) in lp.items():
                positions[nid] = (x, y)
            for nid, (x, y) in rp.items():
                positions[nid] = (x + lw, y)

            center_x = (lw + rw) / 2.0
            positions[node_id] = (center_x, -depth)

            return lw + rw, positions

        width, positions = layout(0, 0)
        if not positions:
            return None

        max_samples = max(tree.n_node_samples.tolist())
        palette = ["#b3e5fc", "#c8e6c9", "#ffe0b2", "#f8bbd0"]

        edges = []
        stack = [(0, 0)]
        while stack:
            node_id, depth = stack.pop()
            if depth > max_depth:
                continue
            left = tree.children_left[node_id]
            right = tree.children_right[node_id]
            if left != -1 and depth < max_depth:
                edges.append((node_id, left))
                stack.append((left, depth + 1))
            if right != -1 and depth < max_depth:
                edges.append((node_id, right))
                stack.append((right, depth + 1))

        edge_x = []
        edge_y = []
        for parent, child in edges:
            if parent not in positions or child not in positions:
                continue
            x0, y0 = positions[parent]
            x1, y1 = positions[child]
            edge_x.extend([x0 / width, x1 / width, None])
            edge_y.extend([y0, y1, None])

        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        node_colors = []
        hover_text = []

        for node_id, (x, y) in positions.items():
            node_x.append(x / width)
            node_y.append(y)

            samples = int(tree.n_node_samples[node_id])
            pop_pct = (samples / total_samples) * 100
            size = 18 + (samples / max_samples) * 18
            node_sizes.append(size)

            depth = int(abs(y))
            node_colors.append(palette[depth % len(palette)])

            target_info = ""
            target_label = ""
            if node_id in target_per_node:
                t = target_per_node[node_id]
                target_info = f"\nTarget: {t['count']}/{t['total']} ({t['pct']:.1f}%)"
                target_label = f" | {t['pct']:.0f}%"

            if tree.feature[node_id] == -2:
                label = f"leaf |{target_label}\n{pop_pct:.1f}% pop"
                hover = f"Leaf\nSamples: {samples} ({pop_pct:.1f}%){target_info}"
            else:
                feature_idx = tree.feature[node_id]
                feature_name = (
                    feature_names[feature_idx]
                    if feature_idx < len(feature_names)
                    else f"X{feature_idx}"
                )
                threshold = tree.threshold[node_id]
                label = (
                    f"{feature_name}<={threshold:.2f}\n{pop_pct:.1f}% pop{target_label}"
                )
                hover = f"{feature_name} <= {threshold:.4f}\nSamples: {samples} ({pop_pct:.1f}%){target_info}"

            node_text.append(label)
            hover_text.append(hover)

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#90a4ae", width=1, shape="spline"),
            hoverinfo="none",
        )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color="#37474f"),
                symbol="circle",
            ),
            hovertext=hover_text,
            hoverinfo="text",
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=f"Tree Visualization (depth <= {max_depth})",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#fafafa",
            margin=dict(l=20, r=20, t=40, b=20),
        )

        return fig

    @app.callback(
        [
            Output("action-selector", "options"),
            Output("action-selector", "value"),
            Output("current-mode", "data"),
            Output("mode-transform", "active"),
            Output("mode-model", "active"),
            Output("tab-model-link", "disabled"),
            Output("tab-compare-link", "disabled"),
            Output("mode-display", "children"),
            Output("mode-indicator", "style"),
        ],
        [
            Input("url", "pathname"),
            Input("mode-transform", "n_clicks"),
            Input("mode-model", "n_clicks"),
            Input("refresh-trigger", "data"),
        ],
        [State("current-mode", "data")],
    )
    def switch_mode(
        pathname, transform_clicks, model_clicks, refresh_trigger, current_mode
    ):
        """Handle mode switching between transformation and modeling."""
        ctx = callback_context

        if not ctx.triggered or ctx.triggered[0]["prop_id"].startswith("url"):
            # Initial load - transformation mode
            actions = get_action_options("transformation")
            mode_style = {
                "padding": "15px",
                "borderRadius": "8px",
                "backgroundColor": "#e3f2fd",
                "border": "2px solid #2196f3",
                "marginTop": "10px",
                "marginBottom": "10px",
            }
            return (
                actions,
                None,
                "transformation",
                True,  # transform active
                False,  # model not active
                True,  # model tab disabled initially
                len(app.state_manager.fitted_models)
                < 2,  # compare tab disabled until 2+ models
                "TRANSFORMATION",
                mode_style,
            )

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "mode-model":
            # Check if split has been done
            if not app.state_manager.is_split_done():
                # Can't switch to modeling without split
                actions = get_action_options("transformation")
                mode_style = {
                    "padding": "15px",
                    "borderRadius": "8px",
                    "backgroundColor": "#e3f2fd",
                    "border": "2px solid #2196f3",
                    "marginTop": "10px",
                    "marginBottom": "10px",
                }
                return (
                    actions,
                    None,
                    "transformation",
                    True,
                    False,
                    True,
                    len(app.state_manager.fitted_models) < 2,
                    "TRANSFORMATION",
                    mode_style,
                )

            new_mode = "modeling"
            app.state_manager.switch_mode(new_mode)
            actions = get_action_options("modeling")
            mode_style = {
                "padding": "15px",
                "borderRadius": "8px",
                "backgroundColor": "#e8f5e9",
                "border": "2px solid #4caf50",
                "marginTop": "10px",
                "marginBottom": "10px",
            }
            return (
                actions,
                None,
                new_mode,
                False,  # transform not active
                True,  # model active
                False,  # model tab enabled
                len(app.state_manager.fitted_models)
                < 2,  # compare tab disabled until 2+ models
                "MODELING",
                mode_style,
            )
        else:
            new_mode = "transformation"
            app.state_manager.switch_mode(new_mode)
            actions = get_action_options("transformation")
            mode_style = {
                "padding": "15px",
                "borderRadius": "8px",
                "backgroundColor": "#e3f2fd",
                "border": "2px solid #2196f3",
                "marginTop": "10px",
                "marginBottom": "10px",
            }
            return (
                actions,
                None,
                new_mode,
                True,
                False,
                not app.state_manager.current_model,  # Enable model tab if model exists
                len(app.state_manager.fitted_models)
                < 2,  # compare tab disabled until 2+ models
                "TRANSFORMATION",
                mode_style,
            )

    @app.callback(
        [
            Output("action-params-form", "children"),
            Output("execute-button", "disabled"),
        ],
        [Input("action-selector", "value")],
        [State("current-mode", "data")],
    )
    def update_params_form(action_name, mode):
        """Update parameter form based on selected action."""
        if not action_name:
            return html.Div("Select an action to see parameters"), True

        try:
            action_class = ActionRegistry.get_action_class(action_name)
            action = action_class()
            params = action.get_parameters()

            if not params:
                return html.Div("This action has no parameters"), False

            form_elements = []

            for param in params:
                label = html.Label(
                    param.label, style={"fontWeight": "bold", "marginTop": "10px"}
                )

                if param.type == "columns":
                    # Multi-select for columns
                    suggestions = action.suggest_columns(app.state_manager.df)
                    all_cols = list(app.state_manager.df.columns)

                    element = dcc.Dropdown(
                        id={"type": "param", "name": param.name},
                        options=[{"label": col, "value": col} for col in all_cols],
                        value=suggestions if suggestions else None,
                        multi=True,
                        placeholder=f"Select {param.label.lower()}...",
                    )

                elif param.type == "column":
                    # Single select for column
                    suggestions = action.suggest_columns(app.state_manager.df)
                    all_cols = list(app.state_manager.df.columns)

                    element = dcc.Dropdown(
                        id={"type": "param", "name": param.name},
                        options=[{"label": col, "value": col} for col in all_cols],
                        value=suggestions[0] if suggestions else None,
                        placeholder=f"Select {param.label.lower()}...",
                    )

                elif param.type == "select":
                    # Dropdown for options
                    element = dcc.Dropdown(
                        id={"type": "param", "name": param.name},
                        options=[
                            {"label": str(opt), "value": opt} for opt in param.options
                        ],
                        value=param.default,
                        clearable=False,
                    )

                elif param.type == "numeric":
                    # Numeric input
                    element = dbc.Input(
                        id={"type": "param", "name": param.name},
                        type="number",
                        value=param.default,
                        step=0.01 if isinstance(param.default, float) else 1,
                    )

                elif param.type == "boolean":
                    # Checkbox
                    element = dbc.Checkbox(
                        id={"type": "param", "name": param.name}, value=param.default
                    )

                elif param.type == "text":
                    # Text input
                    element = dbc.Input(
                        id={"type": "param", "name": param.name},
                        type="text",
                        value=param.default,
                    )

                else:
                    element = html.Div(f"Unknown parameter type: {param.type}")

                # Add description
                desc = html.Small(param.description, className="form-text text-muted")

                form_elements.extend([label, element, desc])

            return html.Div(form_elements), False

        except Exception as e:
            return html.Div(f"Error loading parameters: {str(e)}"), True

    @app.callback(
        [
            Output("action-status", "children"),
            Output("refresh-trigger", "data"),
            Output("action-spinner", "style"),
        ],
        [Input("execute-button", "n_clicks")],
        [
            State("action-selector", "value"),
            State({"type": "param", "name": ALL}, "id"),
            State({"type": "param", "name": ALL}, "value"),
            State("refresh-trigger", "data"),
        ],
    )
    def execute_action(n_clicks, action_name, param_ids, param_values, refresh_count):
        """Execute the selected action."""
        if not n_clicks or not action_name:
            return "", refresh_count, {"display": "none"}

        # Show spinner while processing
        spinner_style = {"display": "block", "marginTop": "10px", "textAlign": "center"}

        try:
            # Build params dictionary
            params = {}
            for param_id, value in zip(param_ids, param_values):
                param_name = param_id["name"]
                params[param_name] = value

            # Get action instance
            action_class = ActionRegistry.get_action_class(action_name)
            action = action_class()

            # Validate
            is_valid, error_msg = action.validate(
                app.state_manager.df,
                params,
                app.state_manager.train_df,
                app.state_manager.test_df,
            )

            if not is_valid:
                return (
                    dbc.Alert(f"[X] Validation Error: {error_msg}", color="danger"),
                    refresh_count,
                    spinner_style,
                )

            # Execute
            result = action.execute(
                app.state_manager.df,
                params,
                app.state_manager.train_df,
                app.state_manager.test_df,
            )

            # Apply to state
            success = app.state_manager.apply_action(action_name, params, result)

            if success:
                msg = dbc.Alert(
                    f"[OK] Successfully executed: {action_name}", color="success"
                )
                return msg, refresh_count + 1, {"display": "none"}
            else:
                return (
                    dbc.Alert("[X] Failed to apply action", color="danger"),
                    refresh_count,
                    {"display": "none"},
                )

        except Exception as e:
            return (
                dbc.Alert(f"[X] Error: {str(e)}", color="danger"),
                refresh_count,
                {"display": "none"},
            )

    @app.callback(
        Output("history-list", "children"), [Input("refresh-trigger", "data")]
    )
    def update_history(trigger):
        """Update history display."""
        history = app.state_manager.get_history_summary()
        checkpoints = app.state_manager.get_checkpoints()

        if not history:
            return html.Div("No actions yet", className="text-muted")

        items = []
        for entry in history:
            # Check if there are checkpoints at this position
            checkpoint_markers = []
            for cp_name, cp_idx in checkpoints.items():
                if cp_idx == entry["index"] - 1:  # Checkpoint before this action
                    checkpoint_markers.append(
                        html.Div(
                            f"[*] {cp_name}",
                            style={
                                "color": "#28a745",
                                "fontWeight": "bold",
                                "fontSize": "13px",
                                "marginBottom": "5px",
                            },
                        )
                    )

            items.extend(checkpoint_markers)

            items.append(
                html.Div(
                    [
                        html.Span(f"{entry['index']}. ", style={"fontWeight": "bold"}),
                        html.Span(entry["action"]),
                        html.Span(
                            f" ({entry['timestamp']})",
                            className="text-muted",
                            style={"fontSize": "11px"},
                        ),
                    ],
                    style={"marginBottom": "8px"},
                )
            )

        return html.Div(items)

    @app.callback(
        Output("checkpoint-name", "value"),
        [Input("create-checkpoint-btn", "n_clicks")],
        [State("checkpoint-name", "value")],
    )
    def create_checkpoint(n_clicks, name):
        """Create a new checkpoint."""
        if not n_clicks or not name:
            return ""

        success = app.state_manager.create_checkpoint(name)
        if success:
            return ""  # Clear input
        return name  # Keep input if failed

    @app.callback(
        Output("download-script", "data"), [Input("export-script-btn", "n_clicks")]
    )
    def export_script(n_clicks):
        """Export Python script."""
        if not n_clicks:
            return None

        script = app.session_logger.generate_script()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"treelab_session_{timestamp}.py"

        return dict(content=script, filename=filename)

    @app.callback(
        Output("download-bq-script", "data"), [Input("export-bq-btn", "n_clicks")]
    )
    def export_bigquery_script(n_clicks):
        """Export BigQuery SQL script."""
        if not n_clicks:
            return None

        script = app.session_logger.generate_bigquery_script()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"treelab_bigquery_{timestamp}.sql"

        return dict(content=script, filename=filename)

    @app.callback(
        Output("tabs", "active_tab"),
        [Input({"type": "col-dist-btn", "column": ALL}, "n_clicks")],
        [State({"type": "col-dist-btn", "column": ALL}, "id")],
    )
    def handle_column_dist_button(n_clicks_list, button_ids):
        """Handle column distribution button clicks to navigate to distributions tab."""
        if not n_clicks_list or not any(n_clicks_list):
            raise PreventUpdate

        # Find which button was clicked
        clicked_idx = None
        for idx, n_clicks in enumerate(n_clicks_list):
            if n_clicks:
                clicked_idx = idx
                break

        if clicked_idx is not None:
            return "tab-dist"

        raise PreventUpdate

    @app.callback(
        Output("tab-content", "children"),
        [Input("tabs", "active_tab"), Input("refresh-trigger", "data")],
    )
    def render_tab_content(active_tab, trigger):
        """Render content for active tab."""
        try:
            if active_tab == "tab-data":
                return render_data_tab()
            elif active_tab == "tab-stats":
                return render_stats_tab()
            elif active_tab == "tab-dist":
                return render_distributions_tab()
            elif active_tab == "tab-corr":
                return render_correlations_tab()
            elif active_tab == "tab-model":
                return render_model_results_tab()
            elif active_tab == "tab-compare":
                return render_model_comparison_tab()
            elif active_tab == "tab-history":
                return render_history_tree_tab()
            elif active_tab == "tab-help":
                return render_help_tab()
            else:
                return html.Div("Select a tab")
        except Exception as e:
            return dbc.Alert(f"Error rendering tab: {str(e)}", color="danger")

    def render_data_tab():
        """Render data view tab."""
        from treelab.ui.layout import create_data_table

        df = app.state_manager.df
        return create_data_table(df)

    def render_stats_tab():
        """Render statistics tab."""
        df = app.state_manager.df

        # Descriptive statistics
        desc = df.describe(include="all").transpose()
        desc = desc.reset_index()
        desc.columns = ["Column"] + list(desc.columns[1:])

        # Missing values
        missing = pd.DataFrame(
            {
                "Column": df.columns,
                "Missing Count": df.isnull().sum().values,
                "Missing %": (df.isnull().sum().values / len(df) * 100).round(2),
            }
        )

        return html.Div(
            [
                html.H5("Descriptive Statistics"),
                dbc.Table.from_dataframe(
                    desc.head(20), striped=True, bordered=True, hover=True, size="sm"
                ),
                html.Hr(),
                html.H5("Missing Values"),
                dbc.Table.from_dataframe(
                    missing, striped=True, bordered=True, hover=True, size="sm"
                ),
            ]
        )

    def render_distributions_tab():
        """Render distributions tab with subplots for all numeric columns."""
        df = app.state_manager.df
        numeric_cols = ColumnAnalyzer.get_numeric_columns(df)

        if not numeric_cols:
            return html.Div("No numeric columns to visualize")

        return html.Div(
            [
                html.P(
                    f"Showing distributions for {len(numeric_cols)} numeric columns",
                    className="text-muted small mb-3",
                ),
                dcc.Dropdown(
                    id="dist-columns-selector",
                    options=[{"label": col, "value": col} for col in numeric_cols],
                    value=numeric_cols,
                    multi=True,
                    placeholder="Select columns to show...",
                ),
                html.Br(),
                dcc.Graph(id="dist-graph"),
            ]
        )

    @app.callback(
        Output("dist-graph", "figure"),
        [
            Input("dist-columns-selector", "value"),
            Input("refresh-trigger", "data"),
        ],
    )
    def update_distributions(selected_columns, trigger):
        """Update distribution plots based on selected columns."""
        df = app.state_manager.df
        numeric_cols = ColumnAnalyzer.get_numeric_columns(df)

        if not numeric_cols:
            fig = go.Figure()
            fig.add_annotation(text="No numeric columns available", showarrow=False)
            fig.update_layout(plot_bgcolor="white")
            return fig

        if not selected_columns:
            selected = numeric_cols
        else:
            selected = (
                selected_columns
                if isinstance(selected_columns, list)
                else [selected_columns]
            )
            selected = [col for col in selected if col in numeric_cols]

        if not selected:
            fig = go.Figure()
            fig.add_annotation(text="No numeric columns selected", showarrow=False)
            fig.update_layout(plot_bgcolor="white")
            return fig

        ncols = min(3, len(selected))
        nrows = (len(selected) + ncols - 1) // ncols

        from plotly.subplots import make_subplots

        vertical_spacing = 0.15 if nrows <= 1 else min(0.12, 1 / (nrows - 1))

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[
                f"{col}<br><sub style='font-size:10px'>Mean: {df[col].mean():.2f} | Med: {df[col].median():.2f}<br>Std: {df[col].std():.2f} | Missing: {df[col].isna().sum()}</sub>"
                for col in selected
            ],
            vertical_spacing=vertical_spacing,
            horizontal_spacing=0.1,
        )

        for idx, col in enumerate(selected):
            row = idx // ncols + 1
            col_idx = idx % ncols + 1

            fig.add_trace(
                go.Histogram(
                    x=df[col].dropna(),
                    nbinsx=30,
                    name=col,
                    showlegend=False,
                    hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
                ),
                row=row,
                col=col_idx,
            )

            mean_val = df[col].mean()
            median_val = df[col].median()

            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                line_width=1,
                row=row,
                col=col_idx,
            )

            fig.add_vline(
                x=median_val,
                line_dash="dot",
                line_color="green",
                line_width=1,
                row=row,
                col=col_idx,
            )

        fig.update_layout(
            showlegend=False,
            hovermode="closest",
            height=nrows * 300,
            plot_bgcolor="white",
        )

        for i in range(1, len(selected) + 1):
            row = (i - 1) // ncols + 1
            col_idx = (i - 1) % ncols + 1
            fig.update_xaxes(
                showgrid=True,
                gridcolor="lightgray",
                row=row,
                col=col_idx,
            )
            fig.update_yaxes(
                showgrid=True,
                gridcolor="lightgray",
                row=row,
                col=col_idx,
            )

        return fig

    def render_correlations_tab():
        """Render correlations tab with multiple correlation methods."""
        df = app.state_manager.df
        numeric_df = df.select_dtypes(include=["number"])

        if numeric_df.shape[1] < 2:
            return html.Div("Need at least 2 numeric columns for correlation")

        # Calculate all correlations
        pearson_corr = numeric_df.corr(method="pearson")
        spearman_corr = numeric_df.corr(method="spearman")
        kendall_corr = numeric_df.corr(method="kendall")

        # Create dropdown to select correlation type (store in hidden component)
        return html.Div(
            [
                html.P("Select correlation method:", className="small text-muted"),
                dcc.RadioItems(
                    id="corr-method-selector",
                    options=[
                        {"label": "Pearson (Linear)", "value": "pearson"},
                        {
                            "label": "Spearman (Non-linear monotonic)",
                            "value": "spearman",
                        },
                        {"label": "Kendall (Rank-based robust)", "value": "kendall"},
                    ],
                    value="pearson",
                    inline=True,
                ),
                dcc.Graph(id="corr-graph"),
                html.Div(id="corr-stats", className="mt-3"),
            ]
        )

    @app.callback(
        [Output("corr-graph", "figure"), Output("corr-stats", "children")],
        [Input("corr-method-selector", "value")],
    )
    def update_correlations(method):
        """Update correlation plot based on selected method."""
        df = app.state_manager.df
        numeric_df = df.select_dtypes(include=["number"])

        if numeric_df.shape[1] < 2:
            return go.Figure(), html.Div("Need at least 2 numeric columns")

        # Calculate correlation based on method
        if method == "pearson":
            corr = numeric_df.corr(method="pearson")
            method_name = "Pearson (Linear Relationship)"
            method_desc = "Measures linear relationships between variables. Best for normally distributed data."
        elif method == "spearman":
            corr = numeric_df.corr(method="spearman")
            method_name = "Spearman (Rank-based)"
            method_desc = "Measures monotonic relationships using rank values. Robust to outliers and non-normal data."
        else:
            corr = numeric_df.corr(method="kendall")
            method_name = "Kendall (Robust Rank-based)"
            method_desc = "Measures ordinal association based on concordant/discordant pairs. Best for smaller datasets."

        # Calculate pairwise stats (upper triangle only)
        pairs = []
        columns = list(corr.columns)
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                coef = float(corr.iloc[i, j])
                pairs.append((columns[i], columns[j], coef))

        strong_positive = [p for p in pairs if p[2] > 0.7]
        strong_negative = [p for p in pairs if p[2] < -0.7]
        moderate_positive = [p for p in pairs if 0.4 < p[2] <= 0.7]
        moderate_negative = [p for p in pairs if -0.7 <= p[2] < -0.4]

        def build_pairs_table(items, title):
            if not items:
                return html.Div([html.H6(title), html.P("None", className="small")])

            df_pairs = pd.DataFrame(
                {
                    "Pair": [f"{a} vs {b}" for a, b, _ in items],
                    "Corr": [f"{coef:.2f}" for _, _, coef in items],
                }
            )
            return html.Div(
                [
                    html.H6(title),
                    dbc.Table.from_dataframe(
                        df_pairs, striped=True, bordered=True, hover=True, size="sm"
                    ),
                ]
            )

        # Create heatmap
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title=f"{method_name}<br><sub>{method_desc}</sub>",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            labels=dict(color="Correlation"),
        )

        fig.update_layout(
            xaxis_title="Features",
            yaxis_title="Features",
            plot_bgcolor="white",
            font=dict(size=10),
            height=min(600, 100 + numeric_df.shape[1] * 40),
        )

        fig.update_traces(
            textfont=dict(size=9),
            hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
        )

        # Stats box
        stats = dbc.Card(
            dbc.CardBody(
                [
                    html.H6("Correlation Statistics", className="mb-3"),
                    html.P(
                        f"Strong positive (>0.7): {len(strong_positive)} pairs",
                        className="small",
                    ),
                    html.P(
                        f"Strong negative (<-0.7): {len(strong_negative)} pairs",
                        className="small",
                    ),
                    html.P(
                        f"Moderate positive (0.4-0.7): {len(moderate_positive)} pairs",
                        className="small",
                    ),
                    html.P(
                        f"Moderate negative (-0.7 to -0.4): {len(moderate_negative)} pairs",
                        className="small",
                    ),
                    html.Hr(),
                    build_pairs_table(strong_positive, "Strong Positive Pairs"),
                    build_pairs_table(strong_negative, "Strong Negative Pairs"),
                    build_pairs_table(moderate_positive, "Moderate Positive Pairs"),
                    build_pairs_table(moderate_negative, "Moderate Negative Pairs"),
                ]
            ),
            className="mt-2",
        )

        return fig, stats

    def render_model_results_tab():
        """Render model results tab."""
        if not app.state_manager.current_model:
            return html.Div("No model fitted yet", className="text-muted")

        metadata = app.state_manager.model_metadata
        task = metadata.get("task", "classification")

        tuning_block = None
        if "best_params" in metadata:
            best_params = metadata.get("best_params", {})
            best_score = metadata.get("best_score")
            params_df = pd.DataFrame(
                {
                    "Parameter": list(best_params.keys()),
                    "Value": [str(v) for v in best_params.values()],
                }
            )
            tuning_block = html.Div(
                [
                    html.H6("Tuning Results"),
                    html.P(
                        f"Best score ({metadata.get('scoring', 'score')}): {best_score:.4f}"
                        if best_score is not None
                        else "Best score unavailable",
                        className="text-muted",
                    ),
                    dbc.Table.from_dataframe(
                        params_df, striped=True, bordered=True, hover=True, size="sm"
                    ),
                    html.Hr(),
                ]
            )

        metrics_df = None
        if task == "regression":
            metrics_df = pd.DataFrame(
                {
                    "Metric": ["Train R2", "Test R2", "MAE", "RMSE"],
                    "Value": [
                        f"{metadata['train_r2']:.4f}",
                        f"{metadata['test_r2']:.4f}",
                        f"{metadata['test_mae']:.4f}",
                        f"{metadata['test_rmse']:.4f}",
                    ],
                }
            )
        elif task == "classification":
            metrics_df = pd.DataFrame(
                {
                    "Metric": [
                        "Train Accuracy",
                        "Test Accuracy",
                        "Precision",
                        "Recall",
                        "F1-Score",
                    ],
                    "Value": [
                        f"{metadata['train_accuracy']:.4f}",
                        f"{metadata['test_accuracy']:.4f}",
                        f"{metadata['test_precision']:.4f}",
                        f"{metadata['test_recall']:.4f}",
                        f"{metadata['test_f1']:.4f}",
                    ],
                }
            )

        fig_cm = None
        regression_scatter = None

        if task == "classification":
            cm = metadata["confusion_matrix"]
            total_samples = sum(sum(row) for row in cm)
            accuracy = metadata["test_accuracy"]

            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=metadata["classes"],
                y=metadata["classes"],
                text_auto=True,
                title=f"Confusion Matrix<br><sub>Accuracy: {accuracy:.1%} | Total Samples: {total_samples}</sub>",
                color_continuous_scale="Blues",
            )

            fig_cm.update_traces(
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            )

            fig_cm.update_layout(plot_bgcolor="white", font=dict(size=11))
        elif task == "regression":
            y_test = metadata.get("y_test", [])
            y_pred = metadata.get("y_pred", [])
            scatter_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

            regression_scatter = px.scatter(
                scatter_df,
                x="Actual",
                y="Predicted",
                title="Actual vs Predicted",
                trendline="ols",
            )
            regression_scatter.update_traces(
                hovertemplate="Actual: %{x}<br>Predicted: %{y}<extra></extra>"
            )
            regression_scatter.update_layout(plot_bgcolor="white", font=dict(size=11))

        fig_imp = None
        if "feature_importance" in metadata:
            feat_imp = metadata["feature_importance"]
            feat_imp_df = (
                pd.DataFrame(
                    {
                        "Feature": list(feat_imp.keys()),
                        "Importance": list(feat_imp.values()),
                    }
                )
                .sort_values("Importance", ascending=False)
                .head(15)
            )

            feat_imp_df["Cumulative"] = feat_imp_df["Importance"].cumsum()
            total_importance_shown = feat_imp_df["Importance"].sum()

            fig_imp = px.bar(
                feat_imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title=f"Top 15 Feature Importances<br><sub>Showing {total_importance_shown:.1%} of total importance | {len(feat_imp)} features total</sub>",
                labels={"Importance": "Importance Score"},
                color="Importance",
                color_continuous_scale="Viridis",
            )

            fig_imp.update_layout(
                plot_bgcolor="white",
                xaxis=dict(
                    showgrid=True,
                    gridcolor="lightgray",
                    range=[0, max(feat_imp_df["Importance"]) * 1.1],
                ),
                yaxis=dict(categoryorder="total ascending"),
                showlegend=False,
            )

            fig_imp.update_traces(
                hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>"
            )

        feature_names = []
        y_train = None
        if app.state_manager.train_df is not None:
            target = metadata.get("target_column")
            feature_names = [
                col for col in app.state_manager.train_df.columns if col != target
            ]
            if target and target in app.state_manager.train_df.columns:
                y_train = app.state_manager.train_df[target].values

        tree_fig = build_tree_figure(
            app.state_manager.current_model, feature_names, max_depth=3, y_train=y_train
        )
        tree_block = (
            html.Div([html.H6("Tree Visualization"), dcc.Graph(figure=tree_fig)])
            if tree_fig is not None
            else html.Div()
        )

        shap_block = None
        if "shap_importance" in metadata:
            shap_imp = metadata["shap_importance"]
            shap_df = (
                pd.DataFrame(
                    {
                        "Feature": list(shap_imp.keys()),
                        "Importance": list(shap_imp.values()),
                    }
                )
                .sort_values("Importance", ascending=False)
                .head(15)
            )

            fig_shap = px.bar(
                shap_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title=f"SHAP Summary (Top 15)<br><sub>Computed on {metadata.get('shap_samples', 0)} samples</sub>",
                labels={"Importance": "Mean |SHAP|"},
                color="Importance",
                color_continuous_scale="Cividis",
            )
            fig_shap.update_layout(
                plot_bgcolor="white",
                yaxis=dict(categoryorder="total ascending"),
                showlegend=False,
            )
            fig_shap.update_traces(
                hovertemplate="%{y}<br>Mean |SHAP|: %{x:.4f}<extra></extra>"
            )

            shap_block = html.Div([html.H6("SHAP Summary"), dcc.Graph(figure=fig_shap)])

            if (
                "shap_values" in metadata
                and "shap_feature_values" in metadata
                and "shap_feature_names" in metadata
            ):
                shap_values = np.array(metadata["shap_values"])
                feature_values = np.array(metadata["shap_feature_values"])
                feature_names = metadata["shap_feature_names"]

                if shap_values.ndim == 3:
                    shap_values = shap_values.mean(axis=-1)

                if shap_values.ndim == 1:
                    shap_values = shap_values.reshape(-1, 1)
                if feature_values.ndim == 1:
                    feature_values = feature_values.reshape(-1, 1)

                mean_abs = np.abs(shap_values).mean(axis=0)
                top_idx = np.argsort(mean_abs)[-10:][::-1]

                beeswarm_rows = []
                rng = np.random.RandomState(42)
                for rank, idx in enumerate(top_idx):
                    shap_col = shap_values[:, idx]
                    feat_col = feature_values[:, idx]
                    jitter = rng.uniform(-0.2, 0.2, size=len(shap_col))
                    for s, v, j in zip(shap_col, feat_col, jitter):
                        beeswarm_rows.append(
                            {
                                "feature": feature_names[idx],
                                "shap": s,
                                "value": v,
                                "y": rank + j,
                            }
                        )

                beeswarm_df = pd.DataFrame(beeswarm_rows)
                fig_beeswarm = px.scatter(
                    beeswarm_df,
                    x="shap",
                    y="feature",
                    color="value",
                    title="SHAP Summary (Beeswarm)",
                    color_continuous_scale="RdBu_r",
                )
                fig_beeswarm.update_layout(plot_bgcolor="white", yaxis_title="")

                sample_idx = 0
                shap_sample = shap_values[sample_idx, :]
                waterfall_df = pd.DataFrame(
                    {
                        "feature": [feature_names[i] for i in top_idx],
                        "contribution": shap_sample[top_idx],
                    }
                ).sort_values("contribution", key=lambda x: np.abs(x), ascending=False)

                fig_waterfall = px.bar(
                    waterfall_df,
                    x="contribution",
                    y="feature",
                    orientation="h",
                    title="SHAP Waterfall (Sample 1)",
                    color="contribution",
                    color_continuous_scale="RdBu_r",
                )
                fig_waterfall.update_layout(plot_bgcolor="white", yaxis_title="")

                shap_block = html.Div(
                    [
                        html.H6("SHAP Summary"),
                        dcc.Graph(figure=fig_beeswarm),
                        html.Hr(),
                        dcc.Graph(figure=fig_waterfall),
                    ]
                )

        scorecard_block = None
        if "scorecards" in metadata:
            scorecard_sections = []
            for entry in metadata["scorecards"]:
                scorecard_df = pd.DataFrame(entry["table"])
                if "EventRate" in scorecard_df.columns:
                    scorecard_df["EventRate"] = (scorecard_df["EventRate"] * 100).round(
                        3
                    ).astype(str) + "%"
                scorecard_sections.extend(
                    [
                        html.H6(f"Scorecard: {entry['feature']}"),
                        html.P(
                            f"Total IV: {entry['total_iv']:.4f}",
                            className="text-muted",
                        ),
                        dbc.Table.from_dataframe(
                            scorecard_df,
                            striped=True,
                            bordered=True,
                            hover=True,
                            size="sm",
                        ),
                        html.Hr(),
                    ]
                )

            scorecard_block = html.Div(
                [
                    html.H6("Binning Scorecards"),
                    html.P(
                        f"Total IV (sum): {metadata.get('scorecard_total_iv', 0):.4f}",
                        className="text-muted",
                    ),
                    *scorecard_sections,
                ]
            )

        return html.Div(
            [
                html.H5(f"Model: {metadata['model_type']}"),
                html.Hr(),
                tuning_block if tuning_block else html.Div(),
                html.H6("Performance Metrics")
                if metrics_df is not None
                else html.Div(),
                dbc.Table.from_dataframe(
                    metrics_df, striped=True, bordered=True, hover=True
                )
                if metrics_df is not None
                else html.Div(),
                html.Hr(),
                dcc.Graph(figure=fig_cm)
                if fig_cm is not None
                else (
                    dcc.Graph(figure=regression_scatter)
                    if regression_scatter is not None
                    else html.Div()
                ),
                html.Hr(),
                tree_block,
                html.Hr(),
                dcc.Graph(figure=fig_imp) if fig_imp is not None else html.Div(),
                html.Hr() if scorecard_block is not None else html.Div(),
                scorecard_block if scorecard_block is not None else html.Div(),
                html.Hr() if shap_block is not None else html.Div(),
                shap_block if shap_block is not None else html.Div(),
            ]
        )

    def render_model_comparison_tab():
        """Render model comparison tab."""
        fitted_models = app.state_manager.get_fitted_models()

        if not fitted_models:
            return html.Div(
                [
                    html.H5("Model Comparison"),
                    html.P(
                        "No models fitted yet. Fit multiple models to compare them.",
                        className="text-muted",
                    ),
                ]
            )

        if len(fitted_models) < 2:
            return html.Div(
                [
                    html.H5("Model Comparison"),
                    html.P(
                        "Fit at least 2 models to compare them.",
                        className="text-muted",
                    ),
                ]
            )

        model_types = [m.get("action_name", "") for m in fitted_models]
        all_same_type = len(set(model_types)) == 1

        model_names = [
            m.get("name", f"Model {i + 1}") for i, m in enumerate(fitted_models)
        ]

        first_metadata = fitted_models[0].get("metadata", {})
        task = first_metadata.get("task", "classification")

        best_idx = 0
        best_score = 0
        for i, m in enumerate(fitted_models):
            meta = m.get("metadata", {})
            if task == "classification":
                score = meta.get("test_accuracy", 0)
            else:
                score = meta.get("test_r2", 0)
            if score > best_score:
                best_score = score
                best_idx = i

        response = [
            html.H4(f"Model Comparison ({len(fitted_models)} models)"),
            dbc.Alert(
                f"🏆 Winner: {model_names[best_idx]} (Best {task.capitalize()} Performance: {best_score:.2%})",
                color="success",
                className="mb-3",
            ),
            html.P(
                f"Comparing {'same model type' if all_same_type else 'different model types'}.",
                className="text-muted",
            ),
            html.Hr(),
        ]

        if task == "classification":
            metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
            metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

            radar_data = []
            for i, model_info in enumerate(fitted_models):
                metadata = model_info.get("metadata", {})
                for j, (metric, label) in enumerate(zip(metrics, metric_labels)):
                    radar_data.append(
                        {
                            "Model": model_names[i],
                            "Metric": label,
                            "Value": metadata.get(metric, 0) * 100,
                        }
                    )

            radar_df = pd.DataFrame(radar_data)
            fig_radar = px.line_polar(
                radar_df,
                r="Value",
                theta="Metric",
                color="Model",
                line_close=True,
                title="Multi-Metric Comparison (Radar Chart)",
                range_r=[0, 100],
            )
            fig_radar.update_traces(fill="toself")
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)))

            response.append(
                dbc.Row(
                    [
                        dbc.Col(
                            [html.H6("Performance Radar"), dcc.Graph(figure=fig_radar)],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.H6("Metrics Summary"),
                                dbc.Table(
                                    [
                                        html.Thead(
                                            [
                                                html.Tr(
                                                    [
                                                        html.Th("Model"),
                                                        html.Th("Acc"),
                                                        html.Th("Prec"),
                                                        html.Th("Rec"),
                                                        html.Th("F1"),
                                                    ]
                                                )
                                            ]
                                        ),
                                        html.Tbody(
                                            [
                                                html.Tr(
                                                    [
                                                        html.Td(
                                                            f"{'⭐ ' if i == best_idx else ''}{model_names[i]}"
                                                        ),
                                                        html.Td(
                                                            f"{m.get('metadata', {}).get('test_accuracy', 0):.2%}"
                                                        ),
                                                        html.Td(
                                                            f"{m.get('metadata', {}).get('test_precision', 0):.2%}"
                                                        ),
                                                        html.Td(
                                                            f"{m.get('metadata', {}).get('test_recall', 0):.2%}"
                                                        ),
                                                        html.Td(
                                                            f"{m.get('metadata', {}).get('test_f1', 0):.2%}"
                                                        ),
                                                    ]
                                                )
                                                for i, m in enumerate(fitted_models)
                                            ]
                                        ),
                                    ],
                                    striped=True,
                                    bordered=True,
                                    hover=True,
                                    size="sm",
                                ),
                            ],
                            width=6,
                        ),
                    ]
                )
            )
        else:
            metrics = ["test_r2", "test_mae", "test_rmse"]
            metric_labels = ["R²", "MAE", "RMSE"]

            radar_data = []
            for i, model_info in enumerate(fitted_models):
                metadata = model_info.get("metadata", {})
                for j, (metric, label) in enumerate(zip(metrics, metric_labels)):
                    val = metadata.get(metric, 0)
                    if metric == "test_mae" or metric == "test_rmse":
                        radar_data.append(
                            {"Model": model_names[i], "Metric": label, "Value": -val}
                        )
                    else:
                        radar_data.append(
                            {
                                "Model": model_names[i],
                                "Metric": label,
                                "Value": val * 100,
                            }
                        )

            radar_df = pd.DataFrame(radar_data)
            fig_radar = px.line_polar(
                radar_df,
                r="Value",
                theta="Metric",
                color="Model",
                line_close=True,
                title="Regression Metrics Comparison",
            )
            fig_radar.update_traces(fill="toself")

            response.append(
                dbc.Row(
                    [
                        dbc.Col(
                            [html.H6("Performance Radar"), dcc.Graph(figure=fig_radar)],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.H6("Metrics Summary"),
                                dbc.Table(
                                    [
                                        html.Thead(
                                            [
                                                html.Tr(
                                                    [
                                                        html.Th("Model"),
                                                        html.Th("R²"),
                                                        html.Th("MAE"),
                                                        html.Th("RMSE"),
                                                    ]
                                                )
                                            ]
                                        ),
                                        html.Tbody(
                                            [
                                                html.Tr(
                                                    [
                                                        html.Td(
                                                            f"{'⭐ ' if i == best_idx else ''}{model_names[i]}"
                                                        ),
                                                        html.Td(
                                                            f"{m.get('metadata', {}).get('test_r2', 0):.4f}"
                                                        ),
                                                        html.Td(
                                                            f"{m.get('metadata', {}).get('test_mae', 0):.4f}"
                                                        ),
                                                        html.Td(
                                                            f"{m.get('metadata', {}).get('test_rmse', 0):.4f}"
                                                        ),
                                                    ]
                                                )
                                                for i, m in enumerate(fitted_models)
                                            ]
                                        ),
                                    ],
                                    striped=True,
                                    bordered=True,
                                    hover=True,
                                    size="sm",
                                ),
                            ],
                            width=6,
                        ),
                    ]
                )
            )

        response.append(html.Hr())

        # Add performance chart
        response.append(html.Hr())

        # Model-agnostic info (works for any model type)
        response.append(html.H6("Common Metrics"))

        common_metrics_data = []
        for i, model_info in enumerate(fitted_models):
            metadata = model_info.get("metadata", {})
            task = metadata.get("task", "classification")
            params = model_info.get("params", {})

            row = {
                "#": i + 1,
                "Name": model_info.get("name", f"Model {i + 1}"),
                "Task": task.capitalize(),
                "Target": metadata.get("target_column", "N/A"),
            }

            # Add task-specific primary metric
            if task == "classification":
                row["Primary Metric"] = f"{metadata.get('test_accuracy', 0):.4f}"
            else:
                row["Primary Metric"] = f"{metadata.get('test_r2', 0):.4f}"

            common_metrics_data.append(row)

        common_df = pd.DataFrame(common_metrics_data)
        response.append(
            dbc.Table.from_dataframe(common_df, striped=True, bordered=True, hover=True)
        )
        response.append(html.Hr())

        # Parameter comparison (only for same model types)
        if all_same_type:
            response.append(html.H6("Parameter Comparison"))

            # Build parameter comparison table
            all_params = set()
            for m in fitted_models:
                params = m.get("params", {})
                # Filter out non-model params
                model_params = {
                    k: v
                    for k, v in params.items()
                    if k
                    not in ["model_name", "target_column", "test_size", "random_state"]
                }
                all_params.update(model_params.keys())

            param_comparison = []
            for i, model_info in enumerate(fitted_models):
                params = model_info.get("params", {})
                row = {"Model": model_info.get("name", f"Model {i + 1}")}
                for param in all_params:
                    row[param] = str(params.get(param, "-"))
                param_comparison.append(row)

            param_df = pd.DataFrame(param_comparison)
            if not param_df.empty:
                response.append(
                    dbc.Table.from_dataframe(
                        param_df, striped=True, bordered=True, hover=True
                    )
                )
            else:
                response.append(
                    html.P("No model parameters to compare.", className="text-muted")
                )

            # Highlight differences
            response.append(html.Hr())
            response.append(html.H6("Parameter Differences"))

            differences = []
            for param in all_params:
                values = [m.get("params", {}).get(param, "-") for m in fitted_models]
                if len(set(values)) > 1:  # Different values
                    differences.append(
                        {
                            "Parameter": param,
                            "Model 1": str(values[0]) if len(values) > 0 else "-",
                            "Model 2": str(values[1]) if len(values) > 1 else "-",
                            "Difference": "Yes",
                        }
                    )

            if differences:
                diff_df = pd.DataFrame(differences)
                response.append(
                    dbc.Table.from_dataframe(
                        diff_df,
                        striped=True,
                        bordered=True,
                        hover=True,
                        color="warning",
                    )
                )
            else:
                response.append(
                    html.P("All parameters are identical.", className="text-muted")
                )
        else:
            # Different model types - show detailed params for each
            response.append(html.H6("Model-Specific Parameters"))
            response.append(
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            title=f"{m.get('name', f'Model {i + 1}')} - {m.get('action_name', '')}",
                            children=[
                                html.Div(
                                    [
                                        html.Strong("Parameters: "),
                                        html.Span(str(m.get("params", {}))),
                                    ]
                                ),
                            ],
                        )
                        for i, m in enumerate(fitted_models)
                    ],
                    always_open=True,
                )
            )

        response.append(html.Hr())
        response.append(html.H6("Full Model Details"))
        response.append(
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        title=f"{m.get('name', f'Model {i + 1}')} - {m.get('action_name', '')}",
                        children=[
                            html.Div(
                                [
                                    html.Strong("Parameters: "),
                                    html.Span(str(m.get("params", {}))),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Strong("Target: "),
                                    html.Span(
                                        str(
                                            m.get("metadata", {}).get(
                                                "target_column", "N/A"
                                            )
                                        )
                                    ),
                                ],
                                style={"marginTop": "5px"},
                            ),
                        ],
                    )
                    for i, m in enumerate(fitted_models)
                ],
                always_open=True,
            )
        )

        return html.Div(response)

    def render_history_tree_tab():
        """Render history tree visualization tab."""
        history = app.state_manager.get_history()
        checkpoints = app.state_manager.get_checkpoints()

        if not history:
            return html.Div(
                [
                    html.H5("History Tree"),
                    html.P("No actions executed yet.", className="text-muted"),
                ]
            )

        history_items = []
        for i, record in enumerate(history):
            checkpoint = None
            for cp_name, cp_idx in checkpoints.items():
                if cp_idx == i:
                    checkpoint = cp_name
                    break

            param_str = ", ".join(
                [f"{k}={v}" for k, v in list(record.params.items())[:3]]
            )
            if len(record.params) > 3:
                param_str += "..."

            history_items.append(
                {
                    "step": i + 1,
                    "action": record.action_name,
                    "params": param_str,
                    "time": record.timestamp.strftime("%H:%M:%S"),
                    "checkpoint": checkpoint,
                }
            )

        history_df = pd.DataFrame(history_items)

        return html.Div(
            [
                html.H5("History Tree"),
                html.P(
                    "Visual representation of your data transformation pipeline.",
                    className="text-muted",
                ),
                html.Hr(),
                html.H6("Summary Statistics"),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [html.H4(len(history)), html.P("Total Actions")]
                                )
                            )
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [html.H4(len(checkpoints)), html.P("Checkpoints")]
                                )
                            )
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H4(app.state_manager.df.shape[1]),
                                        html.P("Current Columns"),
                                    ]
                                )
                            )
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H4(app.state_manager.df.shape[0]),
                                        html.P("Current Rows"),
                                    ]
                                )
                            )
                        ),
                    ]
                ),
                html.Hr(),
                html.H6("Action History"),
                dbc.Table.from_dataframe(
                    history_df, striped=True, bordered=True, hover=True
                ),
            ]
        )

    def render_help_tab():
        """Render help documentation tab."""
        return html.Div(
            [
                html.H3("TreeLab Help & Documentation"),
                html.Hr(),
                # ASCII Art Banner
                html.Pre(
                    """
  _____              _          _     
 |_   _| __ ___  ___| |    __ _| |__  
   | || '__/ _ \\/ _ \\ |   / _` | '_ \\ 
   | || | |  __/  __/ |__| (_| | |_) |
   |_||_|  \\___|\\___|_____\\__,_|_.__/ 
                                       
         Interactive Data Science Laboratory
                   Version 0.3.0
            """,
                    style={"fontSize": "10px", "lineHeight": "1.2"},
                ),
                html.Hr(),
                # Transformation Actions
                html.H4("TRANSFORMATION ACTIONS"),
                html.P("These actions modify your dataset in Transformation Mode:"),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Action"),
                                    html.Th("Description"),
                                    html.Th("Use Case"),
                                ]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(html.Strong("DropColumns")),
                                        html.Td(
                                            "Remove specified columns from dataset"
                                        ),
                                        html.Td(
                                            "Remove ID columns, high-cardinality text, or irrelevant features"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("SimpleImputer")),
                                        html.Td(
                                            "Fill missing values using mean, median, mode, or constant"
                                        ),
                                        html.Td(
                                            "Handle missing data before modeling (required by most algorithms)"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("StandardScaler")),
                                        html.Td(
                                            "Standardize features to mean=0, std=1 (z-score normalization)"
                                        ),
                                        html.Td(
                                            "Scale numeric features for better model performance"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("MinMaxScaler")),
                                        html.Td(
                                            "Scale numeric features to a [0, 1] range"
                                        ),
                                        html.Td(
                                            "Normalize numeric features with bounded ranges"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("OneHotEncoder")),
                                        html.Td(
                                            "Convert categorical variables to binary dummy columns"
                                        ),
                                        html.Td(
                                            "Encode categorical features like gender, city, category"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("LabelEncoder")),
                                        html.Td(
                                            "Encode categorical variables as integer labels"
                                        ),
                                        html.Td(
                                            "Ordinal encoding or high-cardinality categories"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("TrainTestSplit")),
                                        html.Td(
                                            "Split dataset into training and test sets"
                                        ),
                                        html.Td(
                                            "Required before model fitting - creates evaluation holdout set"
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    bordered=True,
                    hover=True,
                    striped=True,
                    size="sm",
                ),
                html.Br(),
                # Modeling Actions
                html.H4("MODELING ACTIONS"),
                html.P("These actions train machine learning models (Modeling Mode):"),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Model"),
                                    html.Th("Description"),
                                    html.Th("Best For"),
                                ]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(html.Strong("DecisionTreeClassifier")),
                                        html.Td(
                                            "Single decision tree for classification"
                                        ),
                                        html.Td(
                                            "Interpretable models, understanding decision logic, small datasets"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("RandomForestClassifier")),
                                        html.Td("Ensemble of decision trees (bagging)"),
                                        html.Td(
                                            "Higher accuracy, feature importance, less overfitting than single tree"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("DecisionTreeRegressor")),
                                        html.Td("Single decision tree for regression"),
                                        html.Td(
                                            "Interpretable regression models for numeric targets"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("RandomForestRegressor")),
                                        html.Td("Ensemble of trees for regression"),
                                        html.Td(
                                            "Higher accuracy regression with feature importance"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("PlotFeatureImportance")),
                                        html.Td(
                                            "Fit a model and compute feature importance for visualization"
                                        ),
                                        html.Td(
                                            "Deeper feature importance analysis with permutation importance"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("TuneHyperparameters")),
                                        html.Td(
                                            "Grid search hyperparameters for tree models"
                                        ),
                                        html.Td(
                                            "Find best model settings using cross-validation"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("SHAPSummary")),
                                        html.Td("Compute SHAP feature importance"),
                                        html.Td(
                                            "Explain model predictions with SHAP values"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("BinningScorecard")),
                                        html.Td("Bin feature and compute WOE/IV"),
                                        html.Td(
                                            "Simple scorecard for binary target features"
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    bordered=True,
                    hover=True,
                    striped=True,
                    size="sm",
                ),
                html.Br(),
                # Tabs Documentation
                html.H4("TABS OVERVIEW"),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Tab"),
                                    html.Th("Purpose"),
                                    html.Th("What You'll See"),
                                ]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(html.Strong("[DATA] Data View")),
                                        html.Td("Browse and explore your dataset"),
                                        html.Td(
                                            "Interactive table with sorting, filtering, and search. Shows current state after transformations."
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("[STATS] Statistics")),
                                        html.Td("View descriptive statistics"),
                                        html.Td(
                                            "Mean, std, min, max, quartiles for all columns. Missing value counts and percentages."
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("[DIST] Distributions")),
                                        html.Td("Visualize data distributions"),
                                        html.Td(
                                            "Histograms for numeric columns, bar charts for categorical. Select column from dropdown."
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("[CORR] Correlations")),
                                        html.Td("Analyze feature relationships"),
                                        html.Td(
                                            "Correlation heatmap for numeric columns. Red = negative, blue = positive correlation."
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("[MODEL] Model Results")),
                                        html.Td("Evaluate model performance"),
                                        html.Td(
                                            "Accuracy metrics, confusion matrix, feature importance. Only available after fitting a model."
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Strong("[HELP] Help")),
                                        html.Td("Documentation and guide"),
                                        html.Td(
                                            "This help page! Reference for all actions and tabs."
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    bordered=True,
                    hover=True,
                    striped=True,
                    size="sm",
                ),
                html.Br(),
                # Workflow Guide
                html.H4("TYPICAL WORKFLOW"),
                html.Ol(
                    [
                        html.Li("Explore data in Data View and Statistics tabs"),
                        html.Li("Drop unnecessary columns (IDs, text fields)"),
                        html.Li("Handle missing values with SimpleImputer"),
                        html.Li("Encode categorical variables with OneHotEncoder"),
                        html.Li("Scale numeric features with StandardScaler"),
                        html.Li("Create checkpoint: 'After Preprocessing'"),
                        html.Li(
                            "Split data with TrainTestSplit (target = prediction column)"
                        ),
                        html.Li("Create checkpoint: 'Ready for Modeling'"),
                        html.Li("Switch to Modeling Mode"),
                        html.Li("Fit a model (DecisionTree or RandomForest)"),
                        html.Li("View results in Model Results tab"),
                        html.Li("Export Python script for reproducibility"),
                    ]
                ),
                html.Br(),
                # Tips
                html.H4("PRO TIPS"),
                dbc.Alert(
                    [
                        html.Strong("Smart Suggestions: "),
                        "TreeLab auto-suggests relevant columns for each action based on data types and characteristics.",
                    ],
                    color="info",
                ),
                dbc.Alert(
                    [
                        html.Strong("Checkpoints: "),
                        "Create checkpoints at key stages to save your progress. You can revert to any checkpoint later.",
                    ],
                    color="info",
                ),
                dbc.Alert(
                    [
                        html.Strong("Validation: "),
                        "All parameters are validated before execution. Clear error messages prevent invalid operations.",
                    ],
                    color="info",
                ),
                dbc.Alert(
                    [
                        html.Strong("Export Script: "),
                        "Download a fully executable Python script of your entire workflow for reproducibility.",
                    ],
                    color="info",
                ),
                html.Br(),
                # Version Info
                html.Hr(),
                html.P(
                    [
                        "TreeLab Version: ",
                        html.Strong("0.3.0"),
                        html.Br(),
                        "Built with: Python, Dash, Plotly, scikit-learn, pandas",
                    ],
                    className="text-muted",
                ),
            ],
            style={"padding": "20px"},
        )
