"""Model results visualization components for TreeLab."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_metrics_cards(metadata: Dict[str, Any], task: str) -> dbc.Row:
    """Create metric cards for the model results header."""
    cards = []

    if task == "classification":
        metrics = [
            ("Accuracy", metadata.get("test_accuracy", 0), "%", "primary"),
            ("Precision", metadata.get("test_precision", 0), "%", "info"),
            ("Recall", metadata.get("test_recall", 0), "%", "info"),
            ("F1-Score", metadata.get("test_f1", 0), "%", "success"),
        ]

        # Add model-specific metrics
        model_type = metadata.get("model_type", "")
        if "Tree" in model_type or "Forest" in model_type:
            if "max_depth" in metadata:
                metrics.append(("Tree Depth", metadata["max_depth"], "", "warning"))
            if "n_leaves" in metadata:
                metrics.append(("Leaf Nodes", metadata["n_leaves"], "", "secondary"))
    else:  # regression
        metrics = [
            ("R² Score", metadata.get("test_r2", 0), "", "primary"),
            ("RMSE", metadata.get("test_rmse", 0), "", "danger"),
            ("MAE", metadata.get("test_mae", 0), "", "warning"),
        ]

        if "max_depth" in metadata:
            metrics.append(("Tree Depth", metadata["max_depth"], "", "info"))

    for label, value, suffix, color in metrics:
        if isinstance(value, float):
            display_value = f"{value:.3f}" if suffix == "" else f"{value:.1%}"
        else:
            display_value = str(value)

        card = dbc.Col(
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H6(label, className="card-subtitle text-muted"),
                            html.H3(
                                display_value,
                                className=f"card-title text-{color}",
                                style={"fontWeight": "bold"},
                            ),
                        ]
                    )
                ],
                className="h-100 shadow-sm",
            ),
            width=2,
            className="mb-3",
        )
        cards.append(card)

    return dbc.Row(cards, className="g-3 mb-4")


def create_confusion_matrix_plot(metadata: Dict[str, Any]) -> go.Figure:
    """Create an enhanced confusion matrix heatmap."""
    cm = metadata["confusion_matrix"]
    classes = metadata["classes"]
    accuracy = metadata["test_accuracy"]

    # Calculate percentages
    cm_array = np.array(cm)
    cm_sum = cm_array.sum(axis=1, keepdims=True)
    cm_perc = cm_array / cm_sum.astype(float) * 100

    # Create annotations with both count and percentage
    annotations = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            annotations.append(
                dict(
                    x=classes[j],
                    y=classes[i],
                    text=f"{cm[i][j]}<br>({cm_perc[i][j]:.1f}%)",
                    showarrow=False,
                    font=dict(color="white" if cm[i][j] > np.max(cm) / 2 else "black"),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Count"),
            hovertemplate="<b>Actual:</b> %{y}<br>"
            + "<b>Predicted:</b> %{x}<br>"
            + "<b>Count:</b> %{z}<br>"
            + "<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Confusion Matrix<br><sup>Accuracy: {accuracy:.2%}</sup>",
            x=0.5,
        ),
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        annotations=annotations,
        plot_bgcolor="white",
        height=500,
    )

    return fig


def create_classification_report_table(metadata: Dict[str, Any]) -> dbc.Table:
    """Create a detailed classification report table."""
    classes = metadata["classes"]
    cm = np.array(metadata["confusion_matrix"])

    # Calculate per-class metrics
    rows = []
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        support = cm[i, :].sum()

        rows.append(
            {
                "Class": cls,
                "Precision": f"{precision:.3f}",
                "Recall": f"{recall:.3f}",
                "F1-Score": f"{f1:.3f}",
                "Support": support,
            }
        )

    # Add overall metrics
    rows.append(
        {
            "Class": "**Overall**",
            "Precision": f"{metadata.get('test_precision', 0):.3f}",
            "Recall": f"{metadata.get('test_recall', 0):.3f}",
            "F1-Score": f"{metadata.get('test_f1', 0):.3f}",
            "Support": cm.sum(),
        }
    )

    df = pd.DataFrame(rows)
    return dbc.Table.from_dataframe(
        df, striped=True, bordered=True, hover=True, size="sm"
    )


def create_feature_importance_plot(
    metadata: Dict[str, Any], top_n: int = 15
) -> go.Figure:
    """Create an enhanced feature importance bar chart."""
    if "feature_importance" not in metadata:
        return None

    feat_imp = metadata["feature_importance"]

    # Sort and get top N
    sorted_items = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_items[:top_n]

    df = pd.DataFrame(top_features, columns=["Feature", "Importance"])
    df["Cumulative"] = df["Importance"].cumsum()
    total_shown = df["Importance"].sum()
    total_all = sum(feat_imp.values())

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart
    fig.add_trace(
        go.Bar(
            x=df["Importance"],
            y=df["Feature"],
            orientation="h",
            name="Importance",
            marker=dict(
                color=df["Importance"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Importance", x=1.15),
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Cumulative line
    fig.add_trace(
        go.Scatter(
            x=df["Cumulative"],
            y=df["Feature"],
            mode="lines+markers",
            name="Cumulative",
            line=dict(color="red", width=2),
            marker=dict(size=6),
            hovertemplate="<b>%{y}</b><br>Cumulative: %{x:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.update_layout(
        title=dict(
            text=f"Feature Importance (Top {top_n})<br><sup>Showing {total_shown:.1%} of total importance ({len(feat_imp)} features)</sup>",
            x=0.5,
        ),
        xaxis_title="Importance Score",
        yaxis=dict(categoryorder="total ascending"),
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        margin=dict(l=150),
    )

    fig.update_xaxes(
        showgrid=True, gridcolor="lightgray", range=[0, max(df["Importance"]) * 1.1]
    )

    return fig


def create_regression_scatter_plot(metadata: Dict[str, Any]) -> go.Figure:
    """Create actual vs predicted scatter plot for regression."""
    y_test = metadata.get("y_test", [])
    y_pred = metadata.get("y_pred", [])

    if not y_test or not y_pred:
        return None

    # Calculate residuals
    residuals = np.array(y_test) - np.array(y_pred)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Actual vs Predicted",
            "Residuals vs Predicted",
            "Residuals Distribution",
            "Q-Q Plot",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.15,
    )

    # Actual vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode="markers",
            marker=dict(
                color="blue",
                opacity=0.5,
                size=8,
            ),
            name="Predictions",
            hovertemplate="Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Perfect Prediction",
        ),
        row=1,
        col=1,
    )

    # Residuals vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker=dict(
                color="green",
                opacity=0.5,
                size=8,
            ),
            name="Residuals",
            hovertemplate="Predicted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

    # Residuals histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            name="Residuals",
            marker_color="purple",
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    # Q-Q plot (simplified)
    sorted_residuals = np.sort(residuals)
    n = len(sorted_residuals)
    theoretical_quantiles = np.array([(i - 0.5) / n for i in range(1, n + 1)])
    from scipy import stats

    theoretical_quantiles = stats.norm.ppf(theoretical_quantiles)

    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_residuals,
            mode="markers",
            marker=dict(color="orange", size=6),
            name="Q-Q",
        ),
        row=2,
        col=2,
    )

    # Add diagonal line for Q-Q
    min_qq = min(min(theoretical_quantiles), min(sorted_residuals))
    max_qq = max(max(theoretical_quantiles), max(sorted_residuals))
    fig.add_trace(
        go.Scatter(
            x=[min_qq, max_qq],
            y=[min_qq, max_qq],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Normal",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text="Regression Analysis",
            x=0.5,
        ),
        showlegend=False,
        plot_bgcolor="white",
        height=800,
    )

    # Update axes labels
    fig.update_xaxes(title_text="Actual", row=1, col=1)
    fig.update_yaxes(title_text="Predicted", row=1, col=1)
    fig.update_xaxes(title_text="Predicted", row=1, col=2)
    fig.update_yaxes(title_text="Residual", row=1, col=2)
    fig.update_xaxes(title_text="Residual", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)

    return fig


def create_roc_curves_plot(metadata: Dict[str, Any]) -> go.Figure:
    """Create ROC curves for classification models."""
    if "roc_curves" not in metadata:
        return None

    roc_data = metadata["roc_curves"]

    fig = go.Figure()

    colors = px.colors.qualitative.Set1

    for i, (cls, data) in enumerate(roc_data.items()):
        fig.add_trace(
            go.Scatter(
                x=data["fpr"],
                y=data["tpr"],
                mode="lines",
                name=f"{cls} (AUC = {data['auc']:.3f})",
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            )
        )

    # Diagonal line (random classifier)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="gray", dash="dash"),
        )
    )

    fig.update_layout(
        title=dict(
            text="ROC Curves (One-vs-Rest)",
            x=0.5,
        ),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        height=500,
    )

    return fig


def create_model_header(metadata: Dict[str, Any]) -> html.Div:
    """Create the model header section."""
    model_type = metadata.get("model_type", "Unknown Model")
    training_time = metadata.get("training_time", 0)
    n_samples_train = metadata.get("n_samples_train", 0)
    n_samples_test = metadata.get("n_samples_test", 0)

    header = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4(
                                f"{model_type}",
                                className="mb-0",
                                style={"color": "#2196f3"},
                            ),
                            html.Small(
                                f"Trained in {training_time:.2f}s",
                                className="text-muted",
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Small(
                                        "Samples:",
                                        className="text-muted d-block",
                                    ),
                                    html.Strong(
                                        f"{n_samples_train:,} train / {n_samples_test:,} test",
                                        style={"fontSize": "14px"},
                                    ),
                                ],
                                className="text-end",
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="align-items-center",
            ),
            html.Hr(),
        ],
        className="mb-3",
    )

    return header


def create_model_details_panel(metadata: Dict[str, Any]) -> dbc.Card:
    """Create the collapsible model details panel."""
    # Parameters section
    params = metadata.get("parameters", {})
    params_rows = [html.Tr([html.Td(k), html.Td(str(v))]) for k, v in params.items()]

    params_table = (
        dbc.Table(
            [html.Tbody(params_rows)],
            bordered=True,
            size="sm",
            className="mb-0",
        )
        if params_rows
        else html.P("No parameters recorded", className="text-muted")
    )

    # Feature importance details
    feat_imp_content = html.Div()
    if "feature_importance" in metadata:
        feat_imp = metadata["feature_importance"]
        sorted_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)

        imp_rows = []
        for feat, imp in sorted_imp[:20]:  # Top 20
            imp_rows.append(html.Tr([html.Td(feat), html.Td(f"{imp:.4f}")]))

        feat_imp_content = dbc.Table(
            [
                html.Thead(html.Tr([html.Th("Feature"), html.Th("Importance")])),
                html.Tbody(imp_rows),
            ],
            bordered=True,
            size="sm",
        )

    panel = dbc.Card(
        [
            dbc.CardHeader(
                html.H6("Model Details", className="mb-0"),
            ),
            dbc.CardBody(
                [
                    html.H6("Parameters", className="mt-2"),
                    params_table,
                    html.Hr(),
                    html.H6("Feature Importance"),
                    feat_imp_content,
                ]
            ),
        ],
        className="shadow-sm",
    )

    return panel


def create_action_bar() -> html.Div:
    """Create the action bar with buttons."""
    return html.Div(
        [
            dbc.ButtonGroup(
                [
                    dbc.Button(
                        [html.I(className="fas fa-save me-2"), "Save Model"],
                        id="save-model-btn",
                        color="primary",
                        size="sm",
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-code me-2"), "Export Python"],
                        id="export-model-python-btn",
                        color="secondary",
                        size="sm",
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-database me-2"), "Export SQL"],
                        id="export-model-sql-btn",
                        color="info",
                        size="sm",
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-balance-scale me-2"), "Compare"],
                        id="compare-model-btn",
                        color="success",
                        size="sm",
                    ),
                ],
                className="me-2",
            ),
        ],
        className="mb-4",
    )
