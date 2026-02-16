"""Enhanced Distributions page for TreeLab."""

import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import html, dcc
from typing import Dict, List, Any, Optional


class DistributionsPage:
    """Enhanced distributions analysis page."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.selected_col = self.numeric_cols[0] if self.numeric_cols else None

    def create_header(self) -> html.Div:
        """Create page header."""
        return html.Div(
            [
                html.H4("Distribution Analysis"),
                html.P(
                    "Explore feature distributions with statistical tests and transformations",
                    className="text-muted",
                ),
                html.Hr(),
            ],
            className="mb-4",
        )

    def create_feature_selector(self) -> html.Div:
        """Create feature selection dropdown."""
        if not self.numeric_cols:
            return html.Div("No numeric columns available", className="text-muted")

        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Select Feature:", className="fw-bold"),
                                dcc.Dropdown(
                                    id="dist-feature-selector",
                                    options=[
                                        {"label": col, "value": col}
                                        for col in self.numeric_cols
                                    ],
                                    value=self.selected_col,
                                    clearable=False,
                                    searchable=True,
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Chart Type:", className="fw-bold"),
                                dcc.Dropdown(
                                    id="dist-chart-type",
                                    options=[
                                        {
                                            "label": "📊 Histogram + KDE",
                                            "value": "histogram",
                                        },
                                        {"label": "📦 Box Plot", "value": "box"},
                                        {
                                            "label": "📈 Q-Q Plot (Normality)",
                                            "value": "qq",
                                        },
                                        {
                                            "label": "📉 Cumulative Distribution",
                                            "value": "cdf",
                                        },
                                        {
                                            "label": "🎨 Histogram by Quartiles",
                                            "value": "histogram_quartile",
                                        },
                                        {
                                            "label": "🎨 Histogram by Deciles",
                                            "value": "histogram_decile",
                                        },
                                    ],
                                    value="histogram",
                                    clearable=False,
                                    className="mt-2",
                                ),
                            ],
                            width=8,
                        ),
                    ],
                    className="mb-4",
                ),
            ]
        )

    def create_distribution_plot(
        self, column: str, chart_type: str = "histogram"
    ) -> go.Figure:
        """Create distribution plot for selected column."""
        if column not in self.df.columns:
            return go.Figure()

        series = self.df[column].dropna()

        if chart_type == "histogram":
            fig = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.1,
                subplot_titles=("Distribution", "Box Plot"),
            )

            # Histogram with KDE
            fig.add_trace(
                go.Histogram(
                    x=series,
                    nbinsx=50,
                    name="Histogram",
                    opacity=0.7,
                    marker_color="#2196f3",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            # Add mean and median lines
            mean_val = series.mean()
            median_val = series.median()

            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_val:.2f}",
                row=1,
                col=1,
            )
            fig.add_vline(
                x=median_val,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Median: {median_val:.2f}",
                row=1,
                col=1,
            )

            # Box plot below
            fig.add_trace(
                go.Box(
                    x=series,
                    name="",
                    boxpoints="outliers",
                    marker_color="#2196f3",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            fig.update_layout(
                title=f"Distribution of {column}",
                height=600,
                showlegend=False,
                plot_bgcolor="white",
            )

        elif chart_type == "box":
            fig = go.Figure()
            fig.add_trace(
                go.Box(
                    y=series,
                    name=column,
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                    marker_color="#2196f3",
                )
            )
            fig.update_layout(
                title=f"Box Plot of {column}",
                height=500,
                plot_bgcolor="white",
                yaxis_title=column,
            )

        elif chart_type == "qq":
            # Q-Q plot
            from scipy.stats import probplot

            # probplot returns (osm, osr), (slope, intercept, r)
            # osm = theoretical quantiles, osr = sample quantiles
            (theoretical, sample), (slope, intercept, r) = probplot(series, dist="norm")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=theoretical,
                    y=sample,
                    mode="markers",
                    marker=dict(color="#2196f3", size=6),
                    name="Q-Q",
                )
            )

            # Add reference line
            line_x = np.array([theoretical.min(), theoretical.max()])
            line_y = slope * line_x + intercept

            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name="Reference",
                )
            )

            fig.update_layout(
                title=f"Q-Q Plot: {column} vs Normal Distribution (R²={r**2:.3f})",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                height=500,
                plot_bgcolor="white",
                showlegend=True,
            )

        elif chart_type == "cdf":
            # Empirical CDF
            sorted_data = np.sort(series)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=sorted_data,
                    y=cdf,
                    mode="lines",
                    line=dict(color="#2196f3", width=2),
                    name="Empirical CDF",
                )
            )

            fig.update_layout(
                title=f"Cumulative Distribution Function: {column}",
                xaxis_title=column,
                yaxis_title="Cumulative Probability",
                height=500,
                plot_bgcolor="white",
            )

        elif chart_type == "histogram_quartile":
            # Histogram colored by quartiles
            quartiles = series.quantile([0.25, 0.5, 0.75])
            q1, q2, q3 = quartiles[0.25], quartiles[0.5], quartiles[0.75]

            # Create color array based on quartiles
            colors = []
            for val in series:
                if val <= q1:
                    colors.append("#2196f3")  # Blue - Q1
                elif val <= q2:
                    colors.append("#4caf50")  # Green - Q2
                elif val <= q3:
                    colors.append("#ff9800")  # Orange - Q3
                else:
                    colors.append("#f44336")  # Red - Q4

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=series,
                    nbinsx=30,
                    marker=dict(
                        color=colors,
                        line=dict(width=1, color="white"),
                    ),
                    showlegend=False,
                )
            )

            # Add vertical lines for quartiles
            fig.add_vline(
                x=q1, line_dash="dash", line_color="#2196f3", annotation_text="Q1"
            )
            fig.add_vline(
                x=q2,
                line_dash="dash",
                line_color="#4caf50",
                annotation_text="Q2 (Median)",
            )
            fig.add_vline(
                x=q3, line_dash="dash", line_color="#ff9800", annotation_text="Q3"
            )

            fig.update_layout(
                title=f"Distribution of {column} by Quartiles",
                xaxis_title=column,
                yaxis_title="Count",
                height=500,
                plot_bgcolor="white",
            )

            # Add legend annotation
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text="Blue=Q1 | Green=Q2 | Orange=Q3 | Red=Q4",
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10),
            )

        elif chart_type == "histogram_decile":
            # Histogram colored by deciles
            deciles = series.quantile(np.arange(0.1, 1.0, 0.1))
            decile_colors = [
                "#2196f3",
                "#42a5f5",
                "#64b5f6",
                "#90caf9",
                "#bbdefb",
                "#fff3e0",
                "#ffe0b2",
                "#ffcc80",
                "#ffb74d",
                "#ff9800",
            ]

            # Create color array based on deciles
            colors = []
            for val in series:
                for i, (decile_val, color) in enumerate(zip(deciles, decile_colors)):
                    if val <= decile_val:
                        colors.append(color)
                        break
                else:
                    colors.append(decile_colors[-1])

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=series,
                    nbinsx=30,
                    marker=dict(
                        color=colors,
                        line=dict(width=1, color="white"),
                    ),
                    showlegend=False,
                )
            )

            # Add vertical lines for deciles
            for i, (decile_val, color) in enumerate(zip(deciles, decile_colors)):
                if i % 2 == 0:  # Only show every other decile line to avoid clutter
                    fig.add_vline(
                        x=decile_val,
                        line_dash="dot",
                        line_color=color,
                        line_width=1,
                    )

            fig.update_layout(
                title=f"Distribution of {column} by Deciles",
                xaxis_title=column,
                yaxis_title="Count",
                height=500,
                plot_bgcolor="white",
            )

            # Add legend annotation
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text="Colors: D1 (Blue) → D10 (Orange)",
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10),
            )

        else:
            # Default empty figure
            fig = go.Figure()
            fig.update_layout(
                title="Select a chart type",
                height=500,
                plot_bgcolor="white",
            )

        return fig

    def create_statistics_panel(self, column: str) -> html.Div:
        """Create statistics panel for selected feature."""
        if column not in self.df.columns:
            return html.Div()

        series = self.df[column].dropna()

        if len(series) == 0:
            return html.Div("No data available", className="text-muted")

        # Basic statistics
        stats_dict = {
            "Count": len(series),
            "Mean": f"{series.mean():.4f}",
            "Median": f"{series.median():.4f}",
            "Mode": f"{series.mode().iloc[0]:.4f}" if len(series.mode()) > 0 else "N/A",
            "Std Dev": f"{series.std():.4f}",
            "Variance": f"{series.var():.4f}",
            "Min": f"{series.min():.4f}",
            "Max": f"{series.max():.4f}",
            "Range": f"{series.max() - series.min():.4f}",
            "IQR": f"{series.quantile(0.75) - series.quantile(0.25):.4f}",
        }

        # Shape statistics
        shape_dict = {
            "Skewness": f"{stats.skew(series):.4f}",
            "Kurtosis": f"{stats.kurtosis(series):.4f}",
            "CV (%)": f"{(series.std() / series.mean()) * 100:.2f}%"
            if series.mean() != 0
            else "N/A",
        }

        # Percentiles
        percentiles = {
            "5th": f"{series.quantile(0.05):.4f}",
            "25th": f"{series.quantile(0.25):.4f}",
            "50th": f"{series.quantile(0.50):.4f}",
            "75th": f"{series.quantile(0.75):.4f}",
            "95th": f"{series.quantile(0.95):.4f}",
        }

        # Normality test
        if len(series) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(series)
            normality_result = f"Shapiro-Wilk: p={shapiro_p:.4f}"
            is_normal = shapiro_p > 0.05
        else:
            # Use Kolmogorov-Smirnov for larger samples
            ks_stat, ks_p = stats.kstest(
                series, "norm", args=(series.mean(), series.std())
            )
            normality_result = f"K-S Test: p={ks_p:.4f}"
            is_normal = ks_p > 0.05

        normality_alert = dbc.Alert(
            f"{normality_result} - {'Normal' if is_normal else 'Not Normal'}",
            color="success" if is_normal else "warning",
            className="mb-3",
        )

        return html.Div(
            [
                html.H6(f"Statistics: {column}", className="mb-3"),
                normality_alert,
                dbc.Tabs(
                    [
                        dbc.Tab(
                            [
                                html.Br(),
                                self._create_stats_table(stats_dict),
                            ],
                            label="Basic",
                        ),
                        dbc.Tab(
                            [
                                html.Br(),
                                self._create_stats_table(shape_dict),
                            ],
                            label="Shape",
                        ),
                        dbc.Tab(
                            [
                                html.Br(),
                                self._create_stats_table(percentiles),
                            ],
                            label="Percentiles",
                        ),
                    ]
                ),
            ],
            className="p-3 border rounded",
        )

    def _create_stats_table(self, data: Dict[str, str]) -> dbc.Table:
        """Create a simple stats table."""
        rows = [
            html.Tr([html.Td(k, className="fw-bold"), html.Td(v)])
            for k, v in data.items()
        ]
        return dbc.Table([html.Tbody(rows)], bordered=True, size="sm", className="mb-0")

    def create_distribution_gallery(self) -> html.Div:
        """Create a gallery of mini distributions for all numeric columns."""
        if not self.numeric_cols:
            return html.Div("No numeric columns available", className="text-muted")

        # Limit to first 12 columns for performance
        display_cols = self.numeric_cols[:12]

        n_cols = 3
        n_rows = (len(display_cols) + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=display_cols,
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
        )

        for idx, col in enumerate(display_cols):
            row = idx // n_cols + 1
            col_idx = idx % n_cols + 1

            series = self.df[col].dropna()
            if len(series) == 0:
                continue

            fig.add_trace(
                go.Histogram(
                    x=series,
                    nbinsx=20,
                    showlegend=False,
                    marker_color="#2196f3",
                    opacity=0.7,
                ),
                row=row,
                col=col_idx,
            )

            # Add mean line
            fig.add_vline(
                x=series.mean(),
                line_dash="dash",
                line_color="red",
                line_width=1,
                row=row,
                col=col_idx,
            )

        fig.update_layout(
            title="Distribution Gallery (First 12 Numeric Features)",
            height=n_rows * 250,
            showlegend=False,
            plot_bgcolor="white",
        )

        return html.Div(
            [
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
                html.Small(
                    "Showing first 12 numeric columns. Red dashed lines indicate mean values.",
                    className="text-muted",
                ),
            ]
        )

    def render(self) -> html.Div:
        """Render the complete distributions page."""
        if not self.numeric_cols:
            return html.Div(
                [
                    self.create_header(),
                    dbc.Alert(
                        "No numeric columns available for distribution analysis",
                        color="info",
                    ),
                ]
            )

        return html.Div(
            [
                self.create_header(),
                self.create_feature_selector(),
                dbc.Tabs(
                    [
                        dbc.Tab(
                            [
                                html.Br(),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dcc.Graph(
                                                    id="dist-main-plot",
                                                    config={"displayModeBar": True},
                                                ),
                                            ],
                                            width=8,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(id="dist-stats-panel"),
                                            ],
                                            width=4,
                                        ),
                                    ]
                                ),
                            ],
                            label="Single Feature",
                            tab_id="dist-single",
                        ),
                        dbc.Tab(
                            [
                                html.Br(),
                                self.create_distribution_gallery(),
                            ],
                            label="Gallery",
                            tab_id="dist-gallery",
                        ),
                    ],
                    id="dist-tabs",
                    active_tab="dist-single",
                ),
            ],
            style={"padding": "20px"},
        )
