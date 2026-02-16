"""Enhanced Statistics page for TreeLab."""

import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import html, dcc
from typing import Dict, List, Any, Optional


class StatisticsPage:
    """Enhanced statistics analysis page."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    def create_header(self) -> html.Div:
        """Create page header with dataset summary."""
        n_rows, n_cols = self.df.shape
        memory_usage = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        missing_pct = (self.df.isnull().sum().sum() / (n_rows * n_cols)) * 100
        duplicate_count = self.df.duplicated().sum()

        # Calculate data quality score
        completeness = (1 - self.df.isnull().sum().sum() / (n_rows * n_cols)) * 100
        uniqueness = (1 - duplicate_count / n_rows) * 100 if n_rows > 0 else 100
        quality_score = (completeness + uniqueness) / 2

        score_color = (
            "success"
            if quality_score >= 80
            else "warning"
            if quality_score >= 60
            else "danger"
        )

        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("Dataset Statistics", className="mb-0"),
                                html.Small(
                                    f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
                                    className="text-muted",
                                ),
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H6(
                                                    "Data Quality Score",
                                                    className="mb-1 text-center",
                                                ),
                                                html.H3(
                                                    f"{quality_score:.0f}/100",
                                                    className=f"text-{score_color} text-center mb-0",
                                                ),
                                            ],
                                            className="py-2",
                                        )
                                    ],
                                    className="shadow-sm",
                                )
                            ],
                            width=4,
                        ),
                    ],
                    className="mb-4 align-items-center",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            self._create_info_card("Rows", f"{n_rows:,}", "primary"),
                            width=2,
                        ),
                        dbc.Col(
                            self._create_info_card("Columns", f"{n_cols}", "info"),
                            width=2,
                        ),
                        dbc.Col(
                            self._create_info_card(
                                "Memory", f"{memory_usage:.1f} MB", "secondary"
                            ),
                            width=2,
                        ),
                        dbc.Col(
                            self._create_info_card(
                                "Missing",
                                f"{missing_pct:.1f}%",
                                "warning" if missing_pct > 5 else "success",
                            ),
                            width=2,
                        ),
                        dbc.Col(
                            self._create_info_card(
                                "Duplicates",
                                f"{duplicate_count:,}",
                                "danger" if duplicate_count > 0 else "success",
                            ),
                            width=2,
                        ),
                        dbc.Col(
                            self._create_info_card(
                                "Numeric", f"{len(self.numeric_cols)}", "info"
                            ),
                            width=2,
                        ),
                    ],
                    className="g-2 mb-4",
                ),
            ]
        )

    def _create_info_card(self, label: str, value: str, color: str) -> dbc.Card:
        """Create a small info card."""
        return dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H6(
                            label, className="card-subtitle text-muted mb-1 text-center"
                        ),
                        html.H5(
                            value, className=f"card-title text-{color} text-center mb-0"
                        ),
                    ],
                    className="py-2",
                )
            ],
            className="h-100 shadow-sm",
        )

    def create_numeric_stats_table(
        self, selected_cols: Optional[List[str]] = None
    ) -> html.Div:
        """Create comprehensive statistics table for numeric columns."""
        cols = selected_cols if selected_cols else self.numeric_cols

        if not cols:
            return html.Div("No numeric columns available", className="text-muted")

        stats_data = []
        for col in cols:
            series = self.df[col].dropna()
            if len(series) == 0:
                continue

            # Basic stats
            stat_dict = {
                "Feature": col,
                "Count": len(series),
                "Mean": series.mean(),
                "Std": series.std(),
                "Min": series.min(),
                "25%": series.quantile(0.25),
                "50%": series.median(),
                "75%": series.quantile(0.75),
                "Max": series.max(),
                "Missing": self.df[col].isnull().sum(),
                "Missing %": (self.df[col].isnull().sum() / len(self.df)) * 100,
                "Skewness": stats.skew(series),
                "Kurtosis": stats.kurtosis(series),
                "Unique": series.nunique(),
            }

            # Normality test (Shapiro-Wilk on sample if large)
            if len(series) <= 5000:
                _, p_value = stats.shapiro(series)
                stat_dict["Normality (p)"] = p_value
            else:
                # Use Anderson-Darling for larger samples
                result = stats.anderson(series, dist="norm")
                stat_dict["Normality (A²)"] = result.statistic

            stats_data.append(stat_dict)

        stats_df = pd.DataFrame(stats_data)

        # Format numeric columns
        for col in stats_df.columns:
            if col not in ["Feature", "Count", "Missing", "Unique"]:
                stats_df[col] = stats_df[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else ""
                )

        return html.Div(
            [
                html.H6(f"Numeric Features ({len(stats_df)} total)", className="mb-3"),
                dbc.Table.from_dataframe(
                    stats_df,
                    striped=True,
                    bordered=True,
                    hover=True,
                    size="sm",
                    responsive=True,
                ),
            ]
        )

    def create_categorical_stats_table(
        self, selected_cols: Optional[List[str]] = None
    ) -> html.Div:
        """Create statistics table for categorical columns."""
        cols = selected_cols if selected_cols else self.categorical_cols

        if not cols:
            return html.Div("No categorical columns available", className="text-muted")

        stats_data = []
        for col in cols:
            series = self.df[col].dropna()
            if len(series) == 0:
                continue

            value_counts = series.value_counts()
            mode_val = value_counts.index[0] if len(value_counts) > 0 else "N/A"
            mode_freq = value_counts.iloc[0] if len(value_counts) > 0 else 0
            mode_pct = (mode_freq / len(series)) * 100

            # Calculate entropy
            probs = value_counts / len(series)
            entropy = -sum(probs * np.log2(probs))
            max_entropy = np.log2(len(value_counts))
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

            stats_data.append(
                {
                    "Feature": col,
                    "Unique": series.nunique(),
                    "Most Common": str(mode_val)[:30],  # Truncate long strings
                    "Frequency": mode_freq,
                    "% of Total": f"{mode_pct:.1f}%",
                    "Missing": self.df[col].isnull().sum(),
                    "Missing %": f"{(self.df[col].isnull().sum() / len(self.df)) * 100:.1f}%",
                    "Entropy": f"{entropy:.2f}",
                    "Norm. Entropy": f"{norm_entropy:.2f}",
                }
            )

        stats_df = pd.DataFrame(stats_data)

        return html.Div(
            [
                html.H6(
                    f"Categorical Features ({len(stats_df)} total)", className="mb-3"
                ),
                dbc.Table.from_dataframe(
                    stats_df,
                    striped=True,
                    bordered=True,
                    hover=True,
                    size="sm",
                    responsive=True,
                ),
            ]
        )

    def create_missing_values_analysis(self) -> html.Div:
        """Create missing values analysis with heatmap."""
        missing = self.df.isnull()
        missing_by_col = missing.sum().sort_values(ascending=False)
        missing_by_col = missing_by_col[missing_by_col > 0]

        if len(missing_by_col) == 0:
            return dbc.Alert("✓ No missing values in dataset!", color="success")

        # Missing values heatmap (sample rows if too large)
        sample_size = min(500, len(self.df))
        sample_df = (
            self.df.sample(sample_size) if len(self.df) > sample_size else self.df
        )
        missing_sample = sample_df.isnull()

        fig = px.imshow(
            missing_sample.T,
            color_continuous_scale=[(0, "white"), (1, "#ff6b6b")],
            aspect="auto",
            title=f"Missing Values Pattern (showing {sample_size} random rows)",
            labels=dict(x="Row Index", y="Feature", color="Missing"),
        )
        fig.update_layout(
            height=100 + len(missing_by_col) * 20,
            plot_bgcolor="white",
        )

        # Missing stats table
        missing_stats = pd.DataFrame(
            {
                "Feature": missing_by_col.index,
                "Missing Count": missing_by_col.values,
                "Missing %": (missing_by_col.values / len(self.df) * 100).round(2),
            }
        )

        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(figure=fig, config={"displayModeBar": True}),
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            [
                                html.H6("Missing Values Summary"),
                                dbc.Table.from_dataframe(
                                    missing_stats,
                                    striped=True,
                                    bordered=True,
                                    hover=True,
                                    size="sm",
                                ),
                            ],
                            width=4,
                        ),
                    ]
                ),
            ]
        )

    def create_duplicate_analysis(self) -> html.Div:
        """Analyze and display duplicate rows."""
        duplicates = self.df.duplicated()
        n_duplicates = duplicates.sum()

        if n_duplicates == 0:
            return dbc.Alert("✓ No duplicate rows found!", color="success")

        # Find duplicate groups
        all_duplicates = self.df[duplicates | self.df.duplicated(keep=False)]

        return html.Div(
            [
                dbc.Alert(
                    f"⚠️ Found {n_duplicates} duplicate rows ({n_duplicates / len(self.df) * 100:.1f}% of dataset)",
                    color="warning",
                ),
                html.H6("Sample of Duplicate Rows"),
                dbc.Table.from_dataframe(
                    all_duplicates.head(20),
                    striped=True,
                    bordered=True,
                    hover=True,
                    size="sm",
                    responsive=True,
                ),
            ]
        )

    def create_outlier_summary(self) -> html.Div:
        """Create outlier detection summary using IQR method."""
        outlier_data = []

        for col in self.numeric_cols[:10]:  # Limit to first 10 for performance
            series = self.df[col].dropna()
            if len(series) < 10:
                continue

            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = series[(series < lower_bound) | (series > upper_bound)]

            outlier_data.append(
                {
                    "Feature": col,
                    "Outliers": len(outliers),
                    "% Outliers": f"{len(outliers) / len(series) * 100:.1f}%",
                    "Lower Bound": f"{lower_bound:.2f}",
                    "Upper Bound": f"{upper_bound:.2f}",
                    "Min": f"{series.min():.2f}",
                    "Max": f"{series.max():.2f}",
                }
            )

        if not outlier_data:
            return html.Div(
                "No outliers detected in numeric columns", className="text-muted"
            )

        outlier_df = pd.DataFrame(outlier_data)

        return html.Div(
            [
                html.H6("Outlier Detection Summary (IQR Method)"),
                dbc.Table.from_dataframe(
                    outlier_df,
                    striped=True,
                    bordered=True,
                    hover=True,
                    size="sm",
                ),
                html.Small(
                    "Using 1.5 × IQR rule. Showing first 10 numeric columns only.",
                    className="text-muted",
                ),
            ]
        )

    def render(self) -> html.Div:
        """Render the complete statistics page."""
        return html.Div(
            [
                self.create_header(),
                html.Hr(),
                # Tabs for different sections
                dbc.Tabs(
                    [
                        dbc.Tab(
                            [
                                html.Br(),
                                self.create_numeric_stats_table(),
                            ],
                            label="Numeric Features",
                            tab_id="stats-numeric",
                        ),
                        dbc.Tab(
                            [
                                html.Br(),
                                self.create_categorical_stats_table(),
                            ],
                            label="Categorical Features",
                            tab_id="stats-categorical",
                        ),
                        dbc.Tab(
                            [
                                html.Br(),
                                self.create_missing_values_analysis(),
                            ],
                            label="Missing Values",
                            tab_id="stats-missing",
                        ),
                        dbc.Tab(
                            [
                                html.Br(),
                                self.create_duplicate_analysis(),
                            ],
                            label="Duplicates",
                            tab_id="stats-duplicates",
                        ),
                        dbc.Tab(
                            [
                                html.Br(),
                                self.create_outlier_summary(),
                            ],
                            label="Outliers",
                            tab_id="stats-outliers",
                        ),
                    ],
                    id="stats-tabs",
                    active_tab="stats-numeric",
                ),
            ],
            style={"padding": "20px"},
        )
