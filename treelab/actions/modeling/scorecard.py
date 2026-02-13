"""Scorecard actions for TreeLab."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, cast, Hashable
from treelab.actions.base import Action, Parameter


class BinningScorecardAction(Action):
    """Create a binning scorecard for numeric features."""

    name = "BinningScorecard"
    description = "Bin a numeric feature and compute WOE/IV scorecard"
    mode = "modeling"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="feature_columns",
                label="Feature Columns",
                type="columns",
                required=True,
                description="Numeric feature(s) to bin",
            ),
            Parameter(
                name="target_column",
                label="Target Column",
                type="column",
                required=True,
                description="Binary target column for WOE/IV",
            ),
            Parameter(
                name="bins",
                label="Number of Bins",
                type="numeric",
                required=False,
                default=5,
                description="Number of bins to create",
            ),
            Parameter(
                name="binning_method",
                label="Binning Method",
                type="select",
                required=False,
                default="quantile",
                options=["quantile", "uniform"],
                description="Quantile (equal frequency) or uniform (equal width)",
            ),
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        feature_columns = params.get("feature_columns", [])
        if not feature_columns:
            return False, "Feature columns are required"

        if not isinstance(feature_columns, list):
            feature_columns = [feature_columns]

        target_col = params.get("target_column")
        if not target_col:
            return False, "Target column is required"

        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            return False, f"Feature columns not found: {missing_features}"

        if target_col not in df.columns:
            return False, f"Target column '{target_col}' not found"

        if df[target_col].nunique() != 2:
            return False, "Target column must be binary for scorecard"

        non_numeric = [
            col for col in feature_columns if not pd.api.types.is_numeric_dtype(df[col])
        ]
        if non_numeric:
            return False, f"Feature columns must be numeric: {non_numeric}"

        bins = params.get("bins", 5)
        if bins < 2:
            return False, "Number of bins must be >= 2"

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        feature_columns = params.get("feature_columns", [])
        if not isinstance(feature_columns, list):
            feature_columns = [feature_columns]

        target_col = params.get("target_column")
        bins = int(params.get("bins", 5))
        method = params.get("binning_method", "quantile")
        target_col = cast(Hashable, target_col)
        target = df[target_col]

        scorecards = []
        total_iv = 0.0

        for feature_column in feature_columns:
            feature = df[feature_column]

            if method == "quantile":
                binned = pd.qcut(feature, q=bins, duplicates="drop")
            else:
                binned = pd.cut(feature, bins=bins)

            bin_df = pd.DataFrame({"bin": binned, "target": target})
            bin_df["bin"] = bin_df["bin"].astype("object").fillna("Missing")

            grouped = bin_df.groupby("bin")
            total = grouped.size()
            events = grouped["target"].sum()
            non_events = total - events

            event_total = events.sum()
            non_event_total = non_events.sum()

            eps = 1e-6
            event_rate = (events / (event_total + eps)).replace(0, eps)
            non_event_rate = (non_events / (non_event_total + eps)).replace(0, eps)

            woe = np.log(event_rate / non_event_rate)
            iv = (event_rate - non_event_rate) * woe

            scorecard = pd.DataFrame(
                {
                    "Bin": total.index.astype(str),
                    "Count": total.values,
                    "Events": events.values,
                    "EventRate": (events / total).fillna(0).values,
                    "WOE": woe.values,
                    "IV": iv.values,
                }
            ).sort_values("Bin")

            scorecards.append(
                {
                    "feature": feature_column,
                    "total_iv": float(iv.sum()),
                    "table": scorecard.to_dict("records"),
                }
            )
            total_iv += float(iv.sum())

        metadata = {
            "model_type": "BinningScorecard",
            "task": "scorecard",
            "target_column": target_col,
            "scorecards": scorecards,
            "scorecard_total_iv": float(total_iv),
        }

        return {
            "df": df,
            "train_df": train_df,
            "test_df": test_df,
            "model": {"type": "scorecard"},
            "metadata": metadata,
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        return []

    def to_python_code(self, params: Dict[str, Any]) -> str:
        feature_columns = params.get("feature_columns", [])
        target_column = params.get("target_column")
        bins = params.get("bins", 5)
        method = params.get("binning_method", "quantile")

        code = "# Binning scorecard (WOE/IV)\n"
        code += "import pandas as pd\n"
        code += "import numpy as np\n"
        code += f"feature_columns = {feature_columns}\n"
        code += f"target_column = '{target_column}'\n"
        code += "for col in feature_columns:\n"
        if method == "quantile":
            code += f"    binned = pd.qcut(df[col], q={bins}, duplicates='drop')\n"
        else:
            code += f"    binned = pd.cut(df[col], bins={bins})\n"
        code += "    tmp = pd.DataFrame({'bin': binned, 'target': df[target_column]})\n"
        code += "    tmp['bin'] = tmp['bin'].astype('object').fillna('Missing')\n"
        code += "    grouped = tmp.groupby('bin')\n"
        code += "    total = grouped.size()\n"
        code += "    events = grouped['target'].sum()\n"
        code += "    non_events = total - events\n"
        code += "    event_rate = (events / events.sum()).replace(0, 1e-6)\n"
        code += (
            "    non_event_rate = (non_events / non_events.sum()).replace(0, 1e-6)\n"
        )
        code += "    woe = np.log(event_rate / non_event_rate)\n"
        code += "    iv = (event_rate - non_event_rate) * woe\n"
        code += "    print(col, 'IV:', iv.sum())"

        return code
