# TreeLab Correlations Page Design Specification

## Overview

A comprehensive correlations analysis page for exploring relationships between features in the dataset. Supports various correlation methods, visualizations, and network analysis.

---

## Page Structure

```
CorrelationsPage
├── HeaderSection
│   ├── Correlation Method Selector
│   ├── Target Variable Selector (optional)
│   └── Significance Level Toggle
├── MainVisualization
│   ├── CorrelationMatrix (heatmap)
│   └── ControlsPanel
├── SecondaryVisualizations
│   ├── NetworkGraph
│   ├── ScatterMatrix
│   └── CorrelationRanking
├── AnalysisPanel
│   ├── HighCorrelationsList
│   ├── RedundancyAnalysis
│   └── FeatureClustering
└── ActionBar
    ├── Export Matrix
    ├── Apply Decorrelation
    └── Generate Report
```

---

## Header Section

### Correlation Configuration
```
┌─────────────────────────────────────────────────────────────────┐
│ Correlation Analysis                                             │
│                                                                  │
│ Method: [Pearson ▼]        Target: [None ▼]    α = [0.05 ▼]    │
│         • Pearson (linear)       • Feature 1                     │
│         • Spearman (rank)        • Feature 2                     │
│         • Kendall (concordance)  • Feature 3                     │
│         • Point-Biserial         • ...                          │
│         • Cramer's V (categorical)                               │
│                                                                  │
│ Features: [All Numeric ▼]    Exclude: [None ▼]                   │
│           • All Numeric            • Constant features           │
│           • High cardinality       • Near-zero variance          │
│           • Custom selection       • High missing %              │
└─────────────────────────────────────────────────────────────────┘
```

### Quick Stats
```
┌─ Summary Statistics ─────────────────────────┐
│ Total Pairs: N                              │
│ Significant (p<0.05): N (X%)                │
│ |r| > 0.7 (strong): N                       │
│ |r| > 0.9 (very strong): N                  │
│ Redundant pairs: N                          │
└─────────────────────────────────────────────┘
```

---

## Primary Visualization: Correlation Matrix Heatmap

### Interactive Heatmap
```
┌─ Correlation Matrix ─────────────────────────────────────────────┐
│                                                                  │
│        feat1  feat2  feat3  feat4  feat5  feat6  feat7  feat8   │
│ feat1   1.00   0.45   0.23  -0.67   0.12   0.89  -0.34   0.01   │
│ feat2   0.45   1.00   0.11  -0.23   0.67   0.45  -0.12   0.56   │
│ feat3   0.23   0.11   1.00   0.05   0.89   0.23  -0.67   0.34   │
│ feat4  -0.67  -0.23   0.05   1.00  -0.45   0.12   0.78  -0.23   │
│ feat5   0.12   0.67   0.89  -0.45   1.00   0.34  -0.56   0.12   │
│ feat6   0.89   0.45   0.23   0.12   0.34   1.00  -0.23   0.67   │
│ feat7  -0.34  -0.12  -0.67   0.78  -0.56  -0.23   1.00  -0.45   │
│ feat8   0.01   0.56   0.34  -0.23   0.12   0.67  -0.45   1.00   │
│                                                                  │
│ Color Scale: [-1] ─────────── [0] ─────────── [1]               │
│              🔵🔵🔵            ⚪            🔴🔴🔴             │
│                                                                  │
│ [Cluster] [Reorder] [Significance Mask] [Values] [Export]       │
└─────────────────────────────────────────────────────────────────┘
```

### Heatmap Controls
```
┌─ Visualization Controls ─────────────────────────────────────────┐
│                                                                  │
│ Color Scheme: [RdBu ▼]    Annotations: [☑ Values] [☑ Stars]     │
│              • RdBu (diverging)     Stars: ★★★ p<0.001         │
│              • Viridis (sequential)        ★★  p<0.01          │
│              • Coolwarm                    ★   p<0.05          │
│                                                                  │
│ Mask: [☑ Upper triangle] [☑ Insignificant] [☑ Diagonal]        │
│                                                                  │
│ Threshold: |r| > [0.0 ▼]    Significance: p < [0.05 ▼]         │
│                                                                  │
│ Size: [Responsive ▼]      Zoom: [100%] [+] [-] [Fit]           │
└─────────────────────────────────────────────────────────────────┘
```

### Cell Interaction
**Hover:**
```
┌─ Tooltip ──────────────────────────┐
│ Feature Pair:                       │
│   feat6 × feat1                     │
│                                     │
│ Correlation:                        │
│   Pearson r = 0.89 ★★★             │
│   p-value = 1.2e-15                 │
│   95% CI: [0.85, 0.92]             │
│                                     │
│ Interpretation:                     │
│   Very strong positive              │
│   linear relationship               │
│                                     │
│ Sample Size: N=1,000                │
│ Missing Pairs: 12                   │
└─────────────────────────────────────┘
```

**Click:**
- Opens detailed scatter plot of the pair
- Shows regression line and statistics
- Allows outlier inspection

---

## Secondary Visualizations

### 1. Correlation Network Graph

```
┌─ Correlation Network ────────────────────────────────────────────┐
│                                                                  │
│              ┌─────────┐                                         │
│              │ feat1   │                                         │
│              └───┬─────┘                                         │
│                  │ r=0.89                                         │
│                  │                                                │
│    ┌─────────┐   │   ┌─────────┐                                 │
│    │ feat3   │◄──┴──►│ feat6   │                                 │
│    │         │       │         │                                 │
│    └────┬────┘       └────┬────┘                                 │
│         │                  │                                     │
│    r=0. │67               │ r=0.45                               │
│         │                  │                                     │
│    ┌────▼────┐       ┌────▼────┐                                 │
│    │ feat2   │       │ feat5   │                                 │
│    └─────────┘       └─────────┘                                 │
│                                                                  │
│ Legend:                                                          │
│ ─── |r| > 0.9    ── |r| > 0.7    ·· |r| > 0.5                   │
│                                                                  │
│ Layout: [Force ▼]    Filter: [|r| > 0.5 ▼]    Labels: [☑]       │
└─────────────────────────────────────────────────────────────────┘
```

**Controls:**
- Node size: By degree / By variance / Equal
- Edge thickness: By |r| value
- Cluster by: Community detection / Feature type
- Physics: Enable/disable force layout

### 2. Scatter Plot Matrix (SPLOM)

```
┌─ Pairwise Relationships ────────────────────────────────────────┐
│                                                                  │
│         feat1      feat2      feat3      feat4                  │
│      ┌─────────┬─────────┬─────────┬─────────┐                  │
│ feat1│ [hist]  │ [scat]  │ [scat]  │ [scat]  │                  │
│      ├─────────┼─────────┼─────────┼─────────┤                  │
│ feat2│ [scat]  │ [hist]  │ [scat]  │ [scat]  │                  │
│      ├─────────┼─────────┼─────────┼─────────┤                  │
│ feat3│ [scat]  │ [scat]  │ [hist]  │ [scat]  │                  │
│      ├─────────┼─────────┼─────────┼─────────┤                  │
│ feat4│ [scat]  │ [scat]  │ [scat]  │ [hist]  │                  │
│      └─────────┴─────────┴─────────┴─────────┘                  │
│                                                                  │
│ Upper: [Scatter ▼]    Diagonal: [Histogram ▼]                   │
│ Lower: [KDE ▼]        Color by: [None ▼]                        │
└─────────────────────────────────────────────────────────────────┘
```

**Features:**
- Click any subplot to expand
- Brushing on one applies to all
- Show correlation coefficient on each
- Regression line toggle

### 3. Correlation Bar Chart

```
┌─ Correlation Strength Ranking ──────────────────────────────────┐
│                                                                  │
│ Target: feat1                                                    │
│                                                                  │
│ Most Positively Correlated:                                      │
│ feat6    ████████████████████████████  r=0.89 ★★★              │
│ feat3    ██████████████████            r=0.67 ★★               │
│ feat8    ███████████████               r=0.56 ★★               │
│                                                                  │
│ Most Negatively Correlated:                                      │
│ feat4    ████████████████████████████  r=-0.67 ★★              │
│ feat7    ██████████████████            r=-0.34 ★               │
│ feat2    ████████████                  r=-0.23                  │
│                                                                  │
│ [Sort by: |r| ▼] [Show All] [Export List]                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Analysis Panel

### High Correlations List
```
┌─ Significant Correlations (|r| > 0.7) ───────────────────────────┐
│                                                                  │
│ Rank │ Feature 1 │ Feature 2 │    r    │ p-value  │ Action      │
│──────┼───────────┼───────────┼─────────┼──────────┼─────────────│
│   1  │ feat6     │ feat1     │  0.89   │ < 0.001  │ [View] [🗑] │
│   2  │ feat5     │ feat3     │  0.85   │ < 0.001  │ [View] [🗑] │
│   3  │ feat4     │ feat7     │ -0.78   │ < 0.001  │ [View] [🗑] │
│   4  │ feat2     │ feat5     │  0.72   │  0.002   │ [View] [🗑] │
│                                                                  │
│ Select: [All] [None] [Inverse Pairs]                            │
│ [Remove Selected] [Create Interaction] [Mark as Redundant]      │
└─────────────────────────────────────────────────────────────────┘
```

### Redundancy Analysis
```
┌─ Multicollinearity Detection ───────────────────────────────────┐
│                                                                  │
│ Variance Inflation Factor (VIF):                                │
│                                                                  │
│ ⚠️ VIF > 10 (High multicollinearity):                           │
│ • feat6: VIF = 12.3 (correlated with feat1, feat3)             │
│ • feat5: VIF = 11.8 (correlated with feat3, feat2)             │
│                                                                  │
│ ⚡ VIF > 5 (Moderate):                                          │
│ • feat3: VIF = 7.2                                              │
│                                                                  │
│ ✓ VIF < 5 (Acceptable):                                         │
│ • feat1: VIF = 2.1                                              │
│ • feat2: VIF = 3.4                                              │
│                                                                  │
│ [Remove High VIF Features] [Apply PCA] [Ridge Regression]       │
└─────────────────────────────────────────────────────────────────┘
```

### Feature Clustering
```
┌─ Hierarchical Clustering of Features ───────────────────────────┐
│                                                                  │
│                    ┌──────────────┐                             │
│           ┌────────┤ Cluster 1    │                             │
│           │        │ (feat1,feat6)│                             │
│     ┌─────┴─────┐  └──────────────┘                             │
│     │           │                                               │
│ ┌───┴───┐   ┌───┴───┐  ┌──────────────┐                        │
│ │feat1  │   │feat3  │  │ Cluster 2    │                        │
│ │feat6  │   │feat5  │──┤ (feat3,feat5)│                        │
│ └───┬───┘   └───────┘  └──────────────┘                        │
│     │                                                           │
│     │  ┌─────────┐                                              │
│     └──┤ feat2   │                                              │
│        └─────────┘                                              │
│                                                                  │
│ Method: [Ward ▼]    Distance: [1-r ▼]    Clusters: [Auto ▼]    │
│ [Dendrogram] [Heatmap] [Silhouette Score]                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Correlation with Target

### Target Analysis View
```
┌─ Correlations with Target: price ───────────────────────────────┐
│                                                                  │
│ [Bar Chart: Feature vs |Correlation|]    [Partial Correlations] │
│                                                                  │
│ Feature Selection:                                               │
│ ☑ feat6    r=0.89 ★★★   [Top predictor]                        │
│ ☑ feat3    r=0.67 ★★    [Strong predictor]                     │
│ ☑ feat5    r=0.45 ★     [Moderate predictor]                   │
│ ☐ feat2    r=0.12       [Weak predictor]                       │
│ ☐ feat8    r=-0.05      [No correlation]                       │
│                                                                  │
│ Selected Features R² = 0.85                                      │
│                                                                  │
│ [Partial Regression Plot] [Added Variable Plot]                 │
└─────────────────────────────────────────────────────────────────┘
```

### Feature Selection Recommendation
```
┌─ Recommended Feature Subset ────────────────────────────────────┐
│                                                                  │
│ Based on correlation analysis:                                   │
│                                                                  │
│ ✓ Keep:                                                          │
│   • feat6 (highest correlation with target)                     │
│   • feat4 (orthogonal to others, negative correlation)          │
│   • feat2 (adds unique variance)                                │
│                                                                  │
│ ⚠️ Consider Removing (redundant):                                │
│   • feat1 (r=0.89 with feat6)                                   │
│   • feat3 (r=0.85 with feat5)                                   │
│                                                                  │
│ 💡 Suggestion: Create interaction term feat6 × feat4            │
│                                                                  │
│ [Apply Recommendations] [Export Feature List]                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Advanced Analysis

### Partial Correlations
```
┌─ Partial Correlation Analysis ──────────────────────────────────┐
│                                                                  │
│ Control for: [feat3 ▼]                                           │
│                                                                  │
│ Original:    feat1 × feat2: r = 0.67                             │
│ Partial:     feat1 × feat2: r = 0.23 (controlling for feat3)     │
│                                                                  │
│ Explanation: 66% of correlation explained by feat3               │
│                                                                  │
│ [Semipartial] [Multiple Control] [Semipartial Plot]             │
└─────────────────────────────────────────────────────────────────┘
```

### Time-Lagged Correlations
```
┌─ Cross-Correlation (Time Series) ───────────────────────────────┐
│                                                                  │
│ [Line plot showing correlation at different lags]               │
│                                                                  │
│ Max correlation: r=0.78 at lag=3                                 │
│ Interpretation: feat2 leads feat1 by 3 time periods             │
│                                                                  │
│ Lag: [-10]────[0]────[+10]                                      │
│                                                                  │
│ [ACF] [PACF] [Granger Causality]                                │
└─────────────────────────────────────────────────────────────────┘
```

### Categorical Correlations
```
┌─ Categorical Association Analysis ──────────────────────────────┐
│                                                                  │
│ Cramer's V Matrix:                                               │
│            cat1    cat2    cat3                                  │
│ cat1       1.00    0.45    0.23                                  │
│ cat2       0.45    1.00    0.67                                  │
│ cat3       0.23    0.67    1.00                                  │
│                                                                  │
│ Contingency Tables:                                              │
│ [cat2 × cat3]    [Chi-square: 45.2, p<0.001]                   │
│                                                                  │
│ [Mosaic Plot] [Association Plot] [Chi-square Test]              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Export & Actions

### Export Options
```
┌─ Export Correlation Analysis ───────────────────────────────────┐
│                                                                  │
│ Format:                                                          │
│ ○ CSV Matrix                                                     │
│ ○ CSV Long Format (triangular)                                   │
│ ○ Excel (with formatting)                                        │
│ ○ PNG/SVG Image                                                  │
│ ○ Python Code (correlation computation)                          │
│                                                                  │
│ Include:                                                         │
│ ☑ Correlation coefficients                                       │
│ ☑ P-values                                                       │
│ ☑ Confidence intervals                                           │
│ ☑ Sample sizes                                                   │
│ ☐ Scatter plot data                                              │
│                                                                  │
│ [Export Current View] [Export Full Report]                      │
└─────────────────────────────────────────────────────────────────┘
```

### Decorrelation Actions
```
┌─ Apply Decorrelation ───────────────────────────────────────────┐
│                                                                  │
│ Method:                                                          │
│ ○ Remove highly correlated features (|r| > threshold)            │
│ ○ Principal Component Analysis (PCA)                             │
│ ○ Factor Analysis                                                │
│ ○ Independent Component Analysis (ICA)                           │
│ ○ Apply Ridge regularization                                     │
│                                                                  │
│ Threshold: [|r| > 0.9 ▼]                                         │
│ Keep: [First occurrence ▼] [Highest variance ▼] [Target corr ▼] │
│                                                                  │
│ Preview: Will remove 3 features: feat1, feat3, feat5            │
│                                                                  │
│ [Preview Changes] [Apply Decorrelation]                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### Correlation State
```python
CorrelationState = {
    "method": str,  # "pearson", "spearman", "kendall", etc.
    "features": List[str],
    "target": Optional[str],
    "matrix": pd.DataFrame,  # correlation matrix
    "p_values": pd.DataFrame,  # p-value matrix
    "confidence_intervals": Dict[Tuple, Tuple],  # (i,j) -> (lower, upper)
    "sample_sizes": pd.DataFrame,
    "significance_level": float,
    "threshold": float,
    "high_correlations": List[Dict],  # pairs with |r| > threshold
    "clusters": Dict[int, List[str]],  # cluster_id -> features
    "vif_scores": Dict[str, float]  # VIF for each feature
}
```

### Correlation Pair Data
```python
CorrelationPair = {
    "feature_1": str,
    "feature_2": str,
    "correlation": float,
    "method": str,
    "p_value": float,
    "ci_lower": float,
    "ci_upper": float,
    "n_samples": int,
    "is_significant": bool,
    "strength": str  # "very weak", "weak", "moderate", "strong", "very strong"
}
```

---

## Technical Implementation

### Efficient Computation
```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.cluster.hierarchy import linkage, dendrogram

def compute_correlation_matrix(df, method='pearson'):
    """Efficient correlation computation with p-values."""
    
    # Base correlation
    corr_matrix = df.corr(method=method)
    
    # P-values (pairwise)
    p_matrix = pd.DataFrame(
        np.zeros_like(corr_matrix),
        index=corr_matrix.index,
        columns=corr_matrix.columns
    )
    
    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i != j:
                if method == 'pearson':
                    _, p = pearsonr(df[col1], df[col2])
                elif method == 'spearman':
                    _, p = spearmanr(df[col1], df[col2])
                elif method == 'kendall':
                    _, p = kendalltau(df[col1], df[col2])
                p_matrix.iloc[i, j] = p
    
    return corr_matrix, p_matrix

def compute_vif(df):
    """Compute Variance Inflation Factor."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) 
                       for i in range(len(df.columns))]
    return vif_data
```

### Caching Strategy
- Cache correlation matrix for unchanged data
- Cache clustering results
- Invalidate on data update
- Store pre-computed scatter plot data for high-correlation pairs

### Performance Optimization
- Use Dask for large datasets (>100K rows)
- Compute correlations in parallel
- Use sampling for initial preview
- Lazy load scatter plots
