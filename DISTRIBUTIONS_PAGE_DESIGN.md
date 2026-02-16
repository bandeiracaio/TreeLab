# TreeLab Distributions Page Design Specification

## Overview

An interactive distributions explorer for visualizing and analyzing the distribution of features in the dataset. Supports both univariate and multivariate distribution analysis with statistical tests and transformations.

---

## Page Structure

```
DistributionsPage
├── HeaderSection
│   ├── Feature Selector (dropdown/search)
│   ├── Distribution Type (auto-detected/manual)
│   └── Transformation Toggle
├── MainVisualizationArea
│   ├── PrimaryChart (large)
│   ├── SecondaryCharts (row of 3)
│   └── ComparisonChart (optional)
├── StatisticsPanel (side or bottom)
│   ├── Distribution Fit Tests
│   ├── Descriptive Statistics
│   └── Transformation Recommendations
├── DistributionGallery
│   ├── All Features Grid
│   └── Filtered Views
└── ActionBar
    ├── Export Chart
    ├── Compare Distributions
    ├── Apply Transformation
    └── Distribution Report
```

---

## Feature Selector

### Search & Filter
```
┌──────────────────────────────────────────────────────┐
│ 🔍 Search features...    [All ▼] [Numeric ▼] [Sort ▼] │
│                                                      │
│ ┌─ Recently Viewed ─┐  ┌─ High Variability ─┐       │
│ │ • feature_1       │  │ • feature_5       │       │
│ │ • feature_3       │  │ • feature_12      │       │
│ └───────────────────┘  └───────────────────┘       │
│                                                      │
│ ┌─ All Numeric Features ─────────────────────┐      │
│ │ ☑ feature_1    ☐ feature_2    ☑ feature_3  │      │
│ │ ☐ feature_4    ☑ feature_5    ☐ feature_6  │      │
│ └────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────┘
```

### Quick Stats Preview
Hover over feature shows:
- Distribution type guess
- Skewness & kurtosis
- Outlier count
- Missing %

---

## Primary Visualization: Distribution Plot

### Chart Types (Toggle)

**1. Histogram with KDE**
```
┌─ Feature: Age ───────────────────────────────────────┐
│                                                      │
│  Frequency                                            │
│  ▲                                                   │
│  │    ┌───┐                                          │
│  │   ┌┘   └┐     ┌────────┐                         │
│  │  ┌┘     └┐   ┌┘        └┐    ┌──┐               │
│  │ ┌┘       └───┘          └────┘  └──┐            │
│  │┌┘                                   └────┐       │
│  └┴────┬────┬────┬────┬────┬────┬────┬────┬───▶    │
│       20   30   40   50   60   70   80   90        │
│                                                      │
│ [Histogram] [KDE ▼] [Rug] [Normal Overlay]          │
│ Bins: [Auto ▼] [20] [+] [-]                          │
│ Bandwidth: [Auto ▼] [0.5] [Slider]                   │
└──────────────────────────────────────────────────────┘
```

**2. Box Plot with Swarm**
```
┌─ Feature: Income ────────────────────────────────────┐
│                                                      │
│   ○  ○                                              │
│  ○ │ ○    ○                                         │
│ ───┼──────┼───────────────────────────────          │
│    │   ┌──┴──┐                                      │
│    └───┤     ├─── Outliers: 12                      │
│        └──┬──┘                                      │
│ ──────────┼───────────────────────────────          │
│           │                                         │
└──────────────────────────────────────────────────────┘
```

**3. Violin Plot with Split**
```
┌─ Feature: Score (by Gender) ─────────────────────────┐
│                                                      │
│    Male  │  Female                                   │
│      /╱   │   /\                                    │
│     /  ╱  │  /  \                                   │
│    /    ╱ │ /    \                                  │
│   /______\│/______\                                 │
│      ▓▓▓    ▓▓▓▓▓▓                                  │
│                                                      │
│ [Split by: Gender ▼] [None] [Category] [Target]     │
└──────────────────────────────────────────────────────┘
```

**4. CDF / ECDF**
```
┌─ Cumulative Distribution ────────────────────────────┐
│                                                      │
│  Cumulative Probability                               │
│  100% ┤                                    ┌─────    │
│   75% ┤                           ┌───────┘         │
│   50% ┤                  ┌───────┘                  │
│   25% ┤         ┌───────┘                           │
│    0% ┼────┬────┴───┬───┴───┬───┴───┬───┴───▶       │
│       0   20      40      60      80     100        │
│                                                      │
│ [Theoretical CDF ▼] [Normal] [Uniform] [Exponential]│
└──────────────────────────────────────────────────────┘
```

**5. Q-Q Plot**
```
┌─ Q-Q Plot vs Normal ─────────────────────────────────┐
│                                                      │
│  Sample Quantiles                                     │
│    ▲                                                 │
│  4 ┤ ○                                              │
│  3 ┤   ○                                            │
│  2 ┤     ○  ○                                       │
│  1 ┤        ○  ○  ○                                 │
│  0 ┼──────────────────────────────────────          │
│ -1 ┤                    ○  ○  ○                     │
│ -2 ┤                             ○  ○               │
│ -3 ┤                                   ○            │
│    └────┬────┬────┬────┬────┬────┬────┬───▶        │
│        -3   -2   -1    0    1    2    3            │
│              Theoretical Quantiles                   │
│                                                      │
│ R² = 0.95    Skewness: -0.23 (left-tailed)         │
└──────────────────────────────────────────────────────┘
```

---

## Secondary Visualizations Row

### Row Layout
```
[Distribution Stats Card]    [Percentile Chart]    [Outlier Analysis]
```

**1. Distribution Statistics Card**
```
┌─ Distribution Characteristics ───────────────────────┐
│                                                      │
│ Shape Analysis:                                      │
│   Skewness: -0.23 ▓▓▓░░░░░ (Slightly left)          │
│   Kurtosis: 2.85  ▓▓▓▓░░░░ (Platykurtic)            │
│                                                      │
│ Tail Analysis:                                       │
│   Left Tail (5%): < 12.5                            │
│   Right Tail (5%): > 87.3                           │
│   Outliers: 12 (1.2%)                               │
│                                                      │
│ Moments:                                             │
│   Mean: 50.2          Variance: 144.5               │
│   Median: 51.0        Std Dev: 12.0                 │
│   Mode: 48.5          IQR: 16.0                     │
│                                                      │
│ [View Full Statistics] [Copy Values]                │
└──────────────────────────────────────────────────────┘
```

**2. Percentile Chart**
```
┌─ Percentile Distribution ────────────────────────────┐
│                                                      │
│  100% ┤                        ┌───────── 98.5      │
│   90% ┤                  ┌─────┘           85.2     │
│   75% ┤            ┌─────┘                  68.4    │
│   50% ┤      ┌─────┘                        51.0    │
│   25% ┤ ┌────┘                              35.6    │
│   10% ┤┘                                    18.3    │
│    0% ┤                                     2.1     │
│                                                      │
│ [Custom Percentile] [25%] [Value: 35.6]             │
└──────────────────────────────────────────────────────┘
```

**3. Outlier Analysis**
```
┌─ Outlier Detection ──────────────────────────────────┐
│                                                      │
│ Method: IQR (1.5 × IQR)                              │
│ Outliers: 12 samples (1.2%)                          │
│                                                      │
│ Outlier List:                                        │
│ ┌─────┬─────────┬──────────┬──────────┐             │
│ │ Row │ Value   │ Z-Score  │ Action   │             │
│ ├─────┼─────────┼──────────┼──────────┤             │
│ │  45 │  125.0  │   6.23   │ [View]   │             │
│ │ 127 │  -15.2  │  -5.45   │ [View]   │             │
│ │ 203 │  118.5  │   5.69   │ [View]   │             │
│ └─────┴─────────┴──────────┴──────────┘             │
│                                                      │
│ [Apply Outlier Treatment] [Export List]             │
└──────────────────────────────────────────────────────┘
```

---

## Distribution Fitting & Testing

### Fit Distribution Tests
```
┌─ Distribution Fit Analysis ──────────────────────────┐
│                                                      │
│ Test Results (Kolmogorov-Smirnov):                   │
│                                                      │
│ Distribution      │ Statistic │ p-value │ Fit       │
│───────────────────┼───────────┼─────────┼───────────│
│ Normal            │   0.042   │  0.234  │ ✓ Good    │
│ Log-Normal        │   0.038   │  0.312  │ ✓ Best    │
│ Gamma             │   0.051   │  0.128  │ ○ Fair    │
│ Exponential       │   0.089   │  0.003  │ ✗ Poor    │
│ Weibull           │   0.045   │  0.198  │ ○ Fair    │
│                                                  │
│ Recommended: Log-Normal distribution               │
│                                                      │
│ [View Fit Overlay] [Export Parameters] [Apply Transform]
└──────────────────────────────────────────────────────┘
```

### Theoretical Overlay
```
[Histogram with Theoretical PDF]
- Original data (bars)
- Fitted distribution (line)
- Residuals (bottom subplot)
```

---

## Transformation Studio

### Transformation Panel
```
┌─ Distribution Transformation ────────────────────────┐
│                                                      │
│ Current Distribution: Right-skewed (2.34)           │
│                                                      │
│ Transformation: [None ▼]                             │
│ • None                                              │
│ • Log (log(x))                                      │
│ • Log1p (log(1+x))                                  │
│ • Square Root                                       │
│ • Box-Cox                                           │
│ • Yeo-Johnson                                       │
│ • Quantile Transform                                │
│ • Power Transform                                   │
│                                                      │
│ Parameters:                                          │
│   Lambda: [Auto ▼] [0.5]                            │
│                                                      │
│ Before ──────→ After                                 │
│ Skewness: 2.34  →  0.12 ✓                           │
│ Kurtosis: 8.92  →  2.85 ✓                           │
│                                                      │
│ [Preview] [Apply to Feature] [Apply to All]         │
│                                                      │
│ [Revert] [Save as New Feature]                      │
└──────────────────────────────────────────────────────┘
```

### Transformation Comparison
```
[Before]    [After]    [Side-by-Side]
   │           │            │
   ▼           ▼            ▼
[Original] [Log Trans] [Split View]
  Skew:2.34  Skew:0.12  [Before|After]
```

---

## Distribution Gallery

### Grid View
```
┌──────────────────────────────────────────────────────┐
│ Distribution Gallery              [Filter ▼] [Sort ▼] │
├──────────────────────────────────────────────────────┤
│                                                      │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│ │ feature_1│ │ feature_2│ │ feature_3│ │ feature_4│ │
│ │ [hist]   │ │ [hist]   │ │ [hist]   │ │ [hist]   │ │
│ │ skew:0.2 │ │ skew:2.1 │ │ skew:-0.5│ │ skew:0.0 │ │
│ │ ⚠️ skewed│ │ ⚠️ skewed│ │ ✓ normal │ │ ✓ normal │ │
│ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
│                                                      │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│ │ feature_5│ │ feature_6│ │ feature_7│ │ feature_8│ │
│ │ [hist]   │ │ [hist]   │ │ [hist]   │ │ [hist]   │ │
│ │ skew:1.8 │ │ skew:-2.3│ │ skew:0.3 │ │ skew:0.1 │ │
│ │ ⚠️ skewed│ │ ⚠️ skewed│ │ ✓ normal │ │ ✓ normal │ │
│ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### Filter Options
- **Skewness**: Highly skewed (>1) / Moderate (0.5-1) / Normal (<0.5)
- **Outliers**: High outlier count / Low outlier count / Clean
- **Missing**: With missing values / Complete
- **Type**: Numeric / Categorical / Date

### Bulk Actions
```
[Select All Skewed] → [Apply Log Transform] → [Review Changes]
[Select All Outliers] → [Apply Outlier Treatment] → [Confirm]
```

---

## Multivariate Distribution Analysis

### Joint Distribution
```
┌─ Joint Distribution: Feature X vs Feature Y ─────────┐
│                                                      │
│ [Hexbin Plot]       [KDE Contour]      [Scatter]    │
│                                                      │
│    Feature Y                                        │
│  100 ┤         ·   ·                                 │
│   80 ┤      ·  ·  ·  ·                               │
│   60 ┤   ·  ·  ███  ·  ·                             │
│   40 ┤·  ·  ███████  ·  ·                            │
│   20 ┤·  █████████████  ·                            │
│    0 ┼────┬────┬────┬────┬────┬────┬───▶            │
│       0   20   40   60   80  100                    │
│                 Feature X                            │
│                                                      │
│ Correlation: 0.78    Regression Line: y = 0.8x + 10 │
│                                                      │
│ [View Marginals] [Regression Analysis] [Clustering] │
└──────────────────────────────────────────────────────┘
```

### Marginal Distributions
```
Top:    [Histogram of Feature X]
Right:  [Histogram of Feature Y] (rotated)
Center: [Joint plot]
```

---

## Export & Reporting

### Export Options
```
┌─ Export Distribution Analysis ───────────────────────┐
│                                                      │
│ Format:                                              │
│ ○ PNG Image (current view)                          │
│ ○ SVG Vector (editable)                             │
│ ○ PDF Report (full analysis)                        │
│ ○ HTML Interactive                                  │
│                                                      │
│ Include:                                             │
│ ☑ Distribution plot                                 │
│ ☑ Statistics summary                                │
│ ☑ Distribution fit tests                            │
│ ☑ Transformation recommendations                    │
│ ☐ Raw data sample                                   │
│                                                      │
│ [Export Current] [Export All Features]              │
└──────────────────────────────────────────────────────┘
```

### Distribution Report
Auto-generates PDF with:
1. Executive Summary
2. Feature-by-feature distributions
3. Problematic distributions highlighted
4. Transformation recommendations
5. Statistical test results

---

## Interactive Features

### Brushing & Linking
- Select range on histogram → Highlights in other views
- Select outliers → Shows in table
- Multiple brush selections supported

### Zoom & Pan
- Mouse wheel zoom on histograms
- Pan with drag
- Reset zoom button
- Zoom history (back/forward)

### Tooltip Information
```
┌─ Tooltip ──────────────────────────┐
│ Bin: 40-50                         │
│ Count: 123                         │
│ Percentage: 12.3%                  │
│ Cumulative: 45.6%                  │
│                                    │
│ Statistics:                        │
│   Mean: 45.2                       │
│   Std: 2.1                         │
└────────────────────────────────────┘
```

### Comparison Mode
```
┌─ Distribution Comparison ────────────────────────────┐
│                                                      │
│    [Feature A]              [Feature B]             │
│         │                       │                    │
│         ▼                       ▼                    │
│    ┌─────────┐             ┌─────────┐              │
│    │  /\    │             │  /\    │              │
│    │ /  \   │    vs       │ /  \   │              │
│    │/    \  │             │/    \  │              │
│    └─────────┘             └─────────┘              │
│                                                      │
│    Kolmogorov-Smirnov test: p=0.023 (different)    │
│                                                      │
│    [Swap] [Overlay] [Statistical Test]             │
└──────────────────────────────────────────────────────┘
```

---

## Data Structures

### Distribution State
```python
DistributionState = {
    "selected_feature": str,
    "chart_type": str,  # "histogram", "box", "violin", "qq", "cdf"
    "transformations": {
        "applied": str,  # "none", "log", "sqrt", etc.
        "lambda": float,
        "before_stats": Dict,
        "after_stats": Dict
    },
    "histogram_config": {
        "bins": int,
        "bin_method": str,  # "auto", "fd", "sturges", "manual"
        "kde": bool,
        "kde_bandwidth": float,
        "rug": bool
    },
    "outliers": {
        "method": str,
        "threshold": float,
        "indices": List[int],
        "values": List[float]
    },
    "fit_tests": {
        "distributions_tested": List[str],
        "best_fit": str,
        "parameters": Dict,
        "statistics": Dict[str, {"statistic": float, "p_value": float}]
    }
}
```

---

## Technical Implementation

### Computation
```python
# Efficient histogram computation
import numpy as np
from scipy import stats

def compute_distribution_stats(series):
    return {
        "basic": series.describe(),
        "moments": {
            "skewness": stats.skew(series),
            "kurtosis": stats.kurtosis(series),
            "moment_3": stats.moment(series, 3),
            "moment_4": stats.moment(series, 4)
        },
        "normality_tests": {
            "shapiro": stats.shapiro(series.sample(min(5000, len(series)))),
            "anderson": stats.anderson(series, dist='norm'),
            "kstest": stats.kstest(series, 'norm', args=(series.mean(), series.std()))
        }
    }
```

### Caching Strategy
- Cache histogram data for different bin counts
- Cache KDE evaluations
- Cache transformation results
- Invalidate on data change

### Performance
- Use WebGL for large scatter plots (>10K points)
- Downsample for preview, full data on zoom
- Lazy load distribution fits
