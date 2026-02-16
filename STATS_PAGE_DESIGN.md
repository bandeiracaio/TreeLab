# TreeLab Statistics Page Design Specification

## Overview

A comprehensive statistics page providing detailed descriptive statistics for the entire dataset, including both univariate and multivariate analyses with interactive visualizations.

---

## Page Structure

```
StatisticsPage
├── HeaderSection
│   ├── Dataset Info (rows, columns, memory usage)
│   ├── Data Quality Summary (missing values, duplicates)
│   └── Last Updated Timestamp
├── QuickStatsGrid
│   ├── Numeric Summary Cards (6 metrics)
│   └── Categorical Summary Cards (top categories)
├── TabContainer
│   ├── Tab1: Numeric Features
│   ├── Tab2: Categorical Features
│   ├── Tab3: Data Quality
│   └── Tab4: Multivariate Stats
└── ActionBar
    ├── Export Full Report
    ├── Compare with Previous
    └── Refresh Statistics
```

---

## Header Section

### Dataset Information Card
```
┌─────────────────────────────────────────────────────┐
│ Dataset: {filename}                                    │
│ Rows: {N:,}    Columns: {M}    Memory: {XX.X} MB      │
│ Missing Values: {X}%    Duplicates: {N} rows ({X}%)   │
│ Last Updated: {timestamp}                            │
└─────────────────────────────────────────────────────┘
```

### Data Quality Score
- **Overall Score**: XX/100 (color-coded: green ≥80, yellow 60-79, red <60)
- **Breakdown**:
  - Completeness: XX%
  - Consistency: XX%
  - Validity: XX%
  - Uniqueness: XX%

---

## Tab 1: Numeric Features

### Feature Selector
- **Search**: Filter features by name
- **Type Filter**: All / Integer / Float
- **Select All / Deselect All**
- **Preset Groups**: High cardinality / Low cardinality / All

### Statistics Table

| Feature | Count | Mean | Std | Min | 25% | 50% | 75% | Max | Missing | Skewness | Kurtosis |
|---------|-------|------|-----|-----|-----|-----|-----|-----|---------|----------|----------|
| feature1 | 1000 | 50.2 | 12.3 | 10 | 42 | 51 | 58 | 100 | 0% | -0.2 | 2.8 |
| feature2 | 1000 | 0.45 | 0.23 | 0.0 | 0.28 | 0.45 | 0.62 | 1.0 | 0% | 0.1 | 2.1 |

**Interactive Features:**
- Sort by any column
- Filter by range (e.g., skewness > 1)
- Export to CSV
- Copy to clipboard

### Distribution Overview

**Visualization Row 1:**
```
[Distribution Grid]              [Outlier Detection Summary]
     (mini histograms for           (count and severity per feature)
      all numeric features)
```

**Visualization Row 2:**
```
[Box Plot Matrix]                [Violin Plot Comparison]
     (side-by-side box plots)      (select features to compare)
```

**Visualization Row 3:**
```
[QQ Plot Grid]                   [Normality Test Results]
     (normality assessment)        (Shapiro-Wilk, Anderson-Darling)
```

### Detailed Single Feature View

When a feature is selected:

**Statistics Panel:**
```
┌─ Feature: {feature_name} ──────────────────────────────┐
│ Type: Numeric (Float64)                                │
│ Unique Values: N ({X}%)                                │
│ Zero Values: N ({X}%)                                  │
│ Negative Values: N ({X}%)                              │
│                                                        │
│ Central Tendency:                                      │
│   Mean: X.XXX ± X.XXX (SEM)                            │
│   Median: X.XXX                                        │
│   Mode: X.XXX                                          │
│   Geometric Mean: X.XXX                                │
│   Harmonic Mean: X.XXX                                 │
│                                                        │
│ Dispersion:                                            │
│   Range: [X.XXX, X.XXX]                                │
│   IQR: X.XXX                                           │
│   Variance: X.XXX                                      │
│   Std Dev: X.XXX                                       │
│   CV (Relative Std): X.XXX%                            │
│   MAD (Median Abs Dev): X.XXX                          │
│                                                        │
│ Shape:                                                 │
│   Skewness: X.XXX ({interpretation})                   │
│   Kurtosis: X.XXX ({normal/excess})                    │
│   Normality: {pass/fail} (p={X.XXX})                   │
│                                                        │
│ Percentiles:                                           │
│   5th: X.XXX    10th: X.XXX    25th: X.XXX             │
│   50th: X.XXX   75th: X.XXX    90th: X.XXX             │
│   95th: X.XXX   99th: X.XXX                            │
└────────────────────────────────────────────────────────┘
```

**Visualizations:**
```
[Histogram with KDE]    [Box Plot with Outliers]    [Cumulative Distribution]
     (bins auto-selected)     (individual points)       (ECDF)
```

---

## Tab 2: Categorical Features

### Statistics Table

| Feature | Unique | Most Common | Frequency | % of Total | Missing | Entropy | Mode Freq |
|---------|--------|-------------|-----------|------------|---------|---------|-----------|
| category1 | 5 | "A" | 450 | 45% | 0% | 1.52 | 450 |
| category2 | 100 | "X123" | 50 | 5% | 2% | 4.21 | 50 |

### Value Distribution

**For Selected Feature:**

```
[Bar Chart - Value Counts]    [Pie Chart - Proportions]    [Treemap - Hierarchical]
```

**Frequency Table:**
| Value | Count | Percentage | Cumulative % | Bar |
|-------|-------|-----------|--------------|-----|
| "A" | 450 | 45% | 45% | ████ |
| "B" | 300 | 30% | 75% | ██ |
| "C" | 150 | 15% | 90% | █ |

### Rare Categories Detection
- **Threshold**: < 1% of data
- **List**: Categories below threshold with counts
- **Recommendation**: Group rare categories or investigate

### Cardinality Analysis
```
[Cardinality Distribution]    [High Cardinality Alert]    [Unique Ratio]
     (unique count per feature)   (features with >100 unique)  (unique/total)
```

---

## Tab 3: Data Quality

### Missing Values Analysis

**Overall Missing Pattern:**
```
[Missing Value Heatmap]    [Missing Value Bar Chart]    [Missing Pattern Matrix]
     (by feature & sample)     (count per feature)         (co-occurrence)
```

**Missing Value Statistics:**
| Feature | Missing Count | Missing % | Type | Pattern | Recommendation |
|---------|--------------|-----------|------|---------|----------------|
| feat1 | 50 | 5% | MCAR | Random | Impute with mean |
| feat2 | 200 | 20% | MAR | Systematic | Investigate cause |

### Duplicate Analysis

**Duplicate Detection:**
```
[Duplicate Rows Count]    [Duplicate Patterns]    [Similarity Score Dist]
     (exact matches)          (fuzzy duplicates)       (string similarity)
```

**Duplicate Groups:**
| Group ID | Rows | Similarity | Key Fields | Action |
|----------|------|-----------|-----------|--------|
| 1 | 3 | 100% | All | Review & drop |
| 2 | 2 | 95% | Most | Fuzzy match |

### Outlier Detection

**Method Selection:**
- IQR Method (1.5 × IQR)
- Z-Score (> 3σ)
- Modified Z-Score
- Isolation Forest

**Outlier Summary:**
| Feature | Method | Outliers | % | Severity | Distribution |
|---------|--------|----------|---|----------|--------------|
| feat1 | IQR | 25 | 2.5% | Low | Right tail |
| feat2 | Z-Score | 10 | 1.0% | High | Both tails |

```
[Outlier Scatter Plot]    [Outlier Impact Analysis]    [Outlier Profile]
     (feature vs feature)      (on statistics)              (common patterns)
```

### Data Types & Validation

**Type Analysis:**
| Feature | Current Type | Suggested Type | Issue | Action |
|---------|-------------|----------------|-------|--------|
| id | Int64 | String | ID treated as numeric | Convert |
| date | Object | DateTime | Not parsed | Parse |

**Validation Rules:**
- Range violations: {count}
- Type mismatches: {count}
- Format inconsistencies: {count}

---

## Tab 4: Multivariate Statistics

### Covariance & Correlation

**Correlation Matrix:**
```
[Correlation Heatmap]    [Correlation Network Graph]    [Significant Correlations]
     (color-coded)          (force-directed layout)       (|r| > 0.7)
```

**Covariance Table:**
|  | feat1 | feat2 | feat3 | ... |
|--|-------|-------|-------|-----|
| feat1 | 12.5 | 3.2 | -1.5 | ... |
| feat2 | 3.2 | 8.7 | 0.9 | ... |

### Cross-tabulation

**Categorical vs Categorical:**
```
[Contingency Table]    [Chi-Square Test]    [Cramer's V]
     (with percentages)     (independence test)   (association strength)
```

**Numeric vs Categorical:**
```
[Group Statistics]    [ANOVA Table]    [Effect Size]
     (mean by category)   (F-test)        (eta-squared)
```

### Principal Component Preview

```
[Explained Variance Ratio]    [Component Loadings]    [Scree Plot]
     (first 5 components)       (feature contributions)   (eigenvalue decay)
```

---

## Interactive Features

### Column Customization
- Show/hide columns in tables
- Reorder columns via drag-drop
- Save custom views

### Filtering
- Filter rows by statistic value
- Filter features by type
- Filter by missing value threshold

### Comparison Mode
```
[Current Dataset]    vs    [Previous Version]
     (side-by-side statistics)
     
[Drift Detection]
- Mean shift: {features}
- Variance change: {features}
- New categories: {features}
```

### Export Options
- **Full Report**: PDF/HTML with all statistics
- **Summary CSV**: Key metrics only
- **Detailed CSV**: All statistics per feature
- **JSON**: Machine-readable format

---

## Data Structures

### Statistics State
```python
StatisticsState = {
    "dataset_info": {
        "n_rows": int,
        "n_columns": int,
        "memory_usage_mb": float,
        "timestamp": datetime
    },
    "numeric_features": {
        "feature_name": {
            "count": int,
            "mean": float,
            "std": float,
            "min": float,
            "25%": float,
            "50%": float,
            "75%": float,
            "max": float,
            "missing_count": int,
            "missing_pct": float,
            "skewness": float,
            "kurtosis": float,
            "unique_count": int,
            "zero_count": int,
            "negative_count": int,
            "normality_test": {
                "shapiro_wilk": {"statistic": float, "p_value": float},
                "anderson_darling": {"statistic": float, "critical_values": List}
            }
        }
    },
    "categorical_features": {
        "feature_name": {
            "unique_count": int,
            "mode": str,
            "mode_freq": int,
            "mode_pct": float,
            "missing_count": int,
            "entropy": float,
            "value_counts": Dict[str, int]
        }
    },
    "data_quality": {
        "total_missing": int,
        "total_missing_pct": float,
        "duplicate_rows": int,
        "outliers": Dict[str, List[int]],  # feature -> outlier indices
        "completeness_score": float,
        "consistency_score": float
    }
}
```

---

## Performance Considerations

### Computation Strategy
- **Lazy Evaluation**: Compute expensive stats on demand
- **Caching**: Cache results for unchanged datasets
- **Sampling**: Use sample for large datasets (>100K rows)
- **Progressive Loading**: Load basic stats first, advanced later

### Optimization
- **Vectorization**: Use NumPy/Pandas vectorized operations
- **Parallel Processing**: Compute per-feature stats in parallel
- **Incremental Updates**: Only recompute changed features

---

## Accessibility

### Screen Reader Support
- Table headers properly labeled
- Charts have data table alternatives
- Summary statistics read aloud

### Keyboard Navigation
- Tab through all interactive elements
- Arrow keys navigate tables
- Space/Enter to expand sections

### Visual Accessibility
- High contrast mode
- Colorblind-friendly palettes
- Adjustable font sizes
