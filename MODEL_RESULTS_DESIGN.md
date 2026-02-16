# TreeLab Model Results Page Designs

This document defines the specialized results pages for each model type in TreeLab. Use this as a template for designing pages for new models.

---

## Page Structure Overview

Each model results page should include:

1. **Header Section** - Model name, timestamp, dataset info
2. **Performance Metrics** - Model-specific metrics
3. **Visualizations** - Charts and graphs specific to the model type
4. **Model Details** - Parameters, feature importance, tree structure (if applicable)
5. **Action Buttons** - Save, export, compare, predict

---

## Classification Models

### DecisionTreeClassifier

#### Header
- Model: Decision Tree Classifier
- Training time: X seconds
- Samples: N train / M test
- Classes: [list of class names]

#### Key Metrics (Top Row Cards)
- **Accuracy**: XX.X%
- **Precision**: XX.X% (macro/micro/weighted)
- **Recall**: XX.X% (macro/micro/weighted)
- **F1-Score**: XX.X% (macro/micro/weighted)
- **Tree Depth**: N levels
- **Nodes**: N total nodes / M leaf nodes

#### Visualizations

**Row 1: Performance Overview**
```
[Confusion Matrix Heatmap]    [Classification Report Table]
     (interactive)                (precision/recall/f1 per class)
```

**Row 2: Tree Structure**
```
[Decision Tree Visualization]
     (collapsible tree diagram)
     - Show/hide branches
     - Node details on hover
     - Export as image
```

**Row 3: Feature Importance & Decision Path**
```
[Feature Importance Bar Chart]    [Sample Decision Paths]
     (top 10 features)              (paths for random samples)
```

**Row 4: Class Distribution**
```
[Predicted vs Actual Distribution]    [ROC Curve per Class]
     (bar chart comparison)              (one-vs-rest curves)
```

#### Model Details Panel (Collapsible)
- **Parameters Used**:
  - max_depth: X
  - min_samples_split: X
  - min_samples_leaf: X
  - criterion: gini/entropy
  - class_weight: balanced/None
  
- **Feature Importance Table**:
  | Feature | Importance | Gini Importance |
  |---------|-----------|----------------|
  
- **Tree Statistics**:
  - Total nodes: N
  - Leaf nodes: M
  - Max depth: D
  - Avg samples per leaf: X

#### Special Features
- **Interactive Tree Explorer**: Click nodes to see decision rules
- **Path Tracer**: Select a sample to see its path through the tree
- **Pruning Suggestion**: Recommend optimal depth based on validation
- **Rule Extraction**: Export decision rules as IF-THEN statements

---

### RandomForestClassifier

#### Header
- Model: Random Forest Classifier
- Training time: X seconds
- Samples: N train / M test
- Classes: [list of class names]
- Trees: N estimators

#### Key Metrics (Top Row Cards)
- **Accuracy**: XX.X%
- **Precision**: XX.X%
- **Recall**: XX.X%
- **F1-Score**: XX.X%
- **OOB Score**: XX.X% (if enabled)
- **CV Score**: XX.X% ± X.XX (if cross-validation performed)

#### Visualizations

**Row 1: Performance Overview**
```
[Confusion Matrix]    [Classification Report]    [ROC Curves]
```

**Row 2: Feature Importance (Multiple Views)**
```
[Mean Importance]    [Importance Distribution]    [Feature Correlation Heatmap]
     (bar chart)         (box plot across trees)       (top features)
```

**Row 3: Out-of-Bag Analysis**
```
[OOB Error Rate by Tree]    [Feature Importance with OOB]
     (learning curve)           (stability analysis)
```

**Row 4: Individual Tree Analysis**
```
[Tree Depth Distribution]    [Leaf Size Distribution]    [Tree Similarity Matrix]
     (histogram)                  (histogram)                (clustering of trees)
```

#### Model Details Panel
- **Parameters**:
  - n_estimators: X
  - max_depth: X
  - min_samples_split: X
  - min_samples_leaf: X
  - max_features: X
  - bootstrap: True/False
  - oob_score: True/False

- **Feature Importance Table**:
  | Feature | Mean Importance | Std Dev | Permutation Importance |
  |---------|----------------|---------|----------------------|

- **Per-Tree Statistics**:
  - Avg tree depth: X
  - Avg leaves per tree: X
  - Best tree index: X (highest individual accuracy)

#### Special Features
- **Feature Selection Helper**: Identify features to drop based on low importance
- **Partial Dependence Plots**: Show feature effect on predictions
- **Tree Sampling**: View a random sample of individual trees
- **Prediction Confidence**: Show confidence intervals for predictions

---

### GradientBoostingClassifier

#### Header
- Model: Gradient Boosting Classifier
- Training time: X seconds
- Samples: N train / M test
- Boosting stages: N estimators
- Learning rate: X.XXX

#### Key Metrics (Top Row Cards)
- **Accuracy**: XX.X%
- **Precision**: XX.X%
- **Recall**: XX.X%
- **F1-Score**: XX.X%
- **Log Loss**: X.XXX
- **Best Stage**: N (early stopping)

#### Visualizations

**Row 1: Performance & Deviance**
```
[Confusion Matrix]    [Deviance by Stage]    [ROC Curves]
     (heatmap)          (training curve)        (one-vs-rest)
```

**Row 2: Boosting Analysis**
```
[Training Error by Stage]    [Feature Importance Accumulation]
     (showing convergence)        (how importance evolves)
```

**Row 3: Tree Analysis**
```
[Tree Depth by Stage]    [Leaf Count by Stage]    [Feature Usage Heatmap]
     (usually decreases)       (stabilizes)            (which features when)
```

**Row 4: Partial Dependence**
```
[Top Feature PDPs]    [Pairwise Interactions]
     (marginal effects)    (feature interactions)
```

#### Model Details Panel
- **Parameters**:
  - n_estimators: X
  - learning_rate: X.XXX
  - max_depth: X
  - min_samples_split: X
  - subsample: X.X
  - loss: deviance/exponential

- **Training Progress**:
  | Stage | Train Loss | Val Loss | Improvement |
  |-------|-----------|----------|-------------|

- **Feature Importance**:
  | Feature | Importance | First Used (Stage) | Times Used |
  |---------|-----------|-------------------|-----------|

#### Special Features
- **Stage-by-Stage Animation**: Visualize how predictions evolve
- **Early Stopping Analysis**: Show optimal stopping point
- **Feature Interaction Detection**: Identify important feature pairs
- **Residual Analysis**: Plot residuals by stage

---

### ExtraTreesClassifier

#### Header
- Model: Extra Trees Classifier
- Training time: X seconds
- Samples: N train / M test
- Trees: N estimators
- Randomness: High (feature & split randomization)

#### Key Metrics (Top Row Cards)
- **Accuracy**: XX.X%
- **Precision**: XX.X%
- **Recall**: XX.X%
- **F1-Score**: XX.X%
- **Bias**: Lower (due to randomization)
- **Variance**: Higher (ensemble reduces this)

#### Visualizations

**Row 1: Performance**
```
[Confusion Matrix]    [Classification Report]    [ROC Curves]
```

**Row 2: Feature Importance**
```
[Mean Importance]    [Importance Variance]    [Comparison with RandomForest]
     (bar chart)         (error bars)             (if both trained)
```

**Row 3: Randomization Analysis**
```
[Split Quality Distribution]    [Feature Selection Frequency]
     (vs RandomForest)            (how often each feature chosen)
```

**Row 4: Tree Characteristics**
```
[Tree Structure Comparison]    [Prediction Correlation Matrix]
     (ExtraTrees vs others)        (between trees in ensemble)
```

#### Model Details Panel
- **Parameters**:
  - n_estimators: X
  - max_depth: X
  - min_samples_split: X
  - min_samples_leaf: X
  - max_features: X
  - bootstrap: True/False

- **Comparison Metrics** (if RandomForest also trained):
  - Accuracy diff: ±X.XX%
  - Training time ratio: X.Xx
  - Feature importance correlation: X.XX

#### Special Features
- **Comparison Mode**: Side-by-side with RandomForest
- **Speed Analysis**: Training time breakdown
- **Overfitting Check**: Compare train/test performance gap
- **Randomization Visualization**: Show effect of random splits

---

## Regression Models

### DecisionTreeRegressor

#### Header
- Model: Decision Tree Regressor
- Training time: X seconds
- Samples: N train / M test
- Target: [target column name]
- Target range: [min] to [max]

#### Key Metrics (Top Row Cards)
- **R² Score**: X.XXX
- **RMSE**: X.XXX
- **MAE**: X.XXX
- **MAPE**: XX.X%
- **Tree Depth**: N levels
- **Nodes**: N total / M leaves

#### Visualizations

**Row 1: Prediction Quality**
```
[Actual vs Predicted Scatter]    [Residuals Plot]    [Residuals Distribution]
     (with perfect line)            (vs predicted)       (histogram + QQ)
```

**Row 2: Tree Structure**
```
[Decision Tree Visualization]
     (similar to classifier)
     - Show split thresholds
     - Leaf value distributions
```

**Row 3: Error Analysis**
```
[Error by Feature]    [Prediction Interval]    [Outlier Detection]
     (which features cause errors)  (confidence bands)     (residual outliers)
```

**Row 4: Feature Importance**
```
[Feature Importance]    [Partial Dependence (top 3)]
     (bar chart)              (how each feature affects target)
```

#### Model Details Panel
- **Parameters**:
  - max_depth: X
  - min_samples_split: X
  - min_samples_leaf: X
  - criterion: mse/mae/friedman_mse
  - splitter: best/random

- **Target Statistics**:
  - Mean: X.XX
  - Std: X.XX
  - Min: X.XX
  - Max: X.XX

- **Error Statistics**:
  | Metric | Train | Test | Difference |
  |--------|-------|------|-----------|
  | R²     | X.XXX | X.XXX | ±X.XXX    |
  | RMSE   | X.XXX | X.XXX | ±X.XXX    |
  | MAE    | X.XXX | X.XXX | ±X.XXX    |

#### Special Features
- **Prediction Intervals**: Calculate confidence intervals for predictions
- **Segment Analysis**: Analyze performance by target value ranges
- **Leaf Analysis**: Show statistics for each leaf node
- **Monotonicity Check**: Verify if relationships make sense

---

### RandomForestRegressor

#### Header
- Model: Random Forest Regressor
- Training time: X seconds
- Samples: N train / M test
- Target: [target column name]
- Trees: N estimators

#### Key Metrics (Top Row Cards)
- **R² Score**: X.XXX
- **RMSE**: X.XXX
- **MAE**: X.XXX
- **Explained Variance**: X.XXX
- **OOB Score**: X.XXX (if enabled)

#### Visualizations

**Row 1: Prediction Analysis**
```
[Actual vs Predicted]    [Residuals Plot]    [Residuals by Feature]
```

**Row 2: Feature Importance**
```
[Mean Importance]    [Importance Stability]    [Permutation Importance]
     (bar chart)         (across trees)            (validation)
```

**Row 3: Uncertainty Quantification**
```
[Prediction Intervals]    [Tree Agreement Plot]    [Out-of-Bag Predictions]
     (mean ± std)            (variance by prediction)    (vs actual)
```

**Row 4: Ensemble Analysis**
```
[Learning Curve]    [Tree Depth Distribution]    [Feature Usage Patterns]
     (more trees = better?)   (across ensemble)           (which trees use which features)
```

#### Model Details Panel
- **Parameters**: (same as classifier)

- **Feature Importance**:
  | Feature | Importance | Std | Importance Trend |
  |---------|-----------|-----|-----------------|

- **Prediction Statistics**:
  - Mean prediction: X.XX
  - Prediction std: X.XX
  - Max tree disagreement: X.XX

#### Special Features
- **Uncertainty Visualization**: Show prediction confidence
- **Interpolation vs Extrapolation**: Identify out-of-distribution predictions
- **Feature Interaction Explorer**: Discover non-linear relationships
- **Learning Curve**: How performance improves with more trees

---

## Analysis & Visualization Actions

### FeatureImportanceAction

#### Header
- Action: Feature Importance Analysis
- Model: Random Forest
- Training time: X seconds
- Features analyzed: N

#### Key Metrics
- **Top Feature**: [name] (XX.X% importance)
- **Cumulative Top 5**: XX.X%
- **Cumulative Top 10**: XX.X%
- **Features with 0 importance**: N

#### Visualizations

**Row 1: Importance Overview**
```
[Horizontal Bar Chart - Top 20]    [Cumulative Importance Curve]
```

**Row 2: Multiple Importance Types**
```
[Gini Importance]    [Permutation Importance]    [Comparison]
     (default RF)        (more reliable)            (correlation)
```

**Row 3: Stability Analysis**
```
[Importance by Subsample]    [Bootstrap Confidence Intervals]
     (robustness check)        (statistical significance)
```

**Row 4: Feature Relationships**
```
[Importance vs Correlation with Target]    [Redundancy Analysis]
     (scatter plot)                            (cluster similar features)
```

#### Recommendations Panel
- **Features to Consider Removing**: List of low-importance features
- **Potential Redundancies**: Highly correlated important features
- **Suggested Next Steps**: Feature engineering suggestions

---

### TuneHyperparametersAction

#### Header
- Action: Hyperparameter Tuning
- Method: Grid Search / Random Search
- Duration: X minutes
- Configurations tried: N
- Best score: X.XXX

#### Key Metrics
- **Best Parameters**: [display key params]
- **Best CV Score**: X.XXX ± X.XXX
- **Test Score**: X.XXX
- **Improvement over default**: +X.XXX%

#### Visualizations

**Row 1: Search Results**
```
[Score Heatmap (2D)]    [Parallel Coordinates Plot]    [Score Distribution]
     (top 2 params)         (all param combinations)      (histogram)
```

**Row 2: Parameter Importance**
```
[Parameter Sensitivity]    [Partial Dependence (hyperparams)]
     (which params matter most)    (effect on performance)
```

**Row 3: Validation Analysis**
```
[CV Score by Fold]    [Learning Curves]    [Overfitting Detection]
     (stability check)    (train vs val)      (gap analysis)
```

**Row 4: Comparison**
```
[Before vs After]    [All Models Comparison Table]
```

#### Results Table
| Rank | Params | CV Score | Test Score | Time |
|------|--------|----------|-----------|------|
| 1    | {...}  | X.XXX    | X.XXX     | Xs   |
| 2    | {...}  | X.XXX    | -         | Xs   |

#### Special Features
- **Export Best Model**: Download pickled model
- **Apply Best Params**: One-click apply to retrain
- **Refinement Suggestion**: Narrow search around best params

---

### SHAPSummaryAction

#### Header
- Action: SHAP Analysis
- Model: Random Forest
- Computation time: X seconds
- Samples analyzed: N (background) / M (explained)

#### Key Metrics
- **Top Driver**: [feature] (mean |SHAP| = X.XX)
- **Most Consistent**: [feature] (low std)
- **Interaction Strength**: [feature pair]

#### Visualizations

**Row 1: Global Importance**
```
[SHAP Summary Plot (beeswarm)]    [SHAP Bar Chart (mean abs)]
     (feature value coloring)        (global importance)
```

**Row 2: Individual Explanations**
```
[Force Plot (sample)]    [Waterfall Plot]    [Decision Plot]
     (push/pull factors)     (cumulative)       (path to prediction)
```

**Row 3: Feature Analysis**
```
[SHAP Dependence (top 3)]    [SHAP Interaction Values]
     (feature vs SHAP value)     (feature pair effects)
```

**Row 4: Model Comparison**
```
[SHAP Comparison (2 models)]    [Prediction Explanation]
     (why models differ)            (natural language summary)
```

#### Special Features
- **Sample Explorer**: Click to see SHAP for any sample
- **Counterfactual**: "What if" analysis
- **Export Explanations**: JSON/CSV of SHAP values
- **Bias Detection**: Identify potential fairness issues

---

### BinningScorecardAction

#### Header
- Action: Binning Scorecard
- Feature: [column name]
- Target: [target column]
- Bins created: N
- IV (Information Value): X.XXX

#### Key Metrics
- **IV Score**: X.XXX ([predictive strength])
  - < 0.02: Weak
  - 0.02-0.1: Medium
  - > 0.1: Strong
- **KS Statistic**: XX.X%
- **Best Bin**: [range] (WOE: X.XX)

#### Visualizations

**Row 1: Binning Overview**
```
[WOE by Bin]    [Event Rate by Bin]    [Distribution by Bin]
     (bar chart)     (line chart)         (histogram)
```

**Row 2: Statistical Analysis**
```
[IV Contribution by Bin]    [WOE Trend]    [Correlation Check]
     (monotonicity)           (should be smooth)   (with target)
```

**Row 3: Scorecard Table**
```
[Detailed Scorecard]
| Bin | Range | Count | Event Rate | WOE | IV | Score |
```

**Row 4: Validation**
```
[Train vs Test WOE]    [Stability Index (PSI)]    [Gini by Bin]
```

#### Scorecard Summary
- **Binning Strategy**: [method used]
- **Monotonic**: Yes/No
- **Missing Treatment**: [strategy]
- **Recommended**: Yes/No with reasoning

#### Special Features
- **Auto-Binning**: Try different bin counts
- **WOE Transformation**: Apply WOE encoding
- **Score Scaling**: Convert to credit score points
- **Export**: Excel/PDF scorecard format

---

## Universal Elements

### All Model Pages Should Include:

1. **Action Bar** (top right)
   - Save Model
   - Export to Python
   - Export to BigQuery SQL
   - Compare with Previous
   - Make Predictions (on new data)

2. **Data Summary** (collapsible)
   - Training samples: N
   - Features used: [list]
   - Target column: [name]
   - Preprocessing applied: [list]

3. **Performance Context**
   - Baseline comparison (if available)
   - Previous model comparison (if applicable)
   - Dataset benchmark (typical scores for this dataset)

4. **Warnings/Alerts**
   - Overfitting detected
   - Underfitting detected
   - Class imbalance warning
   - Feature scaling issues
   - Missing value handling notes

5. **Next Steps Suggestions**
   - Try different hyperparameters
   - Feature engineering ideas
   - Alternative models to try
   - Validation strategies

---

## Future Model Designs

### LinearRegression
- Coefficients table with p-values
- Residual analysis plots
- VIF (multicollinearity) analysis
- Normality tests
- Heteroscedasticity check

### LogisticRegression
- Odds ratios
- ROC and PR curves
- Calibration plot
- Coefficient significance
- Decision boundary visualization (2D)

### SVM (SVC/SVR)
- Support vectors visualization
- Margin visualization (2D)
- Kernel comparison
- C parameter sensitivity
- Decision function plot

### KNN
- K optimization curve
- Distance metric analysis
- Nearest neighbors visualization
- Decision boundary (2D)
- Local outlier factor

### Neural Network (MLP)
- Training curves (loss/accuracy)
- Weight visualization
- Activation patterns
- Gradient flow
- Layer-wise analysis

### GradientBoostingRegressor
- Similar to classifier but with regression metrics
- Residual plots by stage
- Feature importance evolution
- Early stopping analysis

### XGBoost / LightGBM / CatBoost
- Native importance plots
- SHAP integration
- Training curves
- Feature interaction
- Tree visualizer

---

## Unimplemented Models - Detailed Designs

### Classification Models

#### HistGradientBoostingClassifier

#### Header
- Model: Histogram-based Gradient Boosting Classifier
- Training time: X seconds (fast!)
- Samples: N train / M test
- Histogram bins: 255 (default)
- Boosting stages: N estimators

#### Key Metrics (Top Row Cards)
- **Accuracy**: XX.X%
- **Precision**: XX.X%
- **Recall**: XX.X%
- **F1-Score**: XX.X%
- **Training Time**: X.XXs (highlight as fast)
- **Memory Used**: XX MB

#### Visualizations

**Row 1: Performance & Speed**
```
[Confusion Matrix]    [Classification Report]    [Training Time Comparison]
     (heatmap)              (table)                  (vs GradientBoosting)
```

**Row 2: Histogram Analysis**
```
[Feature Histograms]    [Split Points Distribution]    [Bin Usage Heatmap]
     (per feature)            (where splits occur)          (which bins matter)
```

**Row 3: Boosting Progress**
```
[Loss Curve by Stage]    [Feature Importance Evolution]    [Leaf Count per Tree]
     (shows convergence)         (stabilization)               (typically constant)
```

**Row 4: Comparison with Regular GB**
```
[Accuracy Comparison]    [Speedup Factor]    [Memory Usage]
     (Hist vs Regular)       (X times faster)     (much lower)
```

#### Model Details Panel
- **Parameters**:
  - max_iter: X (instead of n_estimators)
  - learning_rate: X.XXX
  - max_depth: X
  - max_bins: 255 (histogram bins)
  - early_stopping: Yes/No
  - scoring: accuracy/log_loss

- **Training Statistics**:
  - Actual iterations: N (may stop early)
  - Best iteration: N
  - Time per iteration: X.XXs
  - Histogram build time: X.XXs

#### Special Features
- **Speed Highlight**: Show dramatic speed improvement
- **Large Dataset Badge**: Optimized for datasets >10K samples
- **Categorical Features**: Native handling without encoding
- **Early Stopping Visualization**: Show where training stopped

---

#### AdaBoostClassifier

#### Header
- Model: AdaBoost Classifier
- Training time: X seconds
- Samples: N train / M test
- Weak learners: N estimators
- Learning algorithm: DecisionTree/SAMME/SAMME.R

#### Key Metrics (Top Row Cards)
- **Accuracy**: XX.X%
- **Precision**: XX.X%
- **Recall**: XX.X%
- **F1-Score**: XX.X%
- **Final Estimator Weight**: X.XX
- **Convergence**: Yes/No

#### Visualizations

**Row 1: Performance**
```
[Confusion Matrix]    [ROC Curves]    [Classification Report]
```

**Row 2: Boosting Dynamics**
```
[Error Rate by Stage]    [Classifier Weights]    [Sample Weights Distribution]
     (should decrease)       (estimator importance)     (how samples get reweighted)
```

**Row 3: Estimator Analysis**
```
[Individual Estimator Accuracy]    [Estimators Over Time]    [Weight vs Accuracy]
     (weak learners improve)          (visual timeline)          (correlation)
```

**Row 4: Misclassification Analysis**
```
[Hard-to-Classify Samples]    [Weight Accumulation]    [Decision Boundaries]
     (consistently misclassified)   (total weight by sample)     (evolution)
```

#### Model Details Panel
- **Parameters**:
  - n_estimators: X
  - learning_rate: X.XXX
  - algorithm: SAMME/SAMME.R
  - estimator: DecisionTree(max_depth=X)

- **Estimators Table**:
  | Stage | Estimator | Weight | Error | Samples Reweighted |
  |-------|-----------|--------|-------|-------------------|

- **Sample Weight Analysis**:
  - Most weighted samples: [list]
  - Average weight progression
  - Weight concentration (Gini)

#### Special Features
- **Estimator Inspector**: View each weak learner
- **Sample Weight Tracker**: See which samples get focused
- **Boosting Animation**: Visualize sequential learning
- **Comparison Mode**: AdaBoost vs Gradient Boosting

---

#### BaggingClassifier

#### Header
- Model: Bagging Classifier
- Training time: X seconds
- Samples: N train / M test
- Base estimator: [type]
- Bootstrap samples: Yes/No

#### Key Metrics (Top Row Cards)
- **Accuracy**: XX.X%
- **Precision**: XX.X%
- **Recall**: XX.X%
- **F1-Score**: XX.X%
- **OOB Score**: XX.X% (if bootstrap=True)
- **Variance Reduction**: XX.X%

#### Visualizations

**Row 1: Performance**
```
[Confusion Matrix]    [ROC Curves]    [Classification Report]
```

**Row 2: Bootstrap Analysis**
```
[Sample Coverage Heatmap]    [OOB Error per Estimator]    [Bootstrap Diversity]
     (how many times each      (out-of-bag validation)      (overlap between bags)
      sample is selected)
```

**Row 3: Ensemble Diversity**
```
[Prediction Correlation Matrix]    [Disagreement Analysis]    [Voting Patterns]
     (between estimators)            (where models disagree)     (ensemble decisions)
```

**Row 4: Base Estimator Analysis**
```
[Single vs Bagged Performance]    [Bias-Variance Decomposition]    [Learning Curve]
     (show improvement)                (theoretical benefits)          (more estimators)
```

#### Model Details Panel
- **Parameters**:
  - base_estimator: [type]
  - n_estimators: X
  - max_samples: X.X (fraction)
  - max_features: X.X (fraction)
  - bootstrap: True/False
  - bootstrap_features: True/False
  - oob_score: True/False

- **Bootstrap Statistics**:
  | Estimator | Samples Used | Features Used | OOB Accuracy |
  |-----------|-------------|--------------|-------------|

- **Diversity Metrics**:
  - Q-statistic: X.XX
  - Correlation: X.XX
  - Disagreement: XX.X%

#### Special Features
- **Bootstrap Visualizer**: Show which samples in each bag
- **Diversity Dashboard**: Measure ensemble disagreement
- **Base Estimator Inspector**: Compare individual models
- **Variance Analysis**: Show reduction in prediction variance

---

#### LogisticRegression

#### Header
- Model: Logistic Regression
- Training time: X seconds
- Samples: N train / M test
- Regularization: L1/L2/ElasticNet
- Solver: [lbfgs/liblinear/saga/etc]

#### Key Metrics (Top Row Cards)
- **Accuracy**: XX.X%
- **Log Loss**: X.XXX
- **AUC-ROC**: X.XXX
- **AUC-PR**: X.XXX
- **Convergence**: Yes/No (iterations: N)
- **Coefficients**: N non-zero (if L1)

#### Visualizations

**Row 1: Performance & Calibration**
```
[Confusion Matrix]    [ROC Curve]    [Calibration Plot]
     (heatmap)            (with AUC)      (predicted vs actual prob)
```

**Row 2: Coefficient Analysis**
```
[Coefficient Bar Chart]    [Odds Ratios]    [Coefficient Paths (if L1)]
     (sorted by magnitude)     (exp(coef))       (regularization path)
```

**Row 3: Statistical Significance**
```
[P-values Heatmap]    [Confidence Intervals]    [Feature Significance]
     (statistical test)     (95% CI for coefs)       (significant features)
```

**Row 4: Decision Boundary**
```
[2D Decision Boundary]    [Probability Heatmap]    [Prediction Confidence]
     (top 2 features)         (probability space)       (distribution)
```

#### Model Details Panel
- **Parameters**:
  - penalty: l1/l2/elasticnet/none
  - C: X.XXX (inverse regularization)
  - solver: [name]
  - max_iter: X (converged in N)
  - class_weight: balanced/None
  - multi_class: ovr/multinomial

- **Coefficients Table**:
  | Feature | Coefficient | Std Error | Z-score | P-value | Odds Ratio |
  |---------|-------------|-----------|---------|---------|-----------|

- **Model Fit Statistics**:
  - Log-likelihood: X.XXX
  - Null deviance: X.XXX
  - Residual deviance: X.XXX
  - AIC: X.XXX
  - Pseudo R²: X.XXX

#### Special Features
- **Regularization Path**: Visualize how coefficients shrink
- **Feature Selection**: Identify which L1 regularization keeps
- **Probability Calibration**: Check if probabilities are well-calibrated
- **Multicollinearity Warning**: VIF scores for features
- **Decision Rule Export**: Human-readable IF-THEN rules

---

#### SVC (Support Vector Classifier)

#### Header
- Model: Support Vector Classifier
- Training time: X seconds
- Samples: N train / M test
- Kernel: [linear/poly/rbf/sigmoid]
- Support vectors: N (X% of training data)

#### Key Metrics (Top Row Cards)
- **Accuracy**: XX.X%
- **Support Vectors**: N (X%)
- **Margin Violations**: N
- **C Parameter**: X.XXX
- **Gamma**: X.XXX (if applicable)
- **Training Time**: X.XXs

#### Visualizations

**Row 1: Performance**
```
[Confusion Matrix]    [ROC Curves]    [Classification Report]
```

**Row 2: Support Vector Analysis**
```
[Support Vector Distribution]    [SV by Class]    [SV Proximity to Boundary]
     (where they lie in space)      (per class)       (distance histogram)
```

**Row 3: Decision Function**
```
[Decision Function Values]    [Margin Width Visualization]    [Misclassified SVs]
     (distribution)                  (2D projection)              (highlight)
```

**Row 4: Kernel Analysis (if non-linear)**
```
[Kernel Matrix Heatmap]    [Feature Space Mapping]    [Decision Boundary (2D)]
     (similarity matrix)        (PCA projection)            (with margin)
```

#### Model Details Panel
- **Parameters**:
  - C: X.XXX (regularization)
  - kernel: [type]
  - degree: X (for poly)
  - gamma: X.XXX (scale/auto/value)
  - coef0: X.XXX (for poly/sigmoid)
  - shrinking: True/False
  - probability: True/False

- **Support Vector Statistics**:
  | Class | Total SVs | Boundary SVs | Violations |
  |-------|-----------|--------------|-----------|

- **Decision Function**:
  - Bias (intercept): X.XXX
  - Dual coefficients shape: (N,)
  - Margin width: X.XXX

#### Special Features
- **Support Vector Explorer**: View support vector samples
- **Kernel Comparison**: Compare different kernels side-by-side
- **C Parameter Analysis**: Show effect of regularization
- **Decision Boundary Visualizer**: Interactive 2D/3D plot
- **SV Influence**: Show which support vectors matter most

---

#### KNeighborsClassifier

#### Header
- Model: K-Nearest Neighbors Classifier
- Training time: X seconds (very fast!)
- Prediction time: X seconds
- Samples: N train / M test
- K neighbors: N

#### Key Metrics (Top Row Cards)
- **Accuracy**: XX.X%
- **K Value**: N
- **Training Time**: X.XXs (instant)
- **Prediction Time**: X.XXs
- **Distance Metric**: [minkowski/euclidean/manhattan/etc]
- **Weighting**: uniform/distance

#### Visualizations

**Row 1: Performance**
```
[Confusion Matrix]    [Classification Report]    [ROC Curves]
```

**Row 2: K Optimization**
```
[Accuracy vs K Curve]    [Optimal K Indicator]    [Cross-Validation Scores]
     (elbow method)           (suggested K)          (for different K values)
```

**Row 3: Neighborhood Analysis**
```
[Distance Distribution]    [Neighbor Class Distribution]    [Voronoi Diagram (2D)]
     (to K-th neighbor)        (class breakdown)               (decision regions)
```

**Row 4: Sample Analysis**
```
[Prediction Confidence]    [Border Samples]    [Outlier Detection]
     (agreement among K)       (hard to classify)   (far from neighbors)
```

#### Model Details Panel
- **Parameters**:
  - n_neighbors: X
  - weights: uniform/distance
  - algorithm: [auto/ball_tree/kd_tree/brute]
  - leaf_size: X (for tree algorithms)
  - metric: [name]
  - p: X (Minkowski power)

- **Neighbor Statistics**:
  | Sample | True Class | Pred Class | Confidence | Avg Distance to K |
  |--------|-----------|-----------|-----------|------------------|

- **Algorithm Info**:
  - Algorithm used: [actual]
  - Tree build time: X.XXs
  - Memory usage: XX MB

#### Special Features
- **K Selector**: Interactive slider to test different K values
- **Neighbor Explorer**: Click sample to see its K neighbors
- **Distance Metric Comparison**: Try different metrics
- **Prediction Confidence**: Based on neighbor agreement
- **Local Outlier Detection**: Identify samples far from any cluster

---

#### Naive Bayes (GaussianNB / MultinomialNB / BernoulliNB)

#### Header
- Model: [Gaussian/Multinomial/Bernoulli] Naive Bayes
- Training time: X seconds (instant)
- Samples: N train / M test
- Features: N
- Classes: [list]

#### Key Metrics (Top Row Cards)
- **Accuracy**: XX.X%
- **Log Loss**: X.XXX
- **Training Time**: X.XXs (lightning fast)
- **Features**: N
- **Class Priors**: [distribution]
- **Conditional Independence**: Assumed

#### Visualizations

**Row 1: Performance**
```
[Confusion Matrix]    [ROC Curves]    [Classification Report]
```

**Row 2: Feature Probabilities**
```
[Feature Distributions by Class]    [Mean/Var by Class]    [Likelihood Ratios]
     (Gaussian curves if GaussianNB)    (table)               (feature importance)
```

**Row 3: Class Probabilities**
```
[Prior vs Posterior]    [Evidence by Feature]    [Decision Surface (2D)]
     (how priors update)    (contribution)          (probabilistic regions)
```

**Row 4: Independence Assumption**
```
[Feature Correlation Matrix]    [Conditional Correlations]    [Violation Analysis]
     (should be low)               (by class)                   (where assumption breaks)
```

#### Model Details Panel
- **Parameters**:
  - var_smoothing: X.XXe-X (Gaussian)
  - alpha: X.XXX (Laplace smoothing for multinomial)
  - fit_prior: True/False
  - class_prior: [list if specified]

- **Class Statistics**:
  | Class | Prior | Samples | Mean (per feature) | Var (per feature) |
  |-------|-------|---------|-------------------|------------------|

- **Feature Likelihoods**:
  | Feature | Class 0 Mean | Class 1 Mean | Separation |
  |---------|-------------|-------------|-----------|

#### Special Features
- **Real-time Prediction**: Instant prediction on new samples
- **Feature Independence Test**: Check assumption validity
- **Comparison with Non-Naive**: vs logistic regression
- **Text Classification Mode**: Special view for MultinomialNB
- **Probability Calibration**: Natural probability outputs

---

#### MLPClassifier (Neural Network)

#### Header
- Model: Multi-Layer Perceptron Classifier
- Training time: X seconds
- Samples: N train / M test
- Architecture: [layers]
- Training iterations: N

#### Key Metrics (Top Row Cards)
- **Accuracy**: XX.X%
- **Final Loss**: X.XXX
- **Iterations**: N
- **Convergence**: Yes/No
- **Parameters**: N (weights + biases)
- **Training Time**: X.XXs

#### Visualizations

**Row 1: Performance**
```
[Confusion Matrix]    [ROC Curves]    [Classification Report]
```

**Row 2: Training Dynamics**
```
[Loss Curve]    [Accuracy Curve]    [Gradient Norms]
     (train/validation)   (per iteration)      (vanishing/exploding check)
```

**Row 3: Network Architecture**
```
[Network Diagram]    [Weight Distribution]    [Activation Patterns]
     (layer topology)    (histogram per layer)     (forward pass visualization)
```

**Row 4: Layer Analysis**
```
[Weight Heatmap by Layer]    [Bias Distribution]    [Feature Importance (permutation)]
```

#### Model Details Panel
- **Architecture**:
  - hidden_layer_sizes: (X, Y, Z)
  - activation: [relu/tanh/logistic]
  - solver: [adam/sgd/lbfgs]
  - alpha: X.XXX (L2 regularization)
  - batch_size: X
  - learning_rate: constant/invscaling/adaptive
  - learning_rate_init: X.XXX

- **Training History**:
  | Iteration | Train Loss | Val Loss | Train Acc | Val Acc | Time |
  |-----------|-----------|----------|-----------|---------|------|

- **Weight Statistics**:
  | Layer | Weights | Biases | Mean Weight | Weight Std |
  |-------|---------|--------|-------------|-----------|

#### Special Features
- **Architecture Builder**: Interactive network design
- **Training Animation**: Watch loss decrease
- **Gradient Checker**: Detect vanishing/exploding gradients
- **Regularization Visualization**: Show weight decay
- **Neuron Activator**: See which neurons fire for inputs
- **Comparison with Shallow**: MLP vs Random Forest

---

#### LinearDiscriminantAnalysis (LDA)

#### Header
- Model: Linear Discriminant Analysis
- Training time: X seconds
- Samples: N train / M test
- Classes: K
- Dimensions: N features → M components

#### Key Metrics (Top Row Cards)
- **Accuracy**: XX.X%
- **Components Used**: M
- **Explained Variance**: XX.X%
- **Separability**: XX.X% (ratio of between/within class variance)
- **Linearity Assumption**: Assumed

#### Visualizations

**Row 1: Performance**
```
[Confusion Matrix]    [ROC Curves]    [Classification Report]
```

**Row 2: Discriminant Components**
```
[LDA Projection (2D)]    [Class Centroids]    [Decision Boundaries]
     (linear combinations)    (in original space)   (linear separations)
```

**Row 3: Variance Analysis**
```
[Between-Class Variance]    [Within-Class Variance]    [Separability Ratio]
     (per component)            (per component)           (eigenvalues)
```

**Row 4: Feature Contribution**
```
[Component Loadings]    [Feature Discriminability]    [Covariance Structure]
     (how features combine)    (univariate separability)    (class covariances)
```

#### Model Details Panel
- **Parameters**:
  - solver: [svd/lsqr/eigen]
  - shrinkage: None/auto/float
  - priors: [class priors]
  - n_components: M
  - store_covariance: True/False

- **Discriminant Components**:
  | Component | Eigenvalue | Explained Var | Cumulative |
  |-----------|-----------|--------------|-----------|

- **Class Statistics**:
  | Class | Prior | Mean Vector | Covariance | Centroid |
  |-------|-------|-------------|-----------|----------|

#### Special Features
- **Dimensionality Reduction**: Show LDA as preprocessing
- **Comparison with PCA**: Supervised vs unsupervised
- **Class Separability**: Measure how well classes are separated
- **Bayes Optimality**: Theoretical best performance
- **Gaussian Assumption Check**: Verify normality of features

---

#### QuadraticDiscriminantAnalysis (QDA)

#### Header
- Model: Quadratic Discriminant Analysis
- Training time: X seconds
- Samples: N train / M test
- Classes: K
- Quadratic boundaries: Yes

#### Key Metrics (Top Row Cards)
- **Accuracy**: XX.X%
- **Parameters**: N (grows with K × features²)
- **Regularization**: X.XXX (if used)
- **Covariance Matrices**: K (one per class)
- **Overfitting Risk**: High if K > features

#### Visualizations

**Row 1: Performance**
```
[Confusion Matrix]    [ROC Curves]    [Classification Report]
```

**Row 2: Decision Boundaries**
```
[2D Decision Regions]    [Quadratic Boundaries]    [Comparison with LDA]
     (curved boundaries)      (mathematical form)        (linear vs quadratic)
```

**Row 3: Covariance Analysis**
```
[Class Covariance Matrices]    [Covariance Differences]    [Regularization Effect]
     (heatmap per class)          (where classes differ)       (shrinkage impact)
```

**Row 4: Parameter Complexity**
```
[Parameters per Class]    [Model Complexity Gauge]    [Overfitting Warning]
     (quadratic growth)       (vs LDA vs data size)        (if applicable)
```

#### Model Details Panel
- **Parameters**:
  - priors: [class priors]
  - reg_param: X.XXX (regularization)
  - store_covariance: True/False
  - tol: X.XXe-X

- **Covariance Matrices**:
  | Class | Covariance Shape | Determinant | Condition Number |
  |-------|-----------------|-------------|-----------------|

- **Quadratic Discriminant**:
  - Equation per class: Show quadratic form
  - Rotations: Class-specific
  - Scaling: Different per class

#### Special Features
- **LDA vs QDA Comparison**: Side-by-side performance
- **Regularization Tuner**: Find optimal reg_param
- **Covariance Visualization**: Interactive matrix explorer
- **Sample Size Warning**: Alert if K > features
- **Decision Surface Plot**: 3D visualization

---

### Regression Models

#### GradientBoostingRegressor

#### Header
- Model: Gradient Boosting Regressor
- Training time: X seconds
- Samples: N train / M test
- Target range: [min] to [max]
- Boosting stages: N estimators

#### Key Metrics (Top Row Cards)
- **R² Score**: X.XXX
- **RMSE**: X.XXX
- **MAE**: X.XXX
- **Best Stage**: N (early stopping)
- **Final Loss**: X.XXX
- **Improvement**: X.XXX% (over first stage)

#### Visualizations

**Row 1: Prediction Quality**
```
[Actual vs Predicted]    [Residuals Plot]    [Residuals Distribution]
```

**Row 2: Boosting Progress**
```
[Loss Curve by Stage]    [Feature Importance Evolution]    [Residuals by Stage]
     (train/validation)        (how importance stabilizes)      (getting smaller)
```

**Row 3: Tree Analysis**
```
[Tree Depth by Stage]    [Leaf Count by Stage]    [Feature Usage Heatmap]
```

**Row 4: Partial Dependence**
```
[Top Feature PDPs]    [Pairwise Interactions]    [ICE Plots]
     (marginal effects)    (feature interactions)      (individual effects)
```

#### Model Details Panel
- **Parameters**: (similar to classifier)
- **Loss Function**: [ls/lad/huber/quantile]
- **Stage-wise Performance**:
  | Stage | Train Loss | Val Loss | RMSE | MAE |
  |-------|-----------|----------|------|-----|

#### Special Features
- **Stage Animation**: Watch residuals shrink
- **Loss Function Comparison**: LS vs Huber vs LAD
- **Quantile Regression Mode**: If loss=quantile
- **Early Stopping Analysis**: Optimal iteration detection

---

#### HistGradientBoostingRegressor

#### Header
- Model: Histogram-based Gradient Boosting Regressor
- Training time: X seconds (fast!)
- Samples: N train / M test
- Histogram bins: 255
- Target range: [min] to [max]

#### Key Metrics (Top Row Cards)
- **R² Score**: X.XXX
- **RMSE**: X.XXX
- **MAE**: X.XXX
- **Training Speed**: X.XXs
- **Memory**: XX MB

#### Visualizations

**Row 1: Prediction Quality**
```
[Actual vs Predicted]    [Residuals]    [Residuals by Bin]
```

**Row 2: Histogram Analysis**
```
[Target Histogram with Bins]    [Split Points]    [Bin-wise Performance]
     (how data is discretized)      (optimal splits)     (error by target range)
```

**Row 3: Speed & Accuracy**
```
[Training Time Comparison]    [Accuracy vs Speed]    [Scalability Chart]
     (vs GradientBoosting)       (trade-off curve)       (time vs samples)
```

**Row 4: Feature Effects**
```
[Partial Dependence]    [Feature Importance]    [Interaction Effects]
```

#### Special Features
- **Speed Badge**: Highlight as optimized for speed
- **Big Data Mode**: Handles millions of samples
- **Categorical Support**: Native categorical handling
- **Monotonic Constraints**: Enforce increasing/decreasing relationships

---

#### AdaBoostRegressor

#### Header
- Model: AdaBoost Regressor
- Training time: X seconds
- Samples: N train / M test
- Target range: [min] to [max]
- Loss function: [linear/square/exponential]

#### Key Metrics (Top Row Cards)
- **R² Score**: X.XXX
- **RMSE**: X.XXX
- **MAE**: X.XXX
- **Final Estimator Weight**: X.XX
- **Loss Type**: [linear/square/exponential]

#### Visualizations

**Row 1: Prediction Quality**
```
[Actual vs Predicted]    [Residuals]    [Error Distribution]
```

**Row 2: Boosting Dynamics**
```
[Error by Stage]    [Sample Weight Progression]    [Outlier Impact]
     (how errors decrease)   (hard samples get weight)     (outlier sensitivity)
```

**Row 3: Estimator Analysis**
```
[Individual Estimator Performance]    [Weight Accumulation]    [Learning Curve]
```

**Row 4: Robustness**
```
[Loss Function Comparison]    [Outlier Detection]    [Sensitivity Analysis]
     (which loss is best)        (weighted samples)      (to noise)
```

#### Special Features
- **Loss Function Selector**: Compare linear/square/exponential
- **Outlier Analysis**: Show which samples get highest weights
- **Robustness Test**: Add noise and measure impact
- **vs Gradient Boosting**: Comparison view

---

#### BaggingRegressor

#### Header
- Model: Bagging Regressor
- Training time: X seconds
- Samples: N train / M test
- Base estimator: [type]
- Bootstrap: Yes/No

#### Key Metrics (Top Row Cards)
- **R² Score**: X.XXX
- **RMSE**: X.XXX
- **MAE**: X.XXX
- **OOB Score**: X.XXX
- **Variance Reduction**: X.XXX

#### Visualizations

**Row 1: Prediction Quality**
```
[Actual vs Predicted]    [Residuals]    [Residuals Distribution]
```

**Row 2: Ensemble Analysis**
```
[Prediction Intervals]    [Tree Agreement]    [Bootstrap Coverage]
     (mean ± std)            (variance by prediction)   (sample selection)
```

**Row 3: Base Estimator**
```
[Single vs Bagged]    [Bias-Variance Trade-off]    [Learning Curve]
```

**Row 4: Uncertainty**
```
[Prediction Variance]    [Confidence Bands]    [High Uncertainty Regions]
```

#### Special Features
- **Uncertainty Quantification**: Natural prediction intervals
- **Bootstrap Inspector**: See which samples in each estimator
- **Diversity Analysis**: Measure ensemble disagreement
- **Base Estimator Tuning**: Optimize weak learner

---

#### LinearRegression

#### Header
- Model: Linear Regression (OLS)
- Training time: X seconds (instant)
- Samples: N train / M test
- Features: N
- Target range: [min] to [max]

#### Key Metrics (Top Row Cards)
- **R² Score**: X.XXX
- **Adjusted R²**: X.XXX
- **RMSE**: X.XXX
- **MAE**: X.XXX
- **F-statistic**: X.XXX
- **Condition Number**: X.XXX

#### Visualizations

**Row 1: Prediction Quality**
```
[Actual vs Predicted]    [Residuals]    [Residuals QQ Plot]
     (with CI bands)        (homoscedasticity)   (normality check)
```

**Row 2: Coefficients**
```
[Coefficient Bar Chart]    [Standard Errors]    [t-statistics]
     (with error bars)         (uncertainty)         (significance)
```

**Row 3: Diagnostics**
```
[Influence Plot]    [Leverage vs Residuals]    [Cook's Distance]
     (outlier detection)   (influential points)      (impact measure)
```

**Row 4: Assumption Checks**
```
[Multicollinearity (VIF)]    [Heteroscedasticity Test]    [Linearity Check]
     (VIF > 10 warning)        (Breusch-Pagan)              (partial residual plots)
```

#### Model Details Panel
- **Parameters**:
  - fit_intercept: True/False
  - normalize: deprecated (use pipeline)
  - n_jobs: X

- **Coefficients Table**:
  | Feature | Coefficient | Std Error | t-value | P-value | CI Lower | CI Upper |
  |---------|-------------|-----------|---------|---------|----------|----------|

- **Model Fit Statistics**:
  - F-statistic: X.XXX (p-value: X.XXX)
  - Log-likelihood: X.XXX
  - AIC: X.XXX
  - BIC: X.XXX
  - Condition number: X.XXX (multicollinearity warning if high)

#### Special Features
- **Diagnostic Dashboard**: Full regression diagnostics
- **Outlier Detection**: Leverage, influence, Cook's distance
- **Assumption Violation Alerts**: Non-linearity, heteroscedasticity, multicollinearity
- **Coefficient Interpretation**: Plain language explanations
- **Prediction Intervals**: Statistical confidence bands

---

#### Ridge Regression

#### Header
- Model: Ridge Regression (L2 Regularization)
- Training time: X seconds
- Samples: N train / M test
- Alpha: X.XXX (regularization strength)
- Solver: [auto/svd/cholesky/etc]

#### Key Metrics (Top Row Cards)
- **R² Score**: X.XXX
- **RMSE**: X.XXX
- **Effective DoF**: X.XX (degrees of freedom)
- **Alpha**: X.XXX
- **Condition Number**: X.XXX (improved from OLS)

#### Visualizations

**Row 1: Prediction Quality**
```
[Actual vs Predicted]    [Residuals]    [Residuals Analysis]
```

**Row 2: Regularization Path**
```
[Coefficient Paths]    [Ridge Trace Plot]    [Optimal Alpha]
     (as alpha varies)      (coefficient shrinkage)   (cross-validated)
```

**Row 3: Bias-Variance Trade-off**
```
[Bias Squared]    [Variance]    [MSE Decomposition]
     (increases)      (decreases)    (optimal balance)
```

**Row 4: Comparison**
```
[Ridge vs OLS]    [Ridge vs Lasso]    [Regularization Effect]
```

#### Model Details Panel
- **Parameters**:
  - alpha: X.XXX
  - fit_intercept: True/False
  - normalize: deprecated
  - solver: [name]
  - positive: True/False

- **Coefficient Comparison**:
  | Feature | OLS Coef | Ridge Coef | Shrinkage |
  |---------|----------|-----------|-----------|

- **Cross-Validation Results** (if used):
  | Alpha | Mean CV Score | Std | Selection |
  |-------|--------------|-----|-----------|

#### Special Features
- **Alpha Tuner**: Interactive slider to see effect
- **GCV Mode**: Generalized Cross-Validation
- **Coefficient Shrinkage Visualization**: See L2 penalty effect
- **Multicollinearity Relief**: Compare condition numbers
- **vs Lasso Comparison**: Side-by-side regularization

---

#### Lasso Regression

#### Header
- Model: Lasso Regression (L1 Regularization)
- Training time: X seconds
- Samples: N train / M test
- Alpha: X.XXX
- Non-zero coefficients: N / M total

#### Key Metrics (Top Row Cards)
- **R² Score**: X.XXX
- **RMSE**: X.XXX
- **Alpha**: X.XXX
- **Features Selected**: N / M (X%)
- **Sparsity**: XX.X%

#### Visualizations

**Row 1: Prediction Quality**
```
[Actual vs Predicted]    [Residuals]    [Residuals by Feature]
```

**Row 2: Regularization Path**
```
[Lasso Path]    [Coefficient Selection]    [BIC/AIC by Alpha]
     (as alpha decreases)   (order of selection)      (model selection criteria)
```

**Row 3: Feature Selection**
```
[Selected Features]    [Dropped Features]    [Selection Path]
     (non-zero coefs)       (zeroed out)          (when each enters)
```

**Row 4: Stability**
```
[Stability Selection]    [Bootstrap Selection]    [Path Consistency]
     (robust selection)      (across resamples)        (alpha sensitivity)
```

#### Model Details Panel
- **Parameters**:
  - alpha: X.XXX
  - fit_intercept: True/False
  - precompute: True/False
  - max_iter: X
  - tol: X.XXX
  - warm_start: True/False
  - positive: True/False
  - selection: cyclic/random

- **Coefficient Table**:
  | Feature | Coefficient | Absolute Value | Selected |
  |---------|-------------|----------------|----------|

- **Regularization Path**:
  | Alpha | R² | Features | AIC | BIC |
  |-------|----|---------|-----|-----|

#### Special Features
- **Feature Selection Mode**: Automatic variable selection
- **Alpha Optimization**: Cross-validated alpha
- **Stability Analysis**: Check selection consistency
- **LARS Visualization**: Least Angle Regression steps
- **Elastic Net Bridge**: Transition to Elastic Net

---

#### ElasticNet

#### Header
- Model: Elastic Net (L1 + L2 Regularization)
- Training time: X seconds
- Samples: N train / M test
- Alpha: X.XXX
- L1 Ratio: X.XX (balance between L1/L2)

#### Key Metrics (Top Row Cards)
- **R² Score**: X.XXX
- **RMSE**: X.XXX
- **Alpha**: X.XXX
- **L1 Ratio**: X.XX (X% L1, Y% L2)
- **Features Selected**: N / M

#### Visualizations

**Row 1: Prediction Quality**
```
[Actual vs Predicted]    [Residuals]    [Feature Importance]
```

**Row 2: Regularization Grid**
```
[Heatmap (Alpha vs L1 Ratio)]    [Optimal Parameters]    [Path Visualization]
     (CV score surface)              (marked)                (regularization paths)
```

**Row 3: L1 vs L2 Balance**
```
[Lasso-like Behavior]    [Ridge-like Behavior]    [Elastic Net Advantage]
     (when l1_ratio=1)        (when l1_ratio=0)        (correlated features)
```

**Row 4: Comparison**
```
[Elastic Net vs Lasso]    [Elastic Net vs Ridge]    [Three-way Comparison]
```

#### Model Details Panel
- **Parameters**:
  - alpha: X.XXX
  - l1_ratio: X.XX
  - fit_intercept: True/False
  - max_iter: X
  - tol: X.XXX
  - warm_start: True/False
  - positive: True/False
  - selection: cyclic/random

- **Grid Search Results** (if used):
  | Alpha | L1 Ratio | CV Score | Features |
  |-------|---------|----------|---------|

- **Coefficient Profile**:
  | Feature | ElasticNet | Lasso | Ridge |
  |---------|-----------|-------|-------|

#### Special Features
- **Grid Visualization**: Alpha vs L1 ratio heatmap
- **Correlation Handling**: Group selection for correlated features
- **Automatic Tuning**: Cross-validated Elastic Net (ElasticNetCV)
- **Three-way Comparison**: Elastic Net vs Lasso vs Ridge

---

#### BayesianRidge

#### Header
- Model: Bayesian Ridge Regression
- Training time: X seconds
- Samples: N train / M test
- Uncertainty: Probabilistic predictions
- Evidence: X.XXX (model evidence)

#### Key Metrics (Top Row Cards)
- **R² Score**: X.XXX
- **RMSE**: X.XXX
- **Alpha (precision)**: X.XXX
- **Lambda (coef precision)**: X.XXX
- **Log Marginal Likelihood**: X.XXX

#### Visualizations

**Row 1: Prediction Quality**
```
[Actual vs Predicted]    [Residuals]    [Predictive Distribution]
     (with uncertainty)      (analysis)       (probabilistic predictions)
```

**Row 2: Uncertainty Quantification**
```
[Prediction Intervals]    [Epistemic Uncertainty]    [Aleatoric Uncertainty]
     (mean ± 2*std)           (model uncertainty)        (data noise)
```

**Row 3: Hyperparameter Evolution**
```
[Alpha Convergence]    [Lambda Convergence]    [Evidence Maximization]
     (noise precision)       (weight precision)        (optimization path)
```

**Row 4: Comparison**
```
[Bayesian vs Frequentist]    [Uncertainty vs Error]    [Calibration Plot]
```

#### Model Details Panel
- **Parameters**:
  - max_iter: X
  - tol: X.XXX
  - alpha_1: X.XXX (Gamma hyperparameter)
  - alpha_2: X.XXX
  - lambda_1: X.XXX
  - lambda_2: X.XXX
  - fit_intercept: True/False
  - compute_score: True/False

- **Posterior Distribution**:
  | Feature | Mean Coef | Std Dev | 95% CI Lower | 95% CI Upper |
  |---------|-----------|---------|--------------|--------------|

- **Model Evidence**:
  - Log marginal likelihood: X.XXX
  - Iterations to converge: N
  - Alpha (noise): X.XXX ± X.XXX
  - Lambda (weights): X.XXX ± X.XXX

#### Special Features
- **Probabilistic Predictions**: Full predictive distributions
- **Uncertainty Decomposition**: Epistemic vs aleatoric
- **Bayesian Model Comparison**: Evidence-based selection
- **Credible Intervals**: Bayesian confidence intervals
- **Prior Sensitivity**: Effect of hyperpriors

---

#### HuberRegressor

#### Header
- Model: Huber Regressor (Robust to outliers)
- Training time: X seconds
- Samples: N train / M test
- Epsilon: X.XX (outlier threshold)
- Alpha: X.XXX (regularization)

#### Key Metrics (Top Row Cards)
- **R² Score**: X.XXX
- **RMSE**: X.XXX
- **Huber Loss**: X.XXX
- **Outliers Identified**: N (X%)
- **Robustness**: Yes/No

#### Visualizations

**Row 1: Prediction Quality**
```
[Actual vs Predicted]    [Residuals]    [Outlier Highlighting]
     (robust fit)            (check normality)   (epsilon band)
```

**Row 2: Robustness Analysis**
```
[Loss Function Shape]    [Inliers vs Outliers]    [Residual Distribution]
     (quadratic vs linear)     (classification)          (by group)
```

**Row 3: Comparison**
```
[Huber vs OLS]    [Huber vs Ridge]    [Outlier Impact Analysis]
     (robustness gain)   (regularization + robust)   (coefficient stability)
```

**Row 4: Sensitivity**
```
[Epsilon Tuning]    [Breakdown Point]    [Influence Function]
     (optimal threshold)   (robustness measure)    (sensitivity curve)
```

#### Model Details Panel
- **Parameters**:
  - epsilon: X.XX (huber parameter)
  - alpha: X.XXX (regularization)
  - fit_intercept: True/False
  - max_iter: X
  - tol: X.XXX

- **Outlier Analysis**:
  | Sample | Residual | Weight | Outlier? | Influence |
  |--------|----------|--------|----------|-----------|

- **Robustness Metrics**:
  - Breakdown point: X.XX%
  - Efficiency: XX.X% (relative to OLS)
  - Outliers downweighted: N

#### Special Features
- **Outlier Detection**: Automatic outlier identification
- **Epsilon Optimization**: Find optimal threshold
- **Robust Diagnostics**: Standardized residuals, influence
- **Breakdown Analysis**: Measure robustness
- **Comparison with OLS**: Show outlier impact

---

#### TheilSenRegressor

#### Header
- Model: Theil-Sen Regressor (Robust multivariate)
- Training time: X seconds
- Samples: N train / M test
- Subpopulation: X samples used
- Breakdown point: XX.X%

#### Key Metrics (Top Row Cards)
- **R² Score**: X.XXX
- **RMSE**: X.XXX
- **MAE**: X.XXX
- **Subsamples**: X (for large datasets)
- **Robustness**: Very High

#### Visualizations

**Row 1: Prediction Quality**
```
[Actual vs Predicted]    [Residuals]    [Residuals by Magnitude]
```

**Row 2: Robustness**
```
[Medoid Selection]    [Slope Distribution]    [Confidence Ellipse]
     (pairwise slopes)    (robust estimate)      (for coefficients)
```

**Row 3: Outlier Resistance**
```
[Outlier Impact Test]    [Breakdown Analysis]    [Efficiency Comparison]
     (add outliers)          (robustness limit)      (vs OLS)
```

**Row 4: Multivariate Analysis**
```
[Coefficient Stability]    [Subsample Consistency]    [Convergence Check]
     (across subsamples)      (if n_samples > max_subpopulation)
```

#### Model Details Panel
- **Parameters**:
  - fit_intercept: True/False
  - copy_X: True/False
  - max_subpopulation: X (for large datasets)
  - n_subsamples: X (for stochastic)
  - random_state: X
  - tol: X.XXX

- **Robustness Analysis**:
  - Breakdown point: XX.X%
  - Efficiency: XX.X%
  - Computation: O(n²) or O(max_subpopulation)
  - Subsamples evaluated: X

#### Special Features
- **Robust Regression**: Highly resistant to outliers
- **Multivariate Capability**: Handles multiple features robustly
- **Subpopulation Mode**: For large datasets
- **Breakdown Demonstration**: Visualize robustness
- **vs RANSAC Comparison**: Two robust approaches

---

#### RANSACRegressor

#### Header
- Model: RANSAC Regressor (Random Sample Consensus)
- Training time: X seconds
- Samples: N train / M test
- Inliers: N (X%)
- Outliers: N (X%)

#### Key Metrics (Top Row Cards)
- **R² Score (Inliers)**: X.XXX
- **RMSE (Inliers)**: X.XXX
- **Inlier Ratio**: XX.X%
- **Iterations**: N
- **Residual Threshold**: X.XXX

#### Visualizations

**Row 1: Prediction Quality**
```
[Actual vs Predicted]    [Inliers Highlighted]    [Outliers Identified]
     (inliers only)          (vs outliers)           (threshold boundary)
```

**Row 2: RANSAC Process**
```
[Iteration History]    [Model Scores]    [Inlier Count Progression]
     (random samples)      (by iteration)       (convergence to best)
```

**Row 3: Inlier Analysis**
```
[Inlier Distribution]    [Outlier Patterns]    [Residual by Group]
     (space coverage)        (systematic?)         (inliers vs outliers)
```

**Row 4: Robustness**
```
[Outlier Percentage vs Accuracy]    [Breakdown Point]    [Comparison with OLS]
     (robustness curve)                  (maximum tolerance)   (with/without outliers)
```

#### Model Details Panel
- **Parameters**:
  - base_estimator: [type]
  - min_samples: X (minimum for fit)
  - residual_threshold: X.XXX
  - max_trials: X
  - max_skips: X
  - stop_n_inliers: X
  - stop_score: X.XXX
  - stop_probability: X.XX

- **RANSAC Results**:
  | Iteration | Inliers | Score | Model Params |
  |-----------|---------|-------|-------------|
  | Best      | N       | X.XXX | {...}       |

- **Inlier/Outlier Split**:
  - Inliers: N (X%) - used for final model
  - Outliers: N (X%) - rejected
  - Random samples tried: X

#### Special Features
- **Outlier Visualizer**: See which points were rejected
- **Iteration Playback**: Watch RANSAC converge
- **Threshold Tuner**: Optimize residual threshold
- **Robustness Test**: Add outliers and measure impact
- **Comparison Mode**: RANSAC vs regular regression

---

#### SVR (Support Vector Regressor)

#### Header
- Model: Support Vector Regressor
- Training time: X seconds
- Samples: N train / M test
- Kernel: [linear/poly/rbf/sigmoid]
- Support vectors: N (X% of training data)

#### Key Metrics (Top Row Cards)
- **R² Score**: X.XXX
- **RMSE**: X.XXX
- **Epsilon (tube width)**: X.XXX
- **Support Vectors**: N (X%)
- **C (regularization)**: X.XXX

#### Visualizations

**Row 1: Prediction Quality**
```
[Actual vs Predicted]    [Residuals]    [Epsilon-Tube Violations]
     (with tube)            (analysis)      (points outside epsilon)
```

**Row 2: Support Vector Analysis**
```
[Support Vector Distribution]    [SV by Position]    [Margin Violations]
     (in feature space)           (on/off boundary)    (outside epsilon)
```

**Row 3: Kernel Analysis**
```
[Kernel Matrix]    [Feature Space Projection]    [Decision Surface (2D)]
```

**Row 4: Epsilon-Sensitivity**
```
[Epsilon vs Performance]    [Trade-off Curve]    [Optimal Epsilon]
     (tube width effect)        (accuracy vs SVs)     (cross-validated)
```

#### Model Details Panel
- **Parameters**:
  - kernel: [type]
  - degree: X (for poly)
  - gamma: X.XXX
  - coef0: X.XXX
  - tol: X.XXX
  - C: X.XXX
  - epsilon: X.XXX
  - shrinking: True/False

- **Support Vector Statistics**:
  - Total SVs: N (X%)
  - Bounded SVs: N (on margin)
  - Free SVs: N (inside tube)
  - Bias: X.XXX

#### Special Features
- **Epsilon-Tube Visualizer**: Show insensitivity region
- **Support Vector Explorer**: Inspect SV samples
- **Kernel Comparison**: Compare RBF, linear, poly
- **C Parameter Analysis**: Regularization effect
- **Prediction Bounds**: Confidence from SVs

---

#### KNeighborsRegressor

#### Header
- Model: K-Nearest Neighbors Regressor
- Training time: X seconds (instant)
- Prediction time: X seconds
- Samples: N train / M test
- K neighbors: N

#### Key Metrics (Top Row Cards)
- **R² Score**: X.XXX
- **RMSE**: X.XXX
- **K Value**: N
- **Weights**: uniform/distance
- **Algorithm**: [auto/ball_tree/kd_tree/brute]

#### Visualizations

**Row 1: Prediction Quality**
```
[Actual vs Predicted]    [Residuals]    [Residuals by Distance]
```

**Row 2: K Optimization**
```
[R² vs K Curve]    [RMSE vs K]    [Cross-Validation Surface]
     (elbow method)    (optimal K)     (different metrics)
```

**Row 3: Local Structure**
```
[Distance to K-th Neighbor]    [Neighbor Values]    [Local Variance]
     (density estimation)          (distribution)        (smoothness)
```

**Row 4: Prediction Confidence**
```
[Neighbor Agreement]    [Distance-Weighted Confidence]    [Uncertainty Map]
     (variance in K)        (closer = more weight)          (spatial)
```

#### Model Details Panel
- **Parameters**: (same as classifier)
- **Algorithm Info**:
  - Algorithm used: [actual]
  - Leaf size: X
  - Metric: [name]
  - Tree build time: X.XXs
  - Query time per sample: X.XXms

- **Neighbor Statistics**:
  | Sample | True | Predicted | K-NN Values | Distance | Confidence |
  |--------|------|-----------|-------------|----------|-----------|

#### Special Features
- **K Selector**: Interactive optimization
- **Neighbor Explorer**: Click to see K nearest
- **Distance Metric Comparison**: Try different metrics
- **Local Outlier Detection**: Identify sparse regions
- **Weighted vs Uniform**: Compare weighting schemes

---

#### MLPRegressor

#### Header
- Model: Multi-Layer Perceptron Regressor
- Training time: X seconds
- Samples: N train / M test
- Architecture: [layers]
- Training iterations: N

#### Key Metrics (Top Row Cards)
- **R² Score**: X.XXX
- **RMSE**: X.XXX
- **Final Loss**: X.XXX
- **Iterations**: N
- **Parameters**: N (weights + biases)

#### Visualizations

**Row 1: Prediction Quality**
```
[Actual vs Predicted]    [Residuals]    [Residuals by Magnitude]
```

**Row 2: Training Dynamics**
```
[Loss Curve]    [Validation Curve]    [Gradient Norms]
     (train/val)      (R² over iterations)   (check stability)
```

**Row 3: Network Analysis**
```
[Architecture Diagram]    [Weight Distribution]    [Activation Patterns]
```

**Row 4: Error Analysis**
```
[Error by Feature]    [Prediction Intervals]    [Calibration Check]
     (which features cause errors)   (bootstrap)   (uncertainty)
```

#### Model Details Panel
- **Parameters**: (same as classifier)
- **Architecture Summary**:
  | Layer | Units | Parameters | Activation |
  |-------|-------|-----------|-----------|

- **Training History**:
  | Iteration | Train Loss | Val Loss | R² Train | R² Val |
  |-----------|-----------|----------|----------|--------|

#### Special Features
- **Architecture Tuner**: Interactive layer design
- **Training Animation**: Watch convergence
- **Early Stopping**: Prevent overfitting
- **Learning Rate Finder**: Optimal LR scheduling
- **Regularization Monitor**: Weight decay tracking

---

## Design Principles

1. **Progressive Disclosure**: Show overview first, details on demand
2. **Action-Oriented**: Every insight should suggest an action
3. **Contextual Help**: Tooltips explain metrics and visualizations
4. **Consistent Layout**: Same structure across all models
5. **Performance First**: Lazy-load heavy visualizations
6. **Mobile-Friendly**: Responsive design for all screen sizes
7. **Accessibility**: Screen reader support, keyboard navigation
8. **Export Everything**: Any visualization can be saved/downloaded

---

## Technical Implementation Notes

- Use Plotly for all interactive charts
- Implement lazy loading for heavy computations (SHAP, partial dependence)
- Cache model results to avoid recomputation
- Use web workers for background calculations
- Implement progressive loading (skeleton screens)
- Store visualizations as JSON for reproducibility

---

## Technical Implementation Specifications

### Component Architecture

```
ModelResultsPage
├── HeaderComponent
│   ├── ModelInfoBadge
│   ├── DatasetInfo
│   └── Timestamp
├── MetricsGrid (6 cards)
│   ├── PrimaryMetrics
│   └── SecondaryMetrics
├── VisualizationContainer
│   ├── Row1 (3 charts)
│   ├── Row2 (3 charts)
│   ├── Row3 (3 charts)
│   └── Row4 (3 charts)
├── DetailsPanel (Collapsible)
│   ├── ParametersSection
│   ├── StatisticsSection
│   └── TablesSection
└── ActionBar
    ├── SaveButton
    ├── ExportDropdown
    ├── CompareButton
    └── PredictButton
```

### Data Structure

```python
ModelResultsState = {
    "model_id": str,
    "model_type": str,  # e.g., "DecisionTreeClassifier"
    "timestamp": datetime,
    "training_info": {
        "duration_seconds": float,
        "n_samples_train": int,
        "n_samples_test": int,
        "n_features": int,
        "target_column": str,
        "classes": List[str]  # for classification
    },
    "metrics": {
        "primary": Dict[str, float],  # e.g., {"accuracy": 0.95}
        "secondary": Dict[str, float],
        "per_class": Dict[str, Dict]  # classification only
    },
    "parameters": Dict[str, Any],
    "visualizations": {
        "charts": List[ChartSpec],
        "tables": List[TableSpec]
    },
    "model_object": bytes,  # pickled model
    "feature_importance": Dict[str, float],
    "warnings": List[str]
}
```

### Chart Specifications

**Plotly Chart Types:**
- **Confusion Matrix**: `plotly.graph_objects.Heatmap`
- **ROC Curves**: `plotly.graph_objects.Scatter` with multiple traces
- **Feature Importance**: `plotly.graph_objects.Bar` (horizontal)
- **Tree Visualization**: Custom D3.js or `plotly.graph_objects.Treemap`
- **Residual Plots**: `plotly.graph_objects.Scatter` with marginal histograms
- **Learning Curves**: `plotly.graph_objects.Scatter` with error bands
- **Partial Dependence**: `plotly.graph_objects.Scatter` with fill

**Chart Configuration:**
```python
ChartSpec = {
    "type": str,  # "heatmap", "scatter", "bar", etc.
    "data": Dict,
    "layout": Dict,
    "config": {
        "responsive": True,
        "displayModeBar": True,
        "toImageButtonOptions": {...}
    }
}
```

### Model-Specific Data Requirements

**Tree Models:**
```python
TreeModelData = {
    "tree_structure": {
        "nodes": List[Node],
        "edges": List[Edge],
        "depth": int,
        "n_leaves": int
    },
    "node_info": Dict[int, Dict],  # node_id -> stats
    "feature_importance": Dict[str, float],
    "decision_paths": List[Path]
}
```

**Linear Models:**
```python
LinearModelData = {
    "coefficients": Dict[str, float],
    "standard_errors": Dict[str, float],
    "p_values": Dict[str, float],
    "confidence_intervals": Dict[str, Tuple],
    "diagnostics": {
        "residuals": List[float],
        "fitted_values": List[float],
        "leverage": List[float],
        "cooks_distance": List[float]
    }
}
```

**Ensemble Models:**
```python
EnsembleModelData = {
    "estimators": List[EstimatorInfo],
    "oob_scores": List[float],
    "feature_importance": {
        "mean": Dict[str, float],
        "std": Dict[str, float],
        "per_estimator": List[Dict]
    },
    "tree_statistics": {
        "depths": List[int],
        "n_leaves": List[int],
        "samples_per_leaf": List[List[int]]
    }
}
```

### Performance Optimization

**Lazy Loading:**
```python
# Heavy computations only when needed
def load_heavy_visualization(chart_id: str):
    if chart_id not in cache:
        cache[chart_id] = compute_expensive_chart()
    return cache[chart_id]
```

**Caching Strategy:**
- Cache model predictions: TTL = session duration
- Cache visualizations: TTL = 1 hour
- Cache SHAP values: TTL = session duration
- Cache permutation importance: TTL = 1 hour

**Progressive Loading:**
1. Show skeleton screens immediately
2. Load primary metrics (fast)
3. Load basic charts (medium)
4. Load advanced visualizations (lazy)

### Export Formats

**Python Script:**
```python
def export_python_script(model_results: ModelResultsState) -> str:
    template = """
import pandas as pd
from sklearn.{module} import {model_class}

# Load data
df = pd.read_csv('{dataset_path}')
X = df[{features}]
y = df['{target}']

# Train model
model = {model_class}({parameters})
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
"""
    return template.format(...)
```

**BigQuery SQL:**
```sql
-- Decision Tree as SQL CASE statements
SELECT
  CASE
    WHEN feature1 > threshold1 THEN
      CASE
        WHEN feature2 > threshold2 THEN 'Class A'
        ELSE 'Class B'
      END
    ELSE 'Class C'
  END as prediction
FROM dataset
```

### Responsive Design

**Breakpoints:**
- Mobile: < 768px (1 chart per row)
- Tablet: 768px - 1024px (2 charts per row)
- Desktop: > 1024px (3 charts per row)

**Collapsible Panels:**
- Details panel: collapsible on mobile
- Action bar: hamburger menu on mobile
- Charts: swipe navigation on mobile

### Accessibility

**ARIA Labels:**
- All charts: `aria-label="Chart showing {description}"`
- Buttons: `aria-label="{action} model"`
- Tables: `aria-label="{type} data table"`

**Keyboard Navigation:**
- Tab through metrics
- Arrow keys navigate charts
- Enter to expand/collapse
- Escape to close modals

**Screen Readers:**
- Charts: text alternatives with data tables
- Tables: proper header associations
- Alerts: live regions for warnings

### State Management

**URL Parameters:**
```
/model/{model_id}?tab=visualizations&chart=feature_importance
```

**Session Storage:**
```python
# Persist page state
session_state = {
    "expanded_panels": List[str],
    "selected_chart": str,
    "zoom_level": float,
    "scroll_position": int
}
```

### Error Handling

**Model Training Errors:**
```python
class ModelTrainingError(Exception):
    def to_ui_message(self) -> Dict:
        return {
            "type": "error",
            "title": "Model Training Failed",
            "message": self.message,
            "suggestion": self.get_suggestion(),
            "action": self.get_recovery_action()
        }
```

**Visualization Errors:**
- Show placeholder with error message
- Log to console for debugging
- Retry button for transient errors

### Testing Strategy

**Unit Tests:**
- Test each metric calculation
- Test chart data generation
- Test export functions

**Integration Tests:**
- End-to-end model training + display
- Export functionality
- Comparison feature

**Visual Regression Tests:**
- Screenshot comparisons for charts
- Responsive layout testing

---

## Implementation Priority Matrix

| Model | Complexity | User Value | Priority |
|-------|-----------|-----------|----------|
| DecisionTreeClassifier | Low | High | P0 |
| RandomForestClassifier | Medium | High | P0 |
| LogisticRegression | Low | High | P0 |
| GradientBoostingClassifier | Medium | High | P1 |
| LinearRegression | Low | High | P1 |
| SVC | High | Medium | P2 |
| KNN | Low | Medium | P2 |
| MLPClassifier | High | Medium | P3 |
| Ridge/Lasso | Medium | High | P1 |

