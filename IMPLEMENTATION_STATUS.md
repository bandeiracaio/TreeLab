# TreeLab - Feature Implementation Status

## Overview

This document lists all available scikit-learn features and indicates which have been implemented in TreeLab.

---

## ✅ Already Implemented

### Transformations

| Feature | Module | Status |
|---------|--------|--------|
| Drop Columns | sklearn | ✅ Implemented |
| Simple Imputer | sklearn.impute | ✅ Implemented |
| Standard Scaler | sklearn.preprocessing | ✅ Implemented |
| MinMax Scaler | sklearn.preprocessing | ✅ Implemented |
| One Hot Encoder | sklearn.preprocessing | ✅ Implemented |
| Label Encoder | sklearn.preprocessing | ✅ Implemented |
| Train/Test Split | sklearn.model_selection | ✅ Implemented |
| PCA | sklearn.decomposition | ✅ Implemented |
| Polynomial Features | sklearn.preprocessing | ✅ Implemented |
| RFE (Recursive Feature Elimination) | sklearn.feature_selection | ✅ Implemented |

### Models

| Feature | Module | Status |
|---------|--------|--------|
| Decision Tree Classifier | sklearn.tree | ✅ Implemented |
| Random Forest Classifier | sklearn.ensemble | ✅ Implemented |
| Decision Tree Regressor | sklearn.tree | ✅ Implemented |
| Random Forest Regressor | sklearn.ensemble | ✅ Implemented |
| Feature Importance | sklearn | ✅ Implemented |
| Hyperparameter Tuning | sklearn.model_selection | ✅ Implemented |
| SHAP Analysis | shap | ✅ Implemented |
| Binning Scorecard | custom | ✅ Implemented |

### Visualizations

| Feature | Status |
|---------|--------|
| Data Table View | ✅ |
| Descriptive Statistics | ✅ |
| Distribution Plots (Histograms) | ✅ |
| Correlation Heatmap | ✅ |
| Confusion Matrix | ✅ |
| Feature Importance Plot | ✅ |
| Tree Visualization | ✅ |
| SHAP Summary Plot | ✅ |
| Model Comparison (Radar Chart) | ✅ |

### Utilities

| Feature | Status |
|---------|--------|
| Checkpoints | ✅ |
| Python Script Export | ✅ |
| BigQuery SQL Export | ✅ |
| Action History | ✅ |

---

## 🔲 Not Yet Implemented

### High Priority

#### Transformations

| Feature | Module | Description |
|---------|--------|-------------|
| Robust Scaler | sklearn.preprocessing | Scale features using statistics robust to outliers |
| MaxAbs Scaler | sklearn.preprocessing | Scale each feature by its maximum absolute value |
| Quantile Transformer | sklearn.preprocessing | Transform features to follow a uniform/normal distribution |
| Power Transformer | sklearn.preprocessing | Apply power transform to make data more Gaussian-like |
| Ordinal Encoder | sklearn.preprocessing | Encode categorical features as ordinal integers |
| KBins Discretizer | sklearn.preprocessing | Discretize continuous features into bins |
| Spline Transformer | sklearn.preprocessing | Generate B-spline basis features |
| Target Encoder | sklearn.preprocessing | Encode categorical features using target statistics |
| Variance Threshold | sklearn.feature_selection | Feature selection based on variance |
| SelectKBest | sklearn.feature_selection | Select k best features |
| SelectPercentile | sklearn.feature_selection | Select features based on percentile |
| SelectFpr/SelectFdr/SelectFwe | sklearn.feature_selection | Statistical feature selection |
| GenericUnivariateSelect | sklearn.feature_selection | Univariate feature selector with configurable mode |
| Polynomial Features (interaction_only) | sklearn.preprocessing | Only interaction features |

#### Models - Classifiers

| Feature | Module | Description |
|---------|--------|-------------|
| Gradient Boosting Classifier | sklearn.ensemble | Gradient boosting for classification |
| Hist Gradient Boosting Classifier | sklearn.ensemble | Histogram-based gradient boosting |
| AdaBoost Classifier | sklearn.ensemble | AdaBoost classifier |
| Bagging Classifier | sklearn.ensemble | Bagging meta-estimator |
| Extra Trees Classifier | sklearn.ensemble | Extremely randomized trees |
| Logistic Regression | sklearn.linear_model | Logistic regression classifier |
| SVM Classifier | sklearn.svm | Support vector machine classifier |
| KNN Classifier | sklearn.neighbors | K-nearest neighbors classifier |
| Naive Bayes | sklearn.naive_bayes | Gaussian/Multinomial/Bernoulli Naive Bayes |
| MLP Classifier | sklearn.neural_network | Multi-layer perceptron classifier |
| Linear Discriminant Analysis (LDA) | sklearn.discriminant_analysis | Linear discriminant analysis |
| Quadratic Discriminant Analysis (QDA) | sklearn.discriminant_analysis | Quadratic discriminant analysis |

#### Models - Regressors

| Feature | Module | Description |
|---------|--------|-------------|
| Gradient Boosting Regressor | sklearn.ensemble | Gradient boosting for regression |
| Hist Gradient Boosting Regressor | sklearn.ensemble | Histogram-based gradient boosting |
| AdaBoost Regressor | sklearn.ensemble | AdaBoost regressor |
| Bagging Regressor | sklearn.ensemble | Bagging regressor |
| Extra Trees Regressor | sklearn.ensemble | Extremely randomized trees regressor |
| Linear Regression | sklearn.linear_model | Ordinary least squares |
| Ridge Regression | sklearn.linear_model | L2-regularized linear regression |
| Lasso Regression | sklearn.linear_model | L1-regularized regression |
| ElasticNet | sklearn.linear_model | L1+L2 regularized regression |
| Bayesian Ridge | sklearn.linear_model | Bayesian ridge regression |
| Huber Regressor | sklearn.linear_model | Robust linear regression |
| Theil-Sen Regressor | sklearn.linear_model | Robust multivariate regression |
| RANSAC Regressor | sklearn.linear_model | RANSAC (RANdom SAmple Consensus) |
| SVM Regressor | sklearn.svm | Support vector machine regressor |
| KNN Regressor | sklearn.neighbors | K-nearest neighbors regressor |
| MLP Regressor | sklearn.neural_network | Multi-layer perceptron regressor |

#### Model Analysis

| Feature | Module | Description |
|---------|--------|-------------|
| Cross-Validation | sklearn.model_selection | K-fold cross-validation |
| ROC Curve | sklearn.metrics | ROC curve visualization |
| Precision-Recall Curve | sklearn.metrics | PR curve visualization |
| Learning Curve | sklearn.model_selection | Learning curve plot |
| Validation Curve | sklearn.model_selection | Validation curve plot |
| Permutation Importance | sklearn.inspection | Permutation-based feature importance |
| Partial Dependence | sklearn.inspection | Partial dependence plots |
| Calibration Curve | sklearn.calibration | Probability calibration curve |
| Prediction Error Plot | sklearn.metrics | Regression prediction errors |

---

### Medium Priority

#### Advanced Transformations

| Feature | Module | Description |
|---------|--------|-------------|
| Truncated SVD | sklearn.decomposition | Dimensionality reduction (LSA) |
| NMF | sklearn.decomposition | Non-negative matrix factorization |
| Fast ICA | sklearn.decomposition | Independent component analysis |
| Factor Analysis | sklearn.decomposition | Factor analysis |
| Kernel PCA | sklearn.decomposition | Kernel PCA |
| Incremental PCA | sklearn.decomposition | PCA for large datasets |
| Sparse PCA | sklearn.decomposition | Sparse PCA |
| Dictionary Learning | sklearn.decomposition | Dictionary learning |

#### Clustering (Unsupervised)

| Feature | Module | Description |
|---------|--------|-------------|
| K-Means | sklearn.cluster | K-means clustering |
| DBSCAN | sklearn.cluster | Density-based clustering |
| Agglomerative Clustering | sklearn.cluster | Hierarchical clustering |
| Spectral Clustering | sklearn.cluster | Spectral clustering |
| Mean Shift | sklearn.cluster | Mean shift clustering |
| Birch | sklearn.cluster | Birch clustering |

#### Outlier Detection

| Feature | Module | Description |
|---------|--------|-------------|
| Isolation Forest | sklearn.ensemble | Isolation forest outlier detector |
| One-Class SVM | sklearn.svm | One-class SVM |
| Elliptic Envelope | sklearn.covariance | Gaussian outlier detection |
| Local Outlier Factor | sklearn.neighbors | LOF outlier detection |

---

### Lower Priority

#### Feature Engineering

| Feature | Module | Description |
|---------|--------|-------------|
| Feature Hasher | sklearn.feature_extraction | Feature hashing for large datasets |
| Dict Vectorizer | sklearn.feature_extraction | Transform dicts to feature vectors |
| Text Features (TF-IDF) | sklearn.feature_extraction | Text feature extraction |
| Image Features | sklearn.feature_extraction | Image patch extraction |

#### Pipelines

| Feature | Module | Description |
|---------|--------|-------------|
| Pipeline Builder | sklearn.pipeline | Chain transformers with estimator |
| Column Transformer | sklearn.compose | Apply different transformers to different columns |

#### Model Selection

| Feature | Module | Description |
|---------|--------|-------------|
| Grid Search CV | sklearn.model_selection | Exhaustive hyperparameter search |
| Randomized Search CV | sklearn.model_selection | Random hyperparameter search |
| Halving Grid Search | sklearn.model_selection | Successive halving grid search |
| Halving Random Search | sklearn.model_selection | Successive halving random search |

#### Multiclass/Multioutput

| Feature | Description |
|---------|-------------|
| OneVsRest Classifier | One-vs-rest multiclass strategy |
| OneVsOne Classifier | One-vs-one multiclass strategy |
| Output Code Classifier | Error-correcting output codes |
| Multioutput Classifier | Multi-label classification |
| Multioutput Regressor | Multi-output regression |

---

## 📊 Implementation Recommendations

### Phase 1 - Essential Additions

1. **Logistic Regression** - Most common classifier
2. **Gradient Boosting** - Powerful ensemble method
3. **Cross-Validation** - Better model evaluation
4. **ROC/PR Curves** - Essential classification diagnostics

### Phase 2 - Common Use Cases

5. **KNN Classifier/Regressor** - Simple but effective
6. **Extra Trees** - Often better than Random Forest
7. **Learning Curves** - Diagnose bias/variance
8. **Pipeline Builder** - Workflow automation

### Phase 3 - Advanced Features

9. **SVM** - Strong classifiers for certain domains
10. **MLP** - Neural network capability
11. **Clustering** - Unsupervised learning
12. **Outlier Detection** - Anomaly detection

### Phase 4 - Full Coverage

13. Remaining transformations
14. All regression models
15. Advanced model analysis
16. Ensemble methods (stacking, voting)

---

## 📝 Notes

- Current focus is on tree-based models (as per name)
- Consider the complexity vs. benefit for each addition
- Some advanced features may require significant UI work
- Consider adding "Pipeline" concept for workflow chaining
