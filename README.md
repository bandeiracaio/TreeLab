# 🧪 TreeLab

An interactive laboratory for data transformation and tree-based machine learning built with Dash and scikit-learn.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

## Features

- **Interactive Data Exploration** - Visualize and understand your data with interactive plots
- **Data Transformations** - Drop columns, impute missing values, scale, encode, and more
- **Tree-Based Modeling** - Train and evaluate Decision Trees and Random Forests
- **Classification & Regression** - Support for both problem types
- **Model Analysis** - Feature importance, SHAP values, confusion matrix, hyperparameter tuning
- **State Management** - Create checkpoints and branch your analysis workflow
- **Code Export** - Auto-generate reproducible Python scripts and BigQuery SQL

## Installation

```bash
git clone https://github.com/bandeiracaio/TreeLab.git
cd TreeLab
pip install -r requirements.txt
```

## Quick Start

```python
from treelab import TreeLab
import pandas as pd

# Load your data (or use default Titanic dataset)
df = pd.read_csv('your_data.csv')

# Launch TreeLab
app = TreeLab(df)
app.run()

# Or use default Titanic dataset
app = TreeLab()
app.run()
```

Then open your browser to http://localhost:8050

## Running on JupyterLab/JupyterHub

### Local JupyterLab

```python
from treelab import TreeLab

app = TreeLab()
app.run(host='127.0.0.1', port=8050)
```

Access at http://127.0.0.1:8050

### JupyterHub

```python
from treelab import TreeLab

app = TreeLab()
app.run(host='0.0.0.0', port=8050)
```

Access at `{your_jupyterhub_url}/proxy/8050`

For detailed instructions, see [JUPYTERHUB_GUIDE.md](./JUPYTERHUB_GUIDE.md).

---

## Implemented Features

### Transformations

| Feature | Description |
|---------|-------------|
| Drop Columns | Remove columns from dataset |
| Simple Imputer | Fill missing values (mean, median, mode, constant) |
| Standard Scaler | Standardize features (zero mean, unit variance) |
| MinMax Scaler | Scale features to a given range |
| One Hot Encoder | One-hot encode categorical features |
| Label Encoder | Encode labels with values between 0 and n_classes-1 |
| Train/Test Split | Split data into train and test sets |
| PCA | Principal Component Analysis for dimensionality reduction |
| Polynomial Features | Generate polynomial and interaction features |
| RFE | Recursive Feature Elimination for feature selection |

### Models

| Model | Type | Description |
|-------|------|-------------|
| Decision Tree Classifier | Classification | Tree-based classifier |
| Random Forest Classifier | Classification | Ensemble of decision trees |
| Decision Tree Regressor | Regression | Tree-based regressor |
| Random Forest Regressor | Regression | Ensemble of decision trees |

### Model Analysis

| Feature | Description |
|---------|-------------|
| Feature Importance | View feature importance scores |
| SHAP Analysis | SHAP summary plots for model interpretability |
| Hyperparameter Tuning | Grid search for optimal parameters |
| Binning Scorecard | Create credit-score-style scorecards |

### Visualizations

- Data Table View with sorting and filtering
- Descriptive Statistics (mean, std, min, max, etc.)
- Distribution Plots (histograms)
- Correlation Heatmap
- Confusion Matrix
- Feature Importance Plot
- Decision Tree Visualization
- SHAP Summary Plot
- Model Comparison (radar chart)

### Utilities

- **Checkpoints** - Save named snapshots of your workflow
- **Action History** - Track all transformations and models applied
- **Python Script Export** - Download reproducible Python code
- **BigQuery SQL Export** - Generate BigQuery SQL from your workflow

---

## Workflow

### 1. Load & Explore Data

- Upload CSV or use default Titanic dataset
- View data table with sorting/filtering
- Analyze descriptive statistics
- Visualize distributions and correlations

### 2. Transform Data

- Drop columns
- Impute missing values
- Scale numeric features
- Encode categorical variables
- Apply dimensionality reduction (PCA)
- Generate polynomial features
- Perform feature selection (RFE)

### 3. Split Data

- Train/test split with configurable ratio
- Stratified sampling for classification

### 4. Train Models

- Decision Tree Classifier/Regressor
- Random Forest Classifier/Regressor

### 5. Analyze Models

- View feature importance
- SHAP analysis for interpretability
- Confusion matrix for classification
- Hyperparameter tuning via grid search
- Model comparison across multiple runs

### 6. Export

- Python script for reproducibility
- BigQuery SQL for deployment

---

## Project Structure

```
TreeLab/
├── treelab/
│   ├── actions/
│   │   ├── transformations/   # Data transformation actions
│   │   └── modeling/          # Model training actions
│   ├── core/                  # State management, logging
│   ├── ui/                    # Dash UI components
│   └── utils/                 # Utilities (column analyzer, etc.)
├── data/                      # Sample datasets
├── notebooks/                 # Example notebooks
├── tests/                     # Unit tests
├── JUPYTERHUB_GUIDE.md        # JupyterLab/JupyterHub setup
└── README.md
```

## Requirements

- Python 3.8+
- dash
- plotly
- pandas
- scikit-learn
- shap
- pandas-gbq (for BigQuery export)

See `requirements.txt` for full list.

## License

MIT
