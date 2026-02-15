# 🧪 TreeLab

An interactive laboratory for data transformation and tree-based machine learning.

## Features

- **Interactive Data Exploration**: Visualize and understand your data with interactive plots
- **Step-by-Step Transformations**: Apply preprocessing actions one at a time
- **Tree-Based Modeling**: Fit and evaluate decision trees and random forests
- **State Management**: Create checkpoints and branch your analysis workflow
- **Reproducible Code**: Auto-generate Python scripts of your entire workflow

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage

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

## Workflow

1. **Transformation Mode**: Apply preprocessing actions
   - Drop columns
   - Impute missing values
   - Scale numeric features
   - Encode categorical variables
   - Split train/test data

2. **Model Fitting Mode**: Train and evaluate tree models
   - Decision Tree Classifier
   - Random Forest Classifier
   - View feature importance
   - Analyze model performance

3. **Save Checkpoints**: Create named snapshots of your workflow
4. **Export Code**: Download Python script to reproduce your analysis

## MVP Features (v0.1)

### Transformations
- DropColumns
- SimpleImputer
- StandardScaler
- OneHotEncoder
- TrainTestSplit

### Models
- DecisionTreeClassifier
- RandomForestClassifier

### Visualizations
- Data table view
- Descriptive statistics
- Distribution plots
- Correlation heatmap
- Model evaluation dashboard

## Roadmap

- [ ] Additional transformations (PCA, RFE, PolynomialFeatures)
- [ ] More tree models (GradientBoosting, ExtraTrees)
- [ ] Regression support
- [ ] Hyperparameter tuning UI
- [ ] SHAP integration
- [ ] Model comparison dashboard

## License

MIT
