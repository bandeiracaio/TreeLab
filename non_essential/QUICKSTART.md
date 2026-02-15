# TreeLab Quick Start Guide

## Installation

```bash
cd TreeLab
pip install -r requirements.txt
```

## Launch TreeLab

### Option 1: With Default Titanic Dataset

```python
from treelab import TreeLab

app = TreeLab()
app.run()
```

Then open http://127.0.0.1:8050 in your browser.

### Option 2: With Your Own Dataset

```python
import pandas as pd
from treelab import TreeLab

df = pd.read_csv('your_data.csv')
app = TreeLab(df)
app.run()
```

## Example Workflow with Titanic Dataset

### 1. Start the App
```python
from treelab import TreeLab

app = TreeLab()  # Loads Titanic dataset by default
app.run()
```

### 2. Transformation Mode (Preprocessing)

1. **Drop Unnecessary Columns**
   - Action: `DropColumns`
   - Columns: Select `passenger_id`, `name`, `ticket`, `cabin`
   - Click "Execute Action"

2. **Handle Missing Values**
   - Action: `SimpleImputer`
   - Columns: Select `age` (auto-suggested)
   - Strategy: `median`
   - Click "Execute Action"

3. **Encode Categorical Variables**
   - Action: `OneHotEncoder`
   - Columns: Select `sex`, `embarked` (auto-suggested)
   - Drop First: ✓ (checked)
   - Click "Execute Action"

4. **Scale Numeric Features**
   - Action: `StandardScaler`
   - Columns: Select `age`, `fare` (auto-suggested)
   - Click "Execute Action"

5. **Create Checkpoint**
   - Checkpoint name: "After Preprocessing"
   - Click "Save"

6. **Split Train/Test**
   - Action: `TrainTestSplit`
   - Target Column: `survived`
   - Test Size: `0.2`
   - Random State: `42`
   - Click "Execute Action"

7. **Create Checkpoint**
   - Checkpoint name: "Ready for Modeling"
   - Click "Save"

### 3. Switch to Modeling Mode

Click the "Modeling" button at the top right.

### 4. Fit Models

1. **Decision Tree**
   - Action: `DecisionTreeClassifier`
   - Max Depth: `5`
   - Click "Execute Action"
   - View results in "Model Results" tab

2. **Random Forest**
   - Action: `RandomForestClassifier`
   - Number of Trees: `100`
   - Max Depth: `10`
   - Click "Execute Action"
   - Compare results!

### 5. Export Python Script

Click "Export Python Script" button to download a reproducible Python file of your entire workflow.

## Tabs Overview

### 📊 Data View
- Interactive data table
- Search and filter columns
- Pagination for large datasets

### 📈 Statistics
- Descriptive statistics (mean, std, min, max, etc.)
- Missing value analysis

### 📉 Distributions
- Histograms for numeric columns
- Select column to visualize

### 🔥 Correlations
- Correlation heatmap
- Only shows numeric columns

### 🎯 Model Results (Modeling Mode Only)
- Performance metrics (accuracy, precision, recall, F1)
- Confusion matrix
- Feature importance chart

## Tips

1. **Smart Suggestions**: TreeLab automatically suggests relevant columns for each action based on data types and characteristics.

2. **Checkpoints**: Create checkpoints at key stages to easily revert and try different approaches.

3. **History**: All actions are recorded. View them in the History panel.

4. **Reproducibility**: Export your workflow as a Python script that can be run independently.

5. **Validation**: TreeLab validates parameters before execution to prevent errors.

## MVP Features

### Transformation Actions (5)
- DropColumns
- SimpleImputer
- StandardScaler  
- OneHotEncoder
- TrainTestSplit

### Tree Models (2)
- DecisionTreeClassifier
- RandomForestClassifier

## Coming Soon

- More transformations (PCA, RFE, PolynomialFeatures)
- More tree models (GradientBoosting, ExtraTrees)
- Regression support
- Hyperparameter tuning UI
- SHAP integration
- Model comparison dashboard

## Troubleshooting

### Port Already in Use
```python
app.run(port=8051)  # Use a different port
```

### Dataset Not Loading
Make sure `titanic.csv` exists in the `data/` folder, or provide your own DataFrame.

### Action Fails
Check the error message in the red alert box. Common issues:
- Need to do train/test split before modeling
- Selected wrong column types (e.g., trying to scale categorical columns)
- Missing values in columns being processed

## Need Help?

Check the README.md for more details or open an issue on GitHub.
