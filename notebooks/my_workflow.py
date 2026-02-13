# TreeLab Session Script
# Generated: 2026-02-12 23:49:48
# Session Duration: 9 seconds
# Total Actions: 0
# Mode: transformation


# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


# Load Data
# Replace the line below with your actual data loading code
# df = pd.read_csv('your_data.csv')
df = pd.read_csv('data/titanic.csv')  # Default Titanic dataset
print(f"Initial data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")


# End of TreeLab Session Script
print("\n=== Script Execution Complete ===")
