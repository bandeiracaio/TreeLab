# TreeLab Changelog

All notable changes to TreeLab will be documented in this file.

## [v0.2.1] - 2026-02-13

### Added
- Multi-column distribution grid showing all numeric columns simultaneously
- ASCII art banner displayed in upper-left header
- Multiple correlation methods (Pearson, Spearman, Kendall) with radio selector
- Correlation statistics panel showing strong/moderate positive/negative pairs
- Hidden column selector component to prevent callback errors

### Fixed
- Column distribution button navigation error (dist-column-selector not found)
- Incorrect bracket closure in render_stats_tab function
- Removed unused update_distribution_graph callback
- Removed filter row from data table

### Changed
- Distribution plots now show ALL numeric columns in a grid layout
- Data table no longer shows filter row for cleaner interface
- Correlations tab now supports three different correlation methods with descriptions

## [v0.2.3] - 2026-02-13

### Fixed
- SHAP summary rendering for multiclass outputs

## [v0.2.2] - 2026-02-13

### Added
- Regression tree models with metrics and plots
- Hyperparameter tuning action (GridSearchCV)
- SHAP summary action and visualization
- Tree visualization in Model Results
- Distribution column selector and correlation pair listing

## [v0.2.0] - 2026-02-13

### Added
- Help tab with comprehensive action and tab documentation
- Column header buttons to jump to distribution view
- ASCII art instead of emojis throughout codebase
- Enhanced plot information density and styling
- Version tracking system (VERSION.txt and __version__)

### Fixed
- Tab navigation bug (Distributions button showing Correlations tab)
- Distribution dropdown now properly updates chart
- Removed all Unicode emojis causing Windows console errors

### Changed
- All emojis replaced with ASCII equivalents
- Plots now show more statistical information
- Better color schemes for visualizations
- Improved information density in charts

## [v0.1.0] - 2026-02-13

### Initial Release
- Core infrastructure (StateManager, ActionRegistry, SessionLogger)
- 5 transformation actions (DropColumns, SimpleImputer, StandardScaler, OneHotEncoder, TrainTestSplit)
- 2 tree models (DecisionTreeClassifier, RandomForestClassifier)
- Interactive Dash web UI with 5 tabs
- Smart column suggestions
- Checkpoint system
- Python script export
- Default Titanic dataset
- Comprehensive documentation
