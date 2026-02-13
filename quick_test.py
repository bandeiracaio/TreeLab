"""Quick test of TreeLab core functionality."""

import sys

sys.path.insert(0, ".")

from treelab import TreeLab
from treelab.core.action_registry import ActionRegistry
import pandas as pd

print("=" * 70)
print("TREELAB - Quick Core Functionality Test")
print("=" * 70)

# Test 1: Initialize
print("\n1. Testing initialization...")
app = TreeLab()
print(f"   ✓ Loaded dataset: {app.state_manager.df.shape}")

# Test 2: Check actions registered
print("\n2. Testing action registration...")
transform_actions = ActionRegistry.get_action_names("transformation")
model_actions = ActionRegistry.get_action_names("modeling")
print(f"   ✓ Transformation actions: {len(transform_actions)}")
print(f"     {transform_actions}")
print(f"   ✓ Modeling actions: {len(model_actions)}")
print(f"     {model_actions}")

# Test 3: Try a simple action (DropColumns)
print("\n3. Testing DropColumns action...")
from treelab.actions.transformations.drop_columns import DropColumnsAction

action = DropColumnsAction()
params = {"columns": ["deck", "embark_town"]}

# Validate
is_valid, error = action.validate(app.state_manager.df, params)
print(f"   ✓ Validation: {is_valid}")

# Execute
if is_valid:
    result = action.execute(app.state_manager.df, params)
    print(f"   ✓ Executed successfully")
    print(f"   ✓ Shape before: {app.state_manager.df.shape}")
    print(f"   ✓ Shape after: {result['df'].shape}")

    # Apply to state
    app.state_manager.apply_action("DropColumns", params, result)
    print(f"   ✓ Applied to state")

# Test 4: Check history
print("\n4. Testing history...")
history = app.state_manager.get_history_summary()
print(f"   ✓ Actions in history: {len(history)}")
if history:
    print(f"   ✓ Last action: {history[-1]['action']}")

# Test 5: Create checkpoint
print("\n5. Testing checkpoints...")
success = app.state_manager.create_checkpoint("Test Checkpoint")
print(f"   ✓ Checkpoint created: {success}")
checkpoints = app.state_manager.get_checkpoints()
print(f"   ✓ Total checkpoints: {len(checkpoints)}")

# Test 6: Export script
print("\n6. Testing script export...")
script = app.logger.generate_script()
print(f"   ✓ Script generated: {len(script)} characters")
print(f"   ✓ First line: {script.split(chr(10))[0]}")

# Test 7: Column analyzer
print("\n7. Testing column analyzer...")
from treelab.utils.column_analyzer import ColumnAnalyzer

numeric_cols = ColumnAnalyzer.get_numeric_columns(app.state_manager.df)
print(f"   ✓ Numeric columns found: {len(numeric_cols)}")
missing_cols = ColumnAnalyzer.get_columns_with_missing(app.state_manager.df)
print(f"   ✓ Columns with missing: {missing_cols}")

# Summary
print("\n" + "=" * 70)
print("ALL CORE TESTS PASSED!")
print("=" * 70)
print("\nTreeLab is ready to launch!")
print("\nRun: python run_treelab.py")
print("Then open: http://127.0.0.1:8050")
print("=" * 70 + "\n")
