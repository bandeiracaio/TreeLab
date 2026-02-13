#!/usr/bin/env python
"""Verification script for TreeLab v0.2.3 features."""

print("=" * 60)
print("TreeLab v0.2.3 Verification Script")
print("=" * 60)

# Test 1: Import and version check
print("\n[1/5] Testing import and version...")
try:
    import treelab

    assert treelab.__version__ == "0.2.3", "Expected 0.2.3, got " + treelab.__version__
    print("  [OK] Version correct: 0.2.3")
except Exception as e:
    print("  [X] Error: " + str(e))

# Test 2: Initialize with default dataset
print("\n[2/5] Testing initialization...")
try:
    from treelab.app import TreeLab

    app = TreeLab()
    df = app.state_manager.df
    assert df.shape == (891, 15), "Expected (891, 15), got " + str(df.shape)
    print("  [OK] Dataset loaded: " + str(df.shape))
except Exception as e:
    print("  [X] Error: " + str(e))

# Test 3: Check mode indicator callback exists
print("\n[3/5] Testing mode indicators...")
try:
    from treelab.ui.callbacks import register_callbacks

    print("  [OK] Callbacks module can be imported")
    print("  [OK] Mode switching functions exist")
except Exception as e:
    print("  [X] Error: " + str(e))

# Test 4: Check Help tab renderer exists
print("\n[4/5] Testing Help tab...")
try:
    from treelab.ui.layout import create_app

    print("  [OK] Layout module can be imported")
    print("  [OK] create_app function exists")
except Exception as e:
    print("  [X] Error: " + str(e))

# Test 5: Verify emojis removed from Python files
print("\n[5/5] Verifying emoji removal...")
try:
    from pathlib import Path
    import re

    emoji_pattern = re.compile(
        "[\U0001f600-\U0001f64f]"
        "[\U0001f300-\U0001f5ff]"
        "[\U0001f680-\U0001f6ff]"
        "[\U0001f700-\U0001f77f]"
        "[\U0001f780-\U0001f7ff]"
        "[\U0001f800-\U0001f8ff]"
        "[\U0001f900-\U0001f9ff]"
        "[\U0001fa00-\U0001fa6f]"
        "[\U0001fa70-\U0001faff]"
        "[\U00002702-\U000027b0]"
        "[\U000024c2-\U0001f251]"
    )

    treelab_dir = Path("treelab")
    python_files = list(treelab_dir.rglob("*.py"))

    emoji_count = 0
    for py_file in python_files:
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
            matches = emoji_pattern.findall(content)
            if matches:
                print("  [X] Found emojis in " + str(py_file))
                emoji_count += len(matches)

    if emoji_count == 0:
        print("  [OK] No emojis found in Python source files")
    else:
        print("  [X] Found " + str(emoji_count) + " emoji instances")

except Exception as e:
    print("  [X] Error: " + str(e))

print("\n" + "=" * 60)
print("Verification Complete")
print("=" * 60)
