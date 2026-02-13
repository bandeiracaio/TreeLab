#!/usr/bin/env python
"""Simple launcher for TreeLab."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from treelab import TreeLab

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  TREELAB - Interactive Data Exploration & Tree Modeling")
    print("=" * 70)
    print("\n  Loading Titanic dataset...")

    try:
        # Initialize TreeLab
        app = TreeLab()

        print("\n  [OK] TreeLab initialized successfully!")
        print(
            f"  [OK] Dataset loaded: {app.state_manager.df.shape[0]} rows x {app.state_manager.df.shape[1]} columns"
        )
        print("\n" + "=" * 70)
        print("  Starting Dash server...")
        print("  Open your browser to: http://127.0.0.1:8050")
        print("=" * 70 + "\n")
        print("  TIP: Press Ctrl+C to stop the server\n")

        # Run the app
        app.run(port=8050, debug=False)

    except KeyboardInterrupt:
        print("\n\n  TreeLab stopped. Goodbye!\n")
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        print("\n  See LAUNCH.md for troubleshooting\n")
        sys.exit(1)
