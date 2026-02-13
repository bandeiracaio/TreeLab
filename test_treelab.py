"""Quick test script for TreeLab MVP."""

import sys

sys.path.insert(0, ".")

from treelab import TreeLab

if __name__ == "__main__":
    print("🧪 Testing TreeLab MVP...")
    print("=" * 60)

    # Create TreeLab instance with default Titanic dataset
    app = TreeLab()

    # Print state info
    state_info = app.get_state()
    print(f"\n✅ TreeLab initialized successfully!")
    print(f"   Mode: {state_info['mode']}")
    print(f"   DataFrame shape: {state_info['df_shape']}")
    print(f"   Actions executed: {state_info['num_actions']}")

    print("\n" + "=" * 60)
    print("🚀 Starting Dash server...")
    print("   Open http://127.0.0.1:8050 in your browser")
    print("=" * 60 + "\n")

    # Launch the app
    app.run(debug=True)
