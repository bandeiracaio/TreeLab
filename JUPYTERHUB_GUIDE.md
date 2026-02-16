# Running TreeLab on JupyterLab/JupyterHub

This guide covers running TreeLab in JupyterLab (local) and JupyterHub (shared server) environments.

---

## Table of Contents

1. [Local JupyterLab](#1-local-jupyterlab)
2. [JupyterHub](#2-jupyterhub)
3. [Using Your Own Data](#3-using-your-own-data)
4. [Troubleshooting](#4-troubleshooting)

---

## 1. Local JupyterLab

### Method A: From Terminal (Recommended for Local)

If you're running JupyterLab locally on your machine:

```bash
# Activate your environment
conda activate treelab   # or source venv/bin/activate

# Start JupyterLab
jupyter lab
```

Then in a new terminal, run TreeLab:

```python
# In a Jupyter notebook cell
from treelab import TreeLab

app = TreeLab()
app.run(host='127.0.0.1', port=8050)
```

Open **http://127.0.0.1:8050** in your browser.

### Method B: Directly in Jupyter Notebook

Create a new notebook and run:

```python
# Install dependencies (first time only)
# !pip install -r requirements.txt

from treelab import TreeLab

app = TreeLab()
app.run(host='127.0.0.1', port=8050)
```

**Important**: For local JupyterLab, you must use `host='127.0.0.1'` (not `0.0.0.0`).

---

## 2. JupyterHub

JupyterHub is a multi-user server - you need to configure the app to bind to all interfaces and use the proxy URL.

### Step 1: Identify Your JupyterHub Proxy URL

When you're logged into JupyterHub, look at your browser's address bar. It typically looks like:

```
https://jupyterhub.yourorg.edu/user/yourusername/
```

This is your **base URL**.

### Step 2: Run TreeLab with Correct Configuration

In a Jupyter notebook cell:

```python
from treelab import TreeLab

app = TreeLab()
app.run(host='0.0.0.0', port=8050)
```

### Step 3: Access the App

The app will start but won't open automatically. You need to construct the proxy URL:

```
# Format
{base_url}proxy/8050

# Example
https://jupyterhub.yourorg.edu/user/johndoe/proxy/8050
```

**Note**: Some JupyterHub deployments use different proxy paths. Try these variations:

- `/user/username/proxy/8050`
- `/proxy/8050`
- `/-/proxy/8050`

---

## 3. Using Your Own Data

### Option A: Upload CSV to JupyterHub

1. Upload your CSV file using the JupyterLab file browser
2. Load it in your notebook:

```python
import pandas as pd
from treelab import TreeLab

df = pd.read_csv('your_data.csv')
app = TreeLab(df)
app.run(host='0.0.0.0', port=8050)
```

### Option B: Load from URL

```python
import pandas as pd
from treelab import TreeLab

url = 'https://example.com/your_data.csv'
df = pd.read_csv(url)
app = TreeLab(df)
app.run(host='0.0.0.0', port=8050)
```

### Option C: Sample Your Data (for large datasets)

```python
import pandas as pd
from treelab import TreeLab

df = pd.read_csv('large_data.csv')

# Use 10% sample for faster loading
app = TreeLab(df, sample_frac=0.1)
app.run(host='0.0.0.0', port=8050)
```

---

## 4. Troubleshooting

### Issue: "Port already in use"

```python
# Use a different port
app.run(host='0.0.0.0', port=8051)
```

### Issue: "This site can't be reached"

1. Check that TreeLab is actually running (look for output in your notebook)
2. Verify the proxy URL format
3. Try different port numbers

### Issue: App loads but is slow or unresponsive

```python
# Use a smaller sample of your data
app = TreeLab(df, sample_frac=0.1)  # 10% sample
app.run(host='0.0.0.0', port=8050)
```

### Issue: "0.0.0.0" vs "127.0.0.1" confusion

| Environment | Host Value | Access URL |
|-------------|------------|------------|
| Local JupyterLab | `127.0.0.1` | http://127.0.0.1:8050 |
| JupyterHub | `0.0.0.0` | `{base_url}/proxy/8050` |
| Local terminal | `127.0.0.1` or `0.0.0.0` | http://localhost:8050 |

### Issue: Multiple Users on Same JupyterHub

Each user needs a unique port. Coordinate with your team or use:

```python
import random
port = random.randint(8050, 8090)

app = TreeLab()
app.run(host='0.0.0.0', port=port)

print(f"Access TreeLab at: {YOUR_BASE_URL}/proxy/{port}")
```

### Issue: Widgets Not Displaying

Ensure you're using a compatible Jupyter environment:

```bash
# Check versions
jupyter --version
pip list | grep -E "jupyter|dash|plotly"
```

---

## Quick Reference

### Minimal Setup Code

```python
# For JupyterHub
from treelab import TreeLab
app = TreeLab()
app.run(host='0.0.0.0', port=8050)
# Access: {your_jupyterhub_url}/proxy/8050
```

```python
# For Local JupyterLab
from treelab import TreeLab
app = TreeLab()
app.run(host='127.0.0.1', port=8050)
# Access: http://127.0.0.1:8050
```

### Common JupyterHub Base URLs

- GitHub Codespaces: `https://username-xxxxxx.apps.github.dev`
- Google Colab: N/A (use ngrok method)
- AWS SageMaker: `https://jupyterlab-username.xxx.sagemaker.aws`
- Azure ML: `https://username.experiments.azureml.net/lab`
- Custom JupyterHub: Check your browser's base URL

---

## Security Notes

1. **Don't expose to public internet** - TreeLab is designed for local/intranet use
2. **Don't run as root** in containers
3. **Session data is temporary** - refresh clears all state
4. **Export your work** - use the "Export Python Script" feature to save workflows

---

For more help, see:
- [LAUNCH.md](./LAUNCH.md) - General launch instructions
- [QUICKSTART.md](./QUICKSTART.md) - Workflow walkthrough
- [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md) - Feature list
