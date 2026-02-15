# 🚀 Launch TreeLab

## Three Ways to Start TreeLab

### 1. Quick Launch (Easiest)

```bash
cd TreeLab
python test_treelab.py
```

Then open **http://127.0.0.1:8050** in your browser.

---

### 2. Python Script

Create a file `run.py`:

```python
from treelab import TreeLab

app = TreeLab()  # Uses default Titanic dataset
app.run(port=8050, debug=True)
```

Run it:
```bash
python run.py
```

---

### 3. Jupyter Notebook

Open `notebooks/example_usage.ipynb` and run the cells.

Or create a new notebook:

```python
from treelab import TreeLab

app = TreeLab()
app.run()
```

---

## Using Your Own Data

```python
import pandas as pd
from treelab import TreeLab

# Load your data
df = pd.read_csv('your_data.csv')

# Launch TreeLab
app = TreeLab(df)
app.run()
```

---

## What to Do Next

Once the app launches in your browser:

### First-Time Walkthrough

1. **Look at the Data**
   - Click the "📊 Data View" tab
   - Explore your dataset

2. **Try Your First Action**
   - Select "DropColumns" from dropdown
   - Choose columns to remove
   - Click "Execute Action"

3. **Create a Checkpoint**
   - Type a name in the checkpoint input
   - Click "Save"
   - See it appear in History

4. **Complete a Full Pipeline**
   - Follow the workflow in QUICKSTART.md
   - Transform → Split → Model → Evaluate

5. **Export Your Work**
   - Click "📥 Export Python Script"
   - Download and run independently!

---

## Troubleshooting

### "Port already in use"
```python
app.run(port=8051)  # Try a different port
```

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Cannot find titanic.csv"
Make sure you're running from the `TreeLab/` directory.

### App won't load in browser
- Check the console for errors
- Make sure no firewall is blocking port 8050
- Try http://localhost:8050 instead of 127.0.0.1

---

## Need Help?

- See **QUICKSTART.md** for a detailed workflow example
- See **MVP_COMPLETE.md** for features and architecture
- Check console output for error messages

---

## 🎉 You're All Set!

TreeLab is ready to explore your data interactively.

**Enjoy!** 🧪
