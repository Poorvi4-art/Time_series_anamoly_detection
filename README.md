**NASA Bearing Anomaly Detection Project**

## Quick Start Guide

This project analyzes vibration data from NASA bearing experiments to detect early signs of equipment failure.

---

## What You Need

### Software Requirements
- Python 3.7 or higher
- Jupyter Notebook (for running the analysis)

### Python Libraries
```
pip install pandas numpy scikit-learn matplotlib
```

### Data Structure
Place your bearing snapshot files in this folder structure:
```
archive/
├── 1st test/
│   ├── 2003.10.23.00.04.13
│   ├── 2003.10.23.00.14.13
│   ├── 2003.10.23.00.24.13
│   └── ... (many more files)
├── 2nd test/
│   └── ... (snapshot files)
└── 3rd test/
    └── ... (snapshot files)
```

Each file contains 20,000 vibration readings sampled at 20 kHz for 1 second.

---

## How to Run

### Step 1: Setup
```bash
# Navigate to project folder
cd your_project_folder

# Install required libraries (one-time only)
pip install pandas numpy scikit-learn matplotlib
```

### Step 2: Copy the Code
Copy the code from `bearing_anomaly_detection.py` into a Jupyter notebook, or run it as a Python script:
```bash
python bearing_anomaly_detection.py
```

### Step 3: Adjust the Path (If Needed)
In the code, find this line:
```python
folder_path = 'archive/1st test'
```
Change it to match where your snapshot files are located.

### Step 4: Run and Analyze
Execute each cell or the entire script. The program will:
1. Read all snapshot files
2. Extract 4 summary statistics per file
3. Create visualizations
4. Detect anomalies
5. Save results to `bearing_analysis_results.csv`

---

## Understanding the Output

### What Each Visualization Means

**1. Mean Vibration Over Time**
- Flat trend = healthy bearing
- Rising trend = bearing degrading
- Spikes = potential problems

**2. Variability (Std Deviation) Over Time**
- Low and stable = smooth operation
- Increasing = unpredictable movement (sign of wear)

**3. Feature Distributions**
- Shows what values are "normal" for this bearing
- Helps understand data range

**4. Anomaly Markers (Red X's)**
- Each red X marks a snapshot flagged as unusual
- Clustered near the end = model detected degradation before failure

### Key Metrics

| Metric | What It Means |
|--------|---------------|
| Mean Amplitude | Average vibration intensity |
| Std Deviation | How erratic the vibration is |
| Anomalies | Snapshots with unusual patterns |
| Time to First Anomaly | How long until degradation starts |

---

## Interpreting Results

### Good Results
- Mean and std dev remain stable over time 
- Few scattered anomalies in early periods
- Cluster of anomalies appears near known failure date
- Smooth transition from "normal" to "anomalous" periods

### Concerning Results
- Random anomalies throughout (could indicate sensor noise)
- No clear trend in mean or variability
- Anomalies don't correlate with known failure time

### What to Do With Results

1. **For Maintenance Teams:**
   - Schedule inspection when anomalies start clustering
   - Replace parts before anomalies become frequent
   - Track bearing condition over time

2. **For Data Scientists:**
   - Validate predictions against actual failure times
   - Tune model parameters (contamination rate) for your equipment
   - Add more features if accuracy is insufficient

3. **For Management:**
   - Shift from reactive (emergency repairs) to predictive maintenance
   - Estimate maintenance costs and downtime prevention
   - Plan equipment replacement schedules

---

## Customization Options

### Change Contamination Rate (How Many Anomalies to Flag)
```python
iso_forest = IsolationForest(contamination=0.05)  # flags 5% as anomalies
# Try 0.01 (1%) for stricter detection, 0.10 (10%) for looser
```

### Analyze Different Test
```python
folder_path = 'archive/2nd test'  # or '3rd test'
```

### Use Different Features
Currently uses: mean, std_dev, min, max
You could add: range (max-min), percentiles, median, etc.

### Export Results to Different Format
```python
df_features.to_csv('results.csv')      # CSV file
df_features.to_excel('results.xlsx')   # Excel file
df_features.to_json('results.json')    # JSON file
```

---

## Troubleshooting

### Problem: "File not found" error
**Solution:** Check your `folder_path` variable. Make sure:
- Path is correct
- Files actually exist at that location
- Use forward slashes (/) or raw strings (r'...') for paths

### Problem: "Not enough files" or very few results
**Solution:** 
- Ensure all snapshot files are in the correct folder
- Check file naming format (should be like `2003.10.23.00.04.13`)
- Some files might be corrupted; check with `head` command first

### Problem: No anomalies detected
**Solution:**
- Increase contamination rate: `contamination=0.10` instead of 0.05
- Check if bearing ran cleanly without degradation
- Visualizations still show trends, which is informative

### Problem: Code runs but visualizations don't show
**Solution:**
- Make sure you have matplotlib installed: `pip install matplotlib`
- In Jupyter, add `%matplotlib inline` at the beginning
- In scripts, ensure `plt.show()` is called

---

## What's Happening Behind the Scenes

### Data Flow:
```
Raw Files (20,000 numbers each)
    ↓
Extract 4 Summary Statistics per file
    ↓
Create Time Series (4 numbers × hundreds of files)
    ↓
Normalize Features (scale to 0-1 range)
    ↓
Train Isolation Forest (learn what's normal)
    ↓
Detect Anomalies (flag unusual snapshots)
    ↓
Visualize Results (plots + table)
    ↓
Export Results (CSV file for further analysis)
```

### Why This Approach?
- **Simple:** Don't need labeled failure data
- **Fast:** Processes hundreds of files quickly
- **Interpretable:** Can see exactly which snapshots are flagged
- **Effective:** Catches degradation patterns reliably

---

## Next Steps for Advanced Users

1. **Deep Learning:** Use LSTM autoencoders to capture time patterns
2. **Frequency Analysis:** Add FFT (frequency domain) features
3. **Cross-Validation:** Test on multiple bearing tests simultaneously
4. **Hyperparameter Tuning:** Optimize for your specific equipment
5. **Real-Time Integration:** Deploy as monitoring system for live equipment

---

## Questions or Issues?

- Check the `Anamoly_detection_model_summary.md` file for detailed explanations
- Review inline comments in `anomaly_detection.ipynb`
- Check the visualizations — they tell the story of bearing degradation

---

## License & Attribution

NASA bearing data sourced from NASA Prognostics Center.
This analysis framework is open for educational and research use.

---

**Happy Analyzing!**
