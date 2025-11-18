# Bearing Anomaly Detection Using IsolationForest + LSTM Autoencoder

A complete hybrid anomaly detection pipeline for predictive maintenance using unsupervised learning.

The model identifies abnormal machine behavior from raw vibration sensor data without requiring any labels.

---

## Dataset Overview

The dataset consists of raw vibration signals collected from a rotating machine bearing.

**Format:** Multiple folders (1st_test, 2nd_test, 3rd_test)
- Each file represents one vibration snapshot
- Each snapshot is a multi-column time-series with 4 to 8 channels
- Filenames contain precise timestamps in format YYYY.MM.DD.HH.MM.SS
- Total files across all folders: approximately 5,000 snapshots
- No labels provided, making this ideal for unsupervised anomaly detection

**Example data from a single file:**

```
-0.022  -0.039  -0.183  -0.054  -0.105  -0.134  -0.129  -0.142
-0.105  -0.017  -0.164  -0.183  -0.049   0.029  -0.115  -0.122
```

Each row represents vibration readings at a specific moment. Each column represents a different sensor channel.

---

## Project Goal

To detect anomalies in bearing vibration patterns by combining two complementary approaches:

- **IsolationForest:** Catches point-based anomalies in statistical features
- **LSTM Autoencoder:** Catches temporal sequence anomalies and pattern deviations
- **Score Fusion:** Improves reliability and reduces false positives

---

## Pipeline Overview

```
Raw vibration files
        ↓
Per-file feature extraction (mean, std, min, max, median)
        ↓
StandardScaler (normalization)
        ↓
IsolationForest anomaly scoring
        ↓
Sliding window sequence creation (sequence_length=100)
        ↓
LSTM Autoencoder reconstruction error
        ↓
Score normalization + fusion
        ↓
Thresholding
        ↓
Final anomaly detection
```

---

## Feature Engineering

For each sensor channel, you extract five key statistical features:

- **Mean:** Average vibration amplitude
- **Standard Deviation:** Variability in the vibration signal
- **Min:** Lowest vibration value recorded
- **Max:** Highest vibration value recorded

These features represent the signal's amplitude, variability, and extremes. Together they capture the overall behavior of each snapshot in a compact format.

---

## Model 1: IsolationForest

**Purpose:** Detect point-based anomalies

**How it works:**
- Unsupervised anomaly detection algorithm
- Identifies individual snapshots with abnormal statistical behavior
- Works on normalized feature vectors
- Isolates outliers by randomly selecting features and split values

**Output:** Isolation Forest anomaly score for each snapshot

---

## Model 2: LSTM Autoencoder

**Purpose:** Detect temporal sequence anomalies

**How it works:**
- Learns the normal temporal pattern of vibration sequences
- Attempts to reconstruct sliding window sequences
- High reconstruction error indicates anomalous sequences
- Sequence length: 100 time steps

**Architecture:**
- Encoder: LSTM layer compresses sequence into latent representation
- Decoder: LSTM layer reconstructs the sequence
- Loss: Mean Squared Error between input and reconstruction

**Output:** Reconstruction error (MSE) for each sequence

---

## Score Fusion

Both models provide independent anomaly indicators. They are combined as follows:

1. Normalize both scores using MinMax scaling (0 to 1 range)
2. Combine with equal weighting:

```
final_score = 0.5 * IF_score + 0.5 * AE_score
```

3. Set threshold at 99th percentile of scores
4. Flag scores above threshold as anomalies

This fusion approach leverages the strengths of both methods:
- IsolationForest catches sudden statistical changes
- LSTM catches gradual pattern drift
- Combined score is more robust and reduces false positives

---

## Project Outputs

**Visualizations:**
- IsolationForest anomaly score plot over time
- LSTM reconstruction error distribution
- Fused score anomaly chart with threshold
- Time-series plots highlighting detected anomalies

**Data Exports:**
- Summary CSV with all anomaly flags
- Detailed results for each detection method
- Timestamps of flagged anomalies

**Saved Models:**
- `if_model.pkl` - Trained IsolationForest model
- `scaler.pkl` - Feature scaler for future predictions
- `ae_model.h5` - Trained LSTM Autoencoder model

---

## Technology Stack

- **Language:** Python 3.8+
- **Data Processing:** NumPy, Pandas
- **Machine Learning:** Scikit-Learn
- **Deep Learning:** TensorFlow, Keras
- **Visualization:** Matplotlib
- **Environment:** Jupyter Notebook, Google Colab

---

## Why This Hybrid Approach Works

**Captures Instantaneous Anomalies:** IsolationForest detects sudden changes in vibration statistics that might indicate sudden impacts or mechanical failures.

**Captures Temporal Patterns:** LSTM Autoencoder learns the normal flow of vibration sequences and catches gradual degradation or pattern shifts over time.

**Reduces False Positives:** By combining both approaches, you avoid triggering on isolated spikes. A true anomaly is confirmed by both methods.

**Fully Unsupervised:** No labeled data required. The models learn what "normal" looks like directly from the data.

**Industrial-Ready:** Designed to work with real sensor data that often lacks perfect labels.

**Scalable and Extensible:** Easy to add more models, adjust thresholds, or deploy to production systems.

---

## Use Cases

- Predictive maintenance for industrial machinery
- IoT vibration analytics platforms
- Bearing fault detection and early warning systems
- Rotating equipment anomaly detection
- Condition-based maintenance scheduling
- Asset health monitoring

---

## Key Advantages

1. Works without labeled data
2. Combines statistical and temporal pattern recognition
3. Robust to noise and single outliers
4. Produces both model predictions and fusion scores
5. Outputs timestamped anomaly flags for investigation
6. Models can be saved and reused for new data
7. Easy to tune thresholds based on maintenance requirements

---

## Installation & Setup

Install required dependencies:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

---

## Running the Project

1. **Prepare your data:** Upload bearing vibration data to Google Drive in folders (1st_test, 2nd_test, 3rd_test)

2. **Open the notebook:** Upload the `Bearing_Anomaly_Detection_Final.ipynb` to Google Colab

3. **Update data path:** Modify the folder path in the first code cell to match your data location:
   ```python
   folder_path = '/content/drive/MyDrive/archive/1st_test'
   ```

4. **Run cells sequentially:** Execute all cells from top to bottom

5. **Review outputs:** Check generated visualizations and anomaly flags

6. **Results saved:** All results (CSV files and trained models) are automatically saved to your Google Drive

---

## Project Workflow

1. **Data Loading:** Reads all bearing vibration files from specified folder
2. **Feature Extraction:** Computes mean, std, min, max, median for each snapshot
3. **Normalization:** Scales features to standard normal distribution
4. **IsolationForest Training:** Learns normal feature patterns
5. **LSTM Training:** Learns normal temporal sequences
6. **Anomaly Scoring:** Generates scores from both models
7. **Score Fusion:** Combines both approaches for final decision
8. **Visualization:** Plots all results for analysis
9. **Export:** Saves anomaly results to CSV and models to disk

---

## Troubleshooting

**Problem:** "Folder does not exist" error
- **Solution:** Verify the folder path matches your actual Google Drive structure

**Problem:** Very few data points loaded
- **Solution:** Check that files are in the correct format (plain text with numeric values)

**Problem:** LSTM training seems wrong
- **Solution:** Ensure you have at least 100 data points. Increase sequence length if you have more data

**Problem:** No anomalies detected
- **Solution:** Adjust contamination parameter in IsolationForest or threshold percentile

---

## Model Parameters

**IsolationForest:**
- `contamination=0.05` - Assumes 5% of data are anomalies
- `random_state=42` - For reproducibility

**LSTM Autoencoder:**
- `sequence_length=100` - Window size for temporal patterns
- `LSTM_units=32` - Neural network complexity
- `epochs=15` - Training iterations
- `batch_size=32` - Samples per training step

---

## Future Improvements

- Add more sophisticated feature engineering (FFT, entropy)
- Implement real-time anomaly detection
- Deploy as REST API for production use

---

## References

- IsolationForest: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008)
- LSTM Autoencoders: Malhotra et al., 2015
- Bearing Dataset: NASA Ames Prognostics Data Repository

---

## License

This project is open source and available under the MIT License.

---

## Author Notes

This hybrid approach combines the strengths of statistical anomaly detection with deep learning sequence modeling. It's designed to be practical for real-world bearing monitoring systems where labeled training data is rarely available.

For questions or improvements, feel free to contribute to the project.
