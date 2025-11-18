# Understanding Your Bearing Vibration Dataset - A Guide

## Let me explain what you're working with:

Imagine you have a machine with a rotating bearing. You place vibration sensors on it to listen to how it's vibrating. Every few seconds, you record a snapshot of those vibrations. That snapshot is what you have in each file.

## How your data is organized:

You have three folders: 1st_test, 2nd_test, and 3rd_test. Together, they contain around 5,000 to 7,000 files. Each file is named with a timestamp like `2004.02.12.10.32.39`, which tells you exactly when that vibration snapshot was recorded. This is really helpful because it means you can follow the machine's behavior over time in chronological order.

## What's inside each file:

Open any file and you'll see rows of numbers. Each row is a different moment in time, and each column is a different sensor channel. Think of it like this: you might have 4 sensors mounted on the bearing, so you'd have 4 columns. Each number in those columns tells you the vibration level at that exact moment.

For example:
```
-0.022  -0.039  -0.183  -0.054
-0.105  -0.017  -0.164  -0.183
-0.049   0.029  -0.115  -0.122
```

A typical file has between 100 and 200 rows of these readings.

## Why this data matters:

Here's the interesting part: when a bearing is healthy, its vibration pattern is predictable and stable. But as it wears down or something goes wrong, the vibration pattern changes. You might see sudden spikes, or the overall pattern might gradually shift. By analyzing these patterns, you can catch problems before they become catastrophic failures.

## The challenge you're solving:

Here's what makes your project interesting: nobody has labeled these files as "normal" or "broken." You have to figure it out on your own using unsupervised learning. This is actually closer to real-world problems, where you rarely have labeled data.

## Your two-pronged approach:

First, you calculate simple summary statistics for each snapshot: the average vibration, how spread out it is, the highest and lowest values. Then you use Isolation Forest to spot snapshots that look unusual compared to the rest.

But here's the clever part: you also feed sequences of these snapshots into an LSTM autoencoder. This model learns what "normal" vibration sequences look like, and then you check if new sequences match that pattern. If a sequence doesn't match what the model learned, it's probably an anomaly.

By combining both approaches, you get a more reliable picture than either method alone would give you.

## In simple terms:

You're essentially teaching a machine to listen to a bearing's vibrations, learn what healthy sounds like, and then alert you when something sounds wrong. And you're doing it two different ways to make sure you catch real problems.

---

## Dataset Summary Table

| Aspect | Details |
|--------|---------|
| Total Files | 5,000 - 7,000 snapshots |
| Organization | 3 folders (1st_test, 2nd_test, 3rd_test) |
| Timestamp Format | YYYY.MM.DD.HH.MM.SS |
| Rows per File | 100 - 200 readings |
| Columns per File | 4 - 8 sensor channels |
| Data Type | Vibration amplitude values |
| Labels | None (unsupervised) |
| Use Case | Anomaly detection, predictive maintenance |

---

## Key Takeaways

1. Each file represents a moment in time when the bearing's vibrations were measured
2. The data is chronologically ordered, so you can track changes over time
3. Normal bearings have predictable vibration patterns
4. Anomalies show as deviations from the learned normal pattern
5. Your dual approach (statistical + sequential) catches different types of anomalies
