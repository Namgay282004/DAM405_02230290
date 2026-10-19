# Practical 5 Report: Implement Logging and Monitoring for an ML Model

**Course:** DAM405 Machine Learning Operations

**Topic:** Practical 5 – Implementing Logging and Monitoring for an ML Model

**Date:** October 9, 2026

---

## 1. Executive Summary

Machine learning models deployed in production degrade silently due to real-world data drift and shift in input environments. Unlike traditional backend systems that crash with unhandled stack traces, ML services often continue returning success status codes with invalid or stale prediction outputs.

This report presents the implementation of robust logging, health telemetry monitoring, and automated feature-drift identification for a Flask service utilizing a persisted scikit-learn Random Forest model (`model.joblib`) from Practical 3. By replacing unformatted standard output statements with structured rotating logging handlers and trailing latency counters, operational visibility and failure diagnosis are established.

---

## 2. Methodology & Implementation Strategy

### 2.1 Model Integration & Structured Logging Architecture (Task 1)

The trained binary model file (`model.joblib`) generated in Practical 3 was loaded upon initialization via `joblib.load()` within an exception block logging severe initialization issues at `CRITICAL` level.

Standard `print()` logging was replaced with a `logging` setup utilizing:

* **StreamHandler:** Directs real-time events to standard output for container engine aggregation.


* **RotatingFileHandler:** Writes logs to `service.log`, limited to a maximum file size of 1 MB with 3 backup rotations to protect storage capacity.


* **Formatter String:** `%(asctime)s | %(levelname)s | %(message)s` to ensure consistent execution tracing across handlers.



### 2.2 Instrumentation & Telemetry Tracking (Tasks 2 & 4)

The inference endpoint `/predict` was instrumented using precise timing blocks (`time.perf_counter()`):

* **Latency Profile:** Every response execution time is recorded in a bounded double-ended queue (`deque(maxlen=100)`) maintaining a sliding window average.


* **Error & Success Counters:** Maintained using a global monitoring state (`stats['requests']`, `stats['errors']`) updated after each request.


* **Periodic Summaries:** A log line summarising request volume, failure rate, and mean response latency is published periodically.



### 2.3 Single-Feature Drift Detection Engine (Task 5)

Data drift occurs when production inputs deviate significantly from the training baseline. The service calculates relative feature shift using the formula:

$$\text{Relative Drift} = \frac{\vert{}\mu_{\text{live}} - \mu_{\text{train}}\vert{}}{\vert{}\mu_{\text{train}}\vert{} + 10^{-9}}$$

If the relative shift exceeds a 25% tolerance threshold ($tol = 0.25$), a `WARNING` entry is issued to log streams.

---

## 3. Implementation Evidence & Results

### 3.1 Verification Evidences

#### Figure 1: Model Initialization & Service Log Level Demo (Tasks 1 & 3)

*(Insert **Screenshot #1** here: Terminal startup output showing model loading from `model.joblib` and Task 3 log demonstration.)*

```text
2026-10-09 09:44:44,500 | INFO | Model successfully loaded from model.joblib
2026-10-09 09:44:44,501 | INFO | normal event service handled a request
2026-10-09 09:44:44,501 | WARNING | unexpected input but still handled
2026-10-09 09:44:44,501 | ERROR | a request failed
2026-10-09 09:44:44,501 | CRITICAL | service cannot continue

```

#### Figure 2: Operational Monitoring, Error Capture & Drift Detection (Tasks 2, 4 & 5)

*(Insert **Screenshot #2** here: Terminal output capturing prediction latencies, bad payload error logging, and the data drift notification.)*

```text
2026-10-09 09:46:13,721 | ERROR | prediction failed: X has 2 features, but RandomForestClassifier is expecting 4 features as input.
2026-10-09 09:46:24,126 | ERROR | prediction failed: Missing 'features' key in payload
2026-10-09 09:49:20,863 | WARNING | possible drift: live=5.100 train=0.500 change=920%
2026-10-09 09:49:20,872 | INFO | prediction=0 latency_ms=8.8
2026-10-09 09:49:20,888 | WARNING | possible drift: live=4.900 train=0.500 change=880%
2026-10-09 09:49:20,914 | INFO | prediction=0 latency_ms=25.3
2026-10-09 09:49:20,914 | INFO | requests=5 errors=3 avg_latency_ms=7.1
2026-10-09 09:49:58,511 | WARNING | possible drift: live=18.500 train=0.500 change=3600%
2026-10-09 09:49:58,515 | INFO | prediction=0 latency_ms=4.5
2026-10-09 09:50:03,052 | WARNING | possible drift: live=5.800 train=0.500 change=1060%
2026-10-09 09:50:03,058 | INFO | prediction=2 latency_ms=5.5
2026-10-09 09:50:03,067 | WARNING | possible drift: live=6.300 train=0.500 change=1160%
2026-10-09 09:50:03,072 | INFO | prediction=2 latency_ms=5.5

```

#### Figure 3: Persistent File Log Inspection (`service.log`)

*(Insert **Screenshot #3** here: Image showing `service.log` contents persisted on disk.)*

---

## 4. Reflection Questions & Answers

### Question 1: Why is the logging module preferable to `print()` for a production service?



**Answer:**

Using standard `print()` statements in production services presents key operational risks:

1. **Lack of Metadata:** `print()` lacks basic contextual info such as precise execution timestamps, process IDs, or severity classification tags.


2. **Flexible Routing & Handlers:** The `logging` library supports routing messages simultaneously to multiple streams (e.g., standard output, rolling disk files, or centralized aggregators like Datadog and CloudWatch).


3. **Storage & Rotation Control:** Standard print statements directing to files will consume disk space continuously until failure; standard logging handlers like `RotatingFileHandler` maintain hard limits on file counts and size.


4. **Dynamic Output Filtering:** Log level configurations allow operators to change output detail without altering application logic.



---

### Question 2: Which log level would you run in production, and why not `DEBUG`?



**Answer:**

Production environments should execute at the **`INFO`** log level (or **`WARNING`** for high-throughput pipelines).

`DEBUG` mode should be avoided in production environments for three primary reasons:

1. **Performance Penalties:** High-volume logging introduces I/O blocking and increases API latency.


2. **Centralized Storage Costs:** Logging detailed diagnostics across high transaction volumes creates storage overhead and inflates log aggregation costs.
3. **Data Security & Privacy:** `DEBUG` statements often log raw input arrays, introducing compliance risks by capturing sensitive data in unencrypted log stores.



---

### Question 3: Your drift check only looks at the mean. What does it miss, and how would you improve it?



**Answer:**

Checking relative mean changes serves as a lightweight heuristic, but fails to capture complex distribution changes:

* **Variance Shifts:** If feature values spread significantly wider while remaining centered around the same point, the mean remains unchanged while model accuracy degrades.
* **Multimodal Distributions:** Shifts between peaks in bimodal distributions will pass unnoticed if the overall mathematical mean remains static.
* **Categorical Features:** Non-numeric categorical values cannot be analyzed using simple mean values.
* **Feature Covariance:** Single-feature drift monitoring misses structural breaks in relationships between multiple features.

**Improvements:**

1. **Statistical Distance Metrics:** Implement tests like the Kolmogorov-Smirnov (KS) test for continuous distributions, Population Stability Index (PSI), or Chi-Square tests for discrete data.
2. **MLOps Drift Frameworks:** Integrate specialized monitoring tools such as **Evidently AI** to compute comprehensive statistical profiles comparing live request batches to baseline training datasets.