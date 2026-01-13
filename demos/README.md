# NoiseChain Investor Demo Package

This directory contains resources for Investor Technical Due Diligence.

## Contents

- **`run_investor_package.py`**: A one-click script that runs the Full E2E Demo and Performance Benchmark.
- **`results/`**: Directory where benchmark reports (JSON) are saved.

## How to Run

```bash
# Execute the full demo suite
python demos/run_investor_package.py
```

## Expected Output

1. **Functional Demo**:
   - Sensor Data Collection -> Token Generation -> Verification -> Storage
   - Simulates the entire trust verification lifecycle.

2. **Performance Benchmark**:
   - Runs 100 iterations of token generation & verification.
   - Measures latency (avg/min/max) and throughput (TPS).
   - Saves detailed report to `demos/results/benchmark_report_YYYYMMDD_HHMMSS.json`.

## Benchmark Results (Typical)

- **Latency**: ~6ms (p50)
- **Throughput**: >150 TPS (Single Thread, Python)
- **Token Size**: 199 bytes

---

*Verified on: 2026-01-14*
