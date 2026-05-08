# MLOps Batch Signal Pipeline

A minimal, reproducible MLOps batch job that loads OHLCV data, computes a rolling-mean signal, and emits structured metrics — runnable locally or inside Docker.

---

## Project Structure

```
.
├── run.py            # Main pipeline script
├── config.yaml       # Job configuration (seed, window, version)
├── data.csv          # 10 000-row OHLCV dataset
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition
├── metrics.json      # Sample output from a successful run
├── run.log           # Sample log from a successful run
└── README.md
```

---

## Local Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

After completion you will find:
- `metrics.json` — machine-readable results
- `run.log` — detailed execution log

---

## Docker Build & Run

### Build the image

```bash
docker build -t mlops-task .
```

### Run the container

```bash
docker run --rm mlops-task
```

The container prints the final metrics JSON to stdout.  
Exit code `0` = success, non-zero = failure.

### (Optional) Copy output files out of the container

```bash
docker run --rm -v "$(pwd)/output:/app/output" mlops-task \
  python run.py \
    --input    data.csv \
    --config   config.yaml \
    --output   output/metrics.json \
    --log-file output/run.log
```

---

## Configuration (`config.yaml`)

| Key       | Type   | Description                              |
|-----------|--------|------------------------------------------|
| `seed`    | int    | NumPy random seed for reproducibility    |
| `window`  | int    | Rolling-mean window size (rows)          |
| `version` | string | Pipeline version tag in metrics output   |

---

## Signal Logic

1. **Rolling mean** — computed over `close` with `window` from config.  
   The first `window - 1` rows produce `NaN` and are **excluded** from signal computation and `rows_processed`.
2. **Signal** — `1` if `close > rolling_mean`, else `0`.
3. **signal_rate** — mean of all valid signal values.

---

## Example `metrics.json` (success)

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.499,
  "latency_ms": 134,
  "seed": 42,
  "status": "success"
}
```

> `rows_processed` is 9 996 because the first 4 rows (window − 1 = 4) are NaN and excluded.

## Example `metrics.json` (error)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found. Columns present: ['open', 'high', 'low', 'volume']"
}
```

---

## Reproducibility

Setting `numpy.random.seed(seed)` at startup ensures deterministic behaviour across runs given the same config and data. All processing is purely deterministic (rolling mean, comparison) so results are fully reproducible.