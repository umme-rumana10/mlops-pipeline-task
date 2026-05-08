import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_job")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def write_metrics(output_path: str, payload: dict) -> None:
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_config(config_path: str, logger: logging.Logger) -> dict:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(p) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping.")

    required = {"seed", "window", "version"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")

    if not isinstance(cfg["seed"], int):
        raise ValueError(f"'seed' must be an integer, got: {type(cfg['seed'])}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"'window' must be a positive integer, got: {cfg['window']}")
    if not isinstance(cfg["version"], str):
        raise ValueError(f"'version' must be a string, got: {type(cfg['version'])}")

    logger.info(f"Config loaded — seed={cfg['seed']}, window={cfg['window']}, version={cfg['version']}")
    return cfg


def load_data(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    p = Path(input_path)

    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        # Read malformed CSV
        df = pd.read_csv(p, header=None)

        # Split single column into multiple columns
        df = df[0].str.split(",", expand=True)

        # Set first row as header
        df.columns = df.iloc[0]

        # Remove first row from data
        df = df[1:].reset_index(drop=True)

        # Convert close column to numeric
        df["close"] = pd.to_numeric(df["close"])

    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")

    if df.empty:
        raise ValueError("Input CSV is empty.")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. Columns present: {list(df.columns)}"
        )

    if not pd.api.types.is_numeric_dtype(df["close"]):
        raise ValueError("Column 'close' must be numeric.")

    logger.info(f"Dataset loaded — {len(df)} rows, columns: {list(df.columns)}")

    return df


def compute_rolling_mean(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.Series:
    # First (window-1) rows will be NaN; these rows are excluded from signal computation.
    rolling = df["close"].rolling(window=window, min_periods=window).mean()
    valid = rolling.notna().sum()
    logger.info(f"Rolling mean computed — window={window}, valid rows={valid}, NaN rows={window - 1}")
    return rolling


def compute_signal(close: pd.Series, rolling_mean: pd.Series, logger: logging.Logger) -> pd.Series:
    # Only compute signal where rolling mean is available
    mask = rolling_mean.notna()
    signal = pd.Series(np.nan, index=close.index)
    signal[mask] = (close[mask] > rolling_mean[mask]).astype(int)
    valid_signals = signal[mask]
    rate = float(valid_signals.mean())
    logger.info(f"Signal generated — rows with signal={mask.sum()}, signal_rate={rate:.6f}")
    return signal


def main():
    parser = argparse.ArgumentParser(description="MLOps batch signal pipeline")
    parser.add_argument("--input",    required=True, help="Path to input CSV")
    parser.add_argument("--config",   required=True, help="Path to YAML config")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path for log file")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    logger.info("=== Job started ===")

    start_ts = time.monotonic()
    version = "unknown"

    try:
        # 1. Load + validate config
        cfg = load_config(args.config, logger)
        version = cfg["version"]
        seed, window = cfg["seed"], cfg["window"]

        # 2. Set seed for reproducibility
        np.random.seed(seed)
        logger.info(f"Random seed set: {seed}")

        # 3. Load + validate dataset
        df = load_data(args.input, logger)

        # 4. Rolling mean
        rolling_mean = compute_rolling_mean(df, window, logger)

        # 5. Signal generation
        signal = compute_signal(df["close"], rolling_mean, logger)

        # 6. Metrics
        valid_mask = signal.notna()
        rows_processed = int(valid_mask.sum())
        signal_rate = float(signal[valid_mask].mean())
        latency_ms = int((time.monotonic() - start_ts) * 1000)

        metrics = {
            "version": version,
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": round(signal_rate, 4),
            "latency_ms": latency_ms,
            "seed": seed,
            "status": "success",
        }

        write_metrics(args.output, metrics)
        logger.info(f"Metrics — rows_processed={rows_processed}, signal_rate={signal_rate:.4f}, latency_ms={latency_ms}")
        logger.info("=== Job finished — status: success ===")

        print(json.dumps(metrics, indent=2))
        sys.exit(0)

    except Exception as exc:
        latency_ms = int((time.monotonic() - start_ts) * 1000)
        logger.error(f"Job failed: {exc}", exc_info=True)
        logger.info("=== Job finished — status: error ===")

        error_payload = {
            "version": version,
            "status": "error",
            "error_message": str(exc),
        }
        try:
            write_metrics(args.output, error_payload)
        except Exception as write_err:
            logger.error(f"Could not write error metrics: {write_err}")

        print(json.dumps(error_payload, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()