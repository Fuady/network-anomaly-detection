"""
src/data_engineering/data_validation.py
────────────────────────────────────────
Data quality checks for network KPI data before model training.

Usage:
    python src/data_engineering/data_validation.py --input data/raw/network_kpis.parquet
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ValidationResult:
    rule: str
    passed: bool
    message: str
    severity: str = "ERROR"

    def __str__(self):
        icon = "✓" if self.passed else ("✗" if self.severity == "ERROR" else "⚠")
        return f"  [{icon}] {self.rule}: {self.message}"


class DataValidator:
    def __init__(self, df: pd.DataFrame, name: str = "dataset"):
        self.df = df
        self.name = name
        self.results: list[ValidationResult] = []

    def _add(self, rule, passed, msg, sev="ERROR"):
        self.results.append(ValidationResult(rule, passed, msg, sev))

    def expect_row_count_above(self, n: int):
        self._add("row_count", len(self.df) >= n,
                  f"{len(self.df):,} rows (need ≥ {n:,})")
        return self

    def expect_columns(self, cols: list):
        missing = [c for c in cols if c not in self.df.columns]
        self._add("required_columns", not missing,
                  f"Missing: {missing}" if missing else f"All {len(cols)} columns present")
        return self

    def expect_no_nulls(self, cols: list):
        for col in cols:
            if col not in self.df.columns: continue
            n = self.df[col].isnull().sum()
            self._add(f"no_nulls:{col}", n == 0,
                      f"{n:,} nulls" if n else "no nulls")
        return self

    def expect_range(self, col: str, lo: float, hi: float, sev="ERROR"):
        if col not in self.df.columns: return self
        out = ((self.df[col] < lo) | (self.df[col] > hi)).sum()
        self._add(f"range:{col}", out == 0,
                  f"{out:,} values outside [{lo}, {hi}]" if out else f"all in [{lo}, {hi}]",
                  severity=sev)
        return self

    def expect_positive(self, col: str):
        if col not in self.df.columns: return self
        neg = (self.df[col] < 0).sum()
        self._add(f"positive:{col}", neg == 0,
                  f"{neg:,} negative values" if neg else "all positive")
        return self

    def expect_no_duplicate_timestamps_per_site(self):
        if "site_id" not in self.df.columns or "timestamp" not in self.df.columns:
            return self
        dups = self.df.duplicated(["site_id", "timestamp"]).sum()
        self._add("no_dup_ts", dups == 0,
                  f"{dups:,} duplicate site×timestamp pairs" if dups else "no duplicates")
        return self

    def expect_timestamp_sorted(self):
        if "timestamp" not in self.df.columns: return self
        sorted_ = self.df["timestamp"].is_monotonic_increasing
        self._add("timestamp_sorted", sorted_,
                  "timestamps NOT monotonically increasing" if not sorted_ else "timestamps sorted")
        return self

    def report(self) -> bool:
        print(f"\n{'='*55}")
        print(f"VALIDATION: {self.name}")
        print(f"{'='*55}")
        errors   = [r for r in self.results if not r.passed and r.severity == "ERROR"]
        warnings = [r for r in self.results if not r.passed and r.severity == "WARNING"]
        for r in self.results:
            print(r)
        print(f"\n  Passed: {sum(r.passed for r in self.results)} | "
              f"Warnings: {len(warnings)} | Errors: {len(errors)}")
        print("=" * 55)
        if errors:
            logger.error(f"Validation FAILED: {len(errors)} errors")
        else:
            logger.success("Validation PASSED")
        return len(errors) == 0


def validate_network_kpis(df: pd.DataFrame) -> bool:
    REQUIRED = [
        "site_id", "timestamp", "rsrq_avg", "rsrp_avg",
        "throughput_mbps", "latency_ms", "packet_loss_pct",
        "connected_users", "prb_utilization", "sinr_avg",
    ]
    v = DataValidator(df, "network_kpis.parquet")
    v.expect_row_count_above(10_000)
    v.expect_columns(REQUIRED)
    v.expect_no_nulls(["site_id", "timestamp", "rsrq_avg", "throughput_mbps"])
    v.expect_range("rsrq_avg",       -25.0, -3.0)
    v.expect_range("rsrp_avg",      -130.0, -60.0)
    v.expect_range("throughput_mbps",  0.0, 500.0)
    v.expect_range("latency_ms",       0.0, 5000.0)
    v.expect_range("packet_loss_pct",  0.0, 100.0)
    v.expect_range("prb_utilization",  0.0, 1.0)
    v.expect_positive("connected_users")
    v.expect_no_duplicate_timestamps_per_site()
    return v.report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/network_kpis.parquet")
    args = parser.parse_args()
    path = Path(args.input)
    if not path.exists():
        logger.error(f"File not found: {path}")
        logger.info("Run: python src/data_engineering/generate_data.py")
        sys.exit(1)
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows")
    ok = validate_network_kpis(df)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
