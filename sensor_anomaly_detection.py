"""Sensor anomaly detection on synthetic multi-sensor data.

Generates a deterministic sensor feed (temperature + humidity), injects known
anomalies, flags anomalous readings with statistical detectors, and reports
precision/recall against the injected ground truth.

The statistical detectors run on *residuals* — the reading minus the expected
periodic signal (base + amplitude * sin, i.e. a known-period seasonal
decomposition). Without detrending, the daily cycle both hides real anomalies
and manufactures false ones; on raw values a trailing-window z-score can even
never exceed ~1.7 sigma for a smooth linear drift.

Detectors:
    range       - raw value outside the sensor's hard physical limits
    global_z    - residual |z| > threshold vs the whole series (extreme outliers)
    rolling_z   - residual |z| > threshold vs a trailing window (local spikes)
    rate        - raw jump between consecutive readings exceeds the rate limit
    flatline    - N consecutive exactly-identical readings (stuck sensor)
    level_shift - residual median of a trailing window vs the preceding window
                  differs by > threshold sigma (catches drift; medians keep a
                  single spike from masquerading as a level shift)

Run with: uv run python sensor_anomaly_detection.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import fmean, median, stdev

SEED = 42
NUM_SAMPLES = 500
SAMPLE_INTERVAL = timedelta(minutes=5)
START_TIME = datetime(2026, 1, 1)

Z_THRESHOLD = 3.5
ROLLING_WINDOW = 25
LEVEL_SHIFT_WINDOW = 20
LEVEL_SHIFT_THRESHOLD = 4.0
MIN_WINDOW_POINTS = 10
FLATLINE_RUN = 6
# A detection counts as a hit if it lands within +/- this many samples of an
# injected anomaly (rate-of-change also flags the recovery step after a spike).
MATCH_TOLERANCE = 1

KIND_SPIKE = "spike"
KIND_OUT_OF_RANGE = "out_of_range"
KIND_FLATLINE = "flatline"
KIND_DRIFT = "drift"

DRIFT_WINDOW = 30
DRIFT_RECOVERY = 15
DRIFT_MAX_OFFSET = 10.0
FLATLINE_RUN_INJECTED = FLATLINE_RUN + 2
# Window detectors keep firing for ~window samples after an anomaly ends while
# their trailing reference clears; flags in that settling tail are not false
# positives. See build_acceptable_zone().
GRACE_AFTER_ANOMALY = LEVEL_SHIFT_WINDOW
MAX_PRINTED_ANOMALIES = 20


@dataclass(frozen=True)
class SensorConfig:
    """Physical limits and signal shape for one synthetic sensor."""

    name: str
    base: float
    amplitude: float
    period: int
    noise: float
    min_value: float
    max_value: float
    rate_limit: float


@dataclass(frozen=True)
class Anomaly:
    """One flagged reading and the detectors that flagged it."""

    index: int
    timestamp: datetime
    value: float
    methods: tuple[str, ...]

    @property
    def severity(self) -> int:
        return len(self.methods)


SENSOR_CONFIGS = (
    SensorConfig(
        name="temperature",
        base=22.0,
        amplitude=4.0,
        period=144,
        noise=0.4,
        min_value=10.0,
        max_value=35.0,
        rate_limit=3.0,
    ),
    SensorConfig(
        name="humidity",
        base=45.0,
        amplitude=5.0,
        period=200,
        noise=1.0,
        min_value=20.0,
        max_value=80.0,
        rate_limit=5.0,
    ),
)


# --------------------------------------------------------------------------
# Synthetic data generation
# --------------------------------------------------------------------------

def expected_value(config: SensorConfig, index: int) -> float:
    """The normal periodic signal at a sample index."""
    return config.base + config.amplitude * math.sin(
        2 * math.pi * index / config.period
    )


def generate_series(config: SensorConfig, rng: random.Random) -> list[float]:
    """Return a clean periodic signal plus Gaussian noise, no anomalies."""
    return [
        expected_value(config, i) + rng.gauss(0, config.noise)
        for i in range(NUM_SAMPLES)
    ]


def residuals(values: list[float], config: SensorConfig) -> list[float]:
    """Detrended readings: value minus the expected periodic signal."""
    return [value - expected_value(config, i) for i, value in enumerate(values)]


def inject_anomalies(
    config: SensorConfig, clean: list[float]
) -> tuple[list[float], dict[str, frozenset[int]]]:
    """Return a copy of `clean` with known anomalies and the ground-truth indices."""
    injected = list(clean)
    truth: dict[str, frozenset[int]] = {}

    if config.name == "temperature":
        spike_index = NUM_SAMPLES // 5
        injected[spike_index] = clean[spike_index] + 8.0
        truth[KIND_SPIKE] = frozenset({spike_index})

        oor_index = 2 * NUM_SAMPLES // 5
        injected[oor_index] = config.max_value + 3.5
        truth[KIND_OUT_OF_RANGE] = frozenset({oor_index})

    if config.name == "humidity":
        run_start = NUM_SAMPLES // 2
        run_end = run_start + FLATLINE_RUN_INJECTED
        injected[run_start:run_end] = [clean[run_start]] * FLATLINE_RUN_INJECTED
        truth[KIND_FLATLINE] = frozenset(range(run_start, run_end))

        drift_start = 3 * NUM_SAMPLES // 4
        top = drift_start + DRIFT_WINDOW
        recovery_end = min(top + DRIFT_RECOVERY, NUM_SAMPLES)
        for i in range(drift_start, top):
            injected[i] = clean[i] + DRIFT_MAX_OFFSET * (i - drift_start) / DRIFT_WINDOW
        for i in range(top, recovery_end):
            injected[i] = clean[i] + DRIFT_MAX_OFFSET * (1 - (i - top) / DRIFT_RECOVERY)
        truth[KIND_DRIFT] = frozenset(range(drift_start, recovery_end))

    return injected, truth


# --------------------------------------------------------------------------
# Detectors (each returns the set of flagged indices)
# --------------------------------------------------------------------------

def detect_range_violations(values: list[float], config: SensorConfig) -> set[int]:
    return {
        i
        for i, value in enumerate(values)
        if value < config.min_value or value > config.max_value
    }


def detect_global_zscore(values: list[float], threshold: float) -> set[int]:
    if len(values) < MIN_WINDOW_POINTS:
        raise ValueError(
            f"need at least {MIN_WINDOW_POINTS} samples for z-score detection"
        )
    mean = fmean(values)
    sd = stdev(values)
    if sd == 0:
        return set()
    return {i for i, value in enumerate(values) if abs(value - mean) / sd > threshold}


def detect_rolling_zscore(
    values: list[float], window: int, threshold: float
) -> set[int]:
    if window < MIN_WINDOW_POINTS:
        raise ValueError(f"window must be at least {MIN_WINDOW_POINTS}")
    if window >= len(values):
        raise ValueError("window must be smaller than the number of samples")
    flagged: set[int] = set()
    for i in range(window, len(values)):
        reference = values[i - window : i]
        mean = fmean(reference)
        sd = stdev(reference)
        if sd > 0 and abs(values[i] - mean) / sd > threshold:
            flagged.add(i)
    return flagged


def detect_rate_of_change(values: list[float], limit: float) -> set[int]:
    return {
        i
        for i in range(1, len(values))
        if abs(values[i] - values[i - 1]) > limit
    }


def detect_flatline(values: list[float], run_length: int) -> set[int]:
    """Flag every index in any run of exactly-identical readings >= run_length."""
    flagged: set[int] = set()
    run_start = 0
    for i in range(1, len(values) + 1):
        if i < len(values) and values[i] == values[run_start]:
            continue
        if i - run_start >= run_length:
            flagged.update(range(run_start, i))
        run_start = i
    return flagged


def detect_level_shift(
    values: list[float], window: int, threshold: float
) -> set[int]:
    """Flag points where the trailing window's median drifts from the preceding
    window's median. Medians ignore one-off spikes; the ramp inside a drift
    shifts both windows' medians apart steadily."""
    if 2 * window >= len(values):
        raise ValueError(
            f"need at least {2 * window + 1} samples for level-shift detection"
        )
    flagged: set[int] = set()
    for i in range(2 * window, len(values)):
        recent = values[i - window : i]
        prior = values[i - 2 * window : i - window]
        diff = median(recent) - median(prior)
        scale = math.sqrt(stdev(recent) ** 2 + stdev(prior) ** 2) / math.sqrt(window)
        if scale > 0 and abs(diff) / scale > threshold:
            flagged.add(i)
    return flagged


# --------------------------------------------------------------------------
# Detection pipeline
# --------------------------------------------------------------------------

def detect_anomalies(
    values: list[float], config: SensorConfig
) -> tuple[dict[int, Anomaly], dict[str, int]]:
    """Run all detectors and return (anomaly records, per-detector counts)."""
    resid = residuals(values, config)
    detector_results = {
        "range": detect_range_violations(values, config),
        "global_z": detect_global_zscore(resid, Z_THRESHOLD),
        "rolling_z": detect_rolling_zscore(resid, ROLLING_WINDOW, Z_THRESHOLD),
        "rate": detect_rate_of_change(values, config.rate_limit),
        "flatline": detect_flatline(values, FLATLINE_RUN),
        "level_shift": detect_level_shift(
            resid, LEVEL_SHIFT_WINDOW, LEVEL_SHIFT_THRESHOLD
        ),
    }
    counts = {name: len(indices) for name, indices in detector_results.items()}

    by_index: dict[int, list[str]] = {}
    for name, indices in detector_results.items():
        for i in indices:
            by_index.setdefault(i, []).append(name)

    anomalies = {
        i: Anomaly(
            index=i,
            timestamp=START_TIME + i * SAMPLE_INTERVAL,
            value=values[i],
            methods=tuple(sorted(methods)),
        )
        for i, methods in sorted(by_index.items())
    }
    return anomalies, counts


def build_acceptable_zone(truth: frozenset[int]) -> frozenset[int]:
    """Indices where a flag is not counted as a false positive.

    Includes the injected indices themselves (with +/- MATCH_TOLERANCE slack)
    plus a GRACE_AFTER_ANOMALY settling tail after the last index of each
    contiguous region: window-based detectors keep firing for ~window samples
    after an anomaly ends until their trailing window clears.
    """
    zone: set[int] = set()
    for t in truth:
        zone.update(range(t - MATCH_TOLERANCE, t + MATCH_TOLERANCE + 1))
    region_ends = {t for t in truth if t + 1 not in truth}
    for end in region_ends:
        zone.update(range(end + 1, end + 1 + GRACE_AFTER_ANOMALY))
    return frozenset(zone)


def score_detection(
    detected: set[int], truth: frozenset[int]
) -> tuple[float, float]:
    """Return (precision, recall).

    Precision: fraction of flagged indices inside the acceptable zone
    (see build_acceptable_zone). Recall: fraction of injected indices that
    were flagged, allowing +/- MATCH_TOLERANCE sample slack.
    """
    if truth:
        matched = sum(
            1
            for t in truth
            if any(abs(t - i) <= MATCH_TOLERANCE for i in detected)
        )
        recall = matched / len(truth)
    else:
        recall = 1.0
    if detected:
        zone = build_acceptable_zone(truth)
        precision = sum(1 for i in detected if i in zone) / len(detected)
    else:
        precision = 1.0
    return precision, recall


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_report(
    config: SensorConfig,
    values: list[float],
    anomalies: dict[int, Anomaly],
    counts: dict[str, int],
    truth: dict[str, frozenset[int]],
) -> None:
    print(f"\n{'=' * 64}")
    print(
        f"Sensor: {config.name}  ({NUM_SAMPLES} samples, "
        f"range {config.min_value}-{config.max_value})"
    )
    print(f"{'=' * 64}")

    print("Detector flags:", end=" ")
    print("  ".join(f"{name}={count}" for name, count in counts.items()))

    detected = set(anomalies)
    all_truth: set[int] = set()
    for kind, indices in truth.items():
        _, recall = score_detection(detected, indices)
        print(f"Injected {kind:<12} ({len(indices):>2} samples) -> recall {recall:.0%}")
        all_truth.update(indices)

    if anomalies:
        precision, recall = score_detection(detected, frozenset(all_truth))
        print(
            f"Overall: {len(anomalies)} flagged | precision {precision:.0%} "
            f"| recall {recall:.0%}"
        )
        print(f"\nFlagged readings (up to {MAX_PRINTED_ANOMALIES} shown):")
        for anomaly in list(anomalies.values())[:MAX_PRINTED_ANOMALIES]:
            methods = ", ".join(anomaly.methods)
            print(
                f"  #{anomaly.index:<4} {anomaly.timestamp:%Y-%m-%d %H:%M} "
                f"value={anomaly.value:8.2f}  severity={anomaly.severity}  "
                f"[{methods}]"
            )
        hidden = len(anomalies) - MAX_PRINTED_ANOMALIES
        if hidden > 0:
            print(f"  ... and {hidden} more")
    else:
        print("No anomalies detected.")


def main() -> None:
    rng = random.Random(SEED)
    for config in SENSOR_CONFIGS:
        clean = generate_series(config, rng)
        injected, truth = inject_anomalies(config, clean)
        anomalies, counts = detect_anomalies(injected, config)
        print_report(config, injected, anomalies, counts, truth)


if __name__ == "__main__":
    main()