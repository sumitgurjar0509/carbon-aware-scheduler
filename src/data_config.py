"""
Which data to use: the fetch window and the dataset tag
=======================================================
One place for the rules that decide which data every script works on.

Environment variables
---------------------
CAS_START, CAS_END  (optional, used only by data_pipeline.py)
    The window to fetch, as ISO 8601 times, e.g. 2026-09-23T03:00Z.
    A time without a timezone is read as UTC.
    Neither set  -> the last 365 days up to the current hour (as before).
    Only CAS_END -> start is 365 days before CAS_END.
    Only CAS_START -> end is the current hour.

CAS_DATA_TAG  (required by every script that reads processed data)
    e.g. 20250923_20260923 -> data/processed/clean_20250923_20260923.csv
    There is deliberately NO "newest file" fallback: picking files by
    modification time once made a run silently use the wrong dataset.

MODEL LABEL
-----------
Trained models are saved to one shared folder. After training, a small
file (data_tag.json) records which dataset and which training stage
produced them, so the simulator can refuse models that don't match.
"""
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

START_ENV = "CAS_START"
END_ENV = "CAS_END"
TAG_ENV = "CAS_DATA_TAG"
DEFAULT_WINDOW_DAYS = 365
TAG_PATTERN = re.compile(r"\d{8}_\d{8}")
MODEL_LABEL_NAME = "data_tag.json"


class ConfigError(SystemExit):
    """Stops a script with a readable message instead of a long traceback."""


# ----------------------------------------------------------------------
# Fetch window
# ----------------------------------------------------------------------
def _parse_utc(name: str, text: str) -> datetime:
    try:
        value = datetime.fromisoformat(text.strip())
    except ValueError:
        raise ConfigError(
            f"{name}={text!r} is not a valid time. Use ISO 8601, e.g. 2026-09-23T03:00Z"
        ) from None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def get_window(env=None, now: datetime | None = None) -> tuple[datetime, datetime]:
    """Returns (start, end) of the fetch window as UTC datetimes."""
    env = os.environ if env is None else env
    now = now or datetime.now(timezone.utc)
    start_text = env.get(START_ENV, "").strip()
    end_text = env.get(END_ENV, "").strip()

    if end_text:
        end = _parse_utc(END_ENV, end_text)
    else:
        end = now.replace(minute=0, second=0, microsecond=0)

    if start_text:
        start = _parse_utc(START_ENV, start_text)
    else:
        start = end - timedelta(days=DEFAULT_WINDOW_DAYS)

    if start >= end:
        raise ConfigError(f"The window start ({start}) must be before its end ({end}). "
                          f"Check {START_ENV} / {END_ENV}.")
    return start, end


# ----------------------------------------------------------------------
# Dataset tag and the files that belong to it
# ----------------------------------------------------------------------
def list_tags(processed_dir: Path = PROCESSED_DIR) -> list[str]:
    """Tags for which a cleaned dataset exists on disk."""
    return sorted(p.stem.removeprefix("clean_") for p in processed_dir.glob("clean_*.csv"))


def get_data_tag(env=None, processed_dir: Path = PROCESSED_DIR) -> str:
    """Reads CAS_DATA_TAG; stops with a clear message if unset or malformed."""
    env = os.environ if env is None else env
    tag = env.get(TAG_ENV, "").strip()
    available = ", ".join(list_tags(processed_dir)) or "none (run src/data_pipeline.py first)"
    if not tag:
        raise ConfigError(
            f"{TAG_ENV} is not set, so I don't know which dataset to use.\n"
            f"Available tags: {available}\n"
            f'PowerShell:  $env:{TAG_ENV} = "20250923_20260923"'
        )
    if not TAG_PATTERN.fullmatch(tag):
        raise ConfigError(f"{TAG_ENV}={tag!r} should look like YYYYMMDD_YYYYMMDD. "
                          f"Available tags: {available}")
    return tag


def _existing_file(prefix: str, tag: str | None, processed_dir: Path, hint: str) -> Path:
    tag = tag or get_data_tag(processed_dir=processed_dir)
    path = processed_dir / f"{prefix}_{tag}.csv"
    if not path.exists():
        available = ", ".join(list_tags(processed_dir)) or "none"
        raise ConfigError(f"File not found: {path}\n{hint}\nAvailable tags: {available}")
    return path


def clean_data_path(tag: str | None = None, processed_dir: Path = PROCESSED_DIR) -> Path:
    return _existing_file("clean", tag, processed_dir,
                          f"Check {TAG_ENV}, or build it with src/data_pipeline.py.")


def gap_report_path(tag: str | None = None, processed_dir: Path = PROCESSED_DIR) -> Path:
    return _existing_file("gap_report", tag, processed_dir,
                          f"Check {TAG_ENV}, or build it with src/data_pipeline.py.")


def workload_path(tag: str | None = None, processed_dir: Path = PROCESSED_DIR) -> Path:
    """Where the workload trace for a tag lives (it may not exist yet)."""
    tag = tag or get_data_tag(processed_dir=processed_dir)
    return processed_dir / f"workload_trace_{tag}.csv"


def existing_workload_path(tag: str | None = None,
                           processed_dir: Path = PROCESSED_DIR) -> Path:
    return _existing_file("workload_trace", tag, processed_dir,
                          f"Run src/workload_generator.py with the same {TAG_ENV}.")


# ----------------------------------------------------------------------
# Model label
# ----------------------------------------------------------------------
def clear_model_label(models_dir: Path):
    """Call BEFORE overwriting models, so a half-finished run is never trusted."""
    (models_dir / MODEL_LABEL_NAME).unlink(missing_ok=True)


def write_model_label(models_dir: Path, tag: str, stage: str):
    """Call AFTER training finished. stage is 'forecasting' or 'calibration'."""
    models_dir.mkdir(parents=True, exist_ok=True)
    label = {"data_tag": tag, "stage": stage}
    (models_dir / MODEL_LABEL_NAME).write_text(json.dumps(label), encoding="utf-8")


def check_model_label(models_dir: Path, tag: str):
    """Stops unless the saved models are calibrated and belong to `tag`."""
    path = models_dir / MODEL_LABEL_NAME
    fix = f"Run src/calibration.py with {TAG_ENV}={tag}."
    if not path.exists():
        raise ConfigError(f"The models in {models_dir} have no dataset label "
                          f"(old or unfinished training). {fix}")
    label = json.loads(path.read_text(encoding="utf-8"))
    if label.get("data_tag") != tag:
        raise ConfigError(f"The saved models were trained on dataset {label.get('data_tag')}, "
                          f"but {TAG_ENV}={tag}. {fix}")
    if label.get("stage") != "calibration":
        raise ConfigError(f"The saved models come from '{label.get('stage')}' and are not "
                          f"calibrated. {fix}")
