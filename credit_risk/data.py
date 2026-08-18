from __future__ import annotations

import math
import zipfile
from pathlib import Path
from typing import List, Tuple
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np

UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"
UCI_DATA_FILE = "5year.arff"


def _download_and_extract(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    arff_path = data_dir / UCI_DATA_FILE
    if arff_path.exists():
        return arff_path

    zip_path = data_dir / "polish-bankruptcy.zip"
    try:
        urlretrieve(UCI_DATA_URL, zip_path)
    except URLError as exc:
        raise RuntimeError(
            "Unable to download the UCI dataset in this environment. "
            "Provide a local path to 5year.arff via load_polish_bankruptcy_five_year(data_path=...)."
        ) from exc
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extract(UCI_DATA_FILE, path=data_dir)
    return arff_path


def _parse_arff(arff_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_names: List[str] = []
    rows: List[List[float]] = []
    targets: List[int] = []
    in_data = False

    with arff_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            lowered = line.lower()
            if lowered.startswith("@attribute") and not in_data:
                parts = line.split()
                if len(parts) >= 2 and parts[1].lower() != "class":
                    feature_names.append(parts[1])
                continue
            if lowered.startswith("@data"):
                in_data = True
                continue
            if not in_data:
                continue

            values = [value.strip() for value in line.split(",")]
            if len(values) < 2:
                continue
            rows.append([math.nan if value == "?" else float(value) for value in values[:-1]])
            targets.append(int(values[-1]))

    return np.array(rows, dtype=float), np.array(targets, dtype=int), feature_names


def load_polish_bankruptcy_five_year(
    data_dir: str | Path = "data",
    data_path: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the five-year ARFF subset from a path or the local data cache."""
    arff_path = Path(data_path) if data_path else _download_and_extract(Path(data_dir))
    return _parse_arff(arff_path)
