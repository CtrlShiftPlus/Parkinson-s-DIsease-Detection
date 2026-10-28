"""
===============================================================================
Tremor Model Runner - Parkinson's Biomechanical & Inertial Sensor Analysis
===============================================================================
Wraps the 5-fold ensemble of ExtraTrees, RandomForest, and HistGradientBoosting
classifiers trained on tri-axial accelerometer, gyroscope, and magnetometer signals.
Matches the feature extraction, windowing, and time-blocked validation of the
original add-tremor-detection branch.
===============================================================================
"""

import io
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List

import joblib
import numpy as np
import pandas as pd

from tremor_features import (
    SENSOR_COLUMNS,
    WINDOW,
    STRIDE,
    SEGMENT_LEN,
    window_features,
    segment_windows,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TREMOR_MODEL_PATH = BASE_DIR / "tremor_model" / "tremor_ensemble.pkl"
DEFAULT_TREMOR_CONFIG_PATH = BASE_DIR / "tremor_model" / "tremor_model_config.json"
DEFAULT_TREMOR_CSV_PATH = BASE_DIR / "tremor_data" / "Dataset.csv"


class TremorModelRunner:
    def __init__(
        self,
        model_path: Path = DEFAULT_TREMOR_MODEL_PATH,
        config_path: Path = DEFAULT_TREMOR_CONFIG_PATH,
        csv_path: Path = DEFAULT_TREMOR_CSV_PATH,
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.csv_path = Path(csv_path)

        self.bundle: Optional[Dict[str, Any]] = None
        self.config: Optional[Dict[str, Any]] = None
        self.data: Optional[pd.DataFrame] = None
        self.n_segments: int = 111
        self.threshold: float = 0.5

        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Loads model ensemble, configuration, and reference benchmark dataset."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Tremor model ensemble file missing: {self.model_path}. "
                "Ensure tremor_model/tremor_ensemble.pkl is present."
            )

        # Load model bundle (suppress unpickle version warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.bundle = joblib.load(self.model_path)

        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            except Exception:
                self.config = {}
        else:
            self.config = {}

        if self.csv_path.exists():
            try:
                self.data = pd.read_csv(self.csv_path)
            except Exception as e:
                print(f"Warning: Could not read tremor benchmark dataset: {e}")
                self.data = None
        else:
            self.data = None

        self.n_segments = self.bundle.get("n_segments", 111)
        self.threshold = float(self.bundle.get("threshold", 0.5))

    def get_segment_data(self, number: int) -> Tuple[np.ndarray, float, int, int]:
        """Extract sensor rows and true label from the benchmark dataset for segment `number`."""
        if self.data is None:
            raise FileNotFoundError(f"Benchmark dataset is missing at {self.csv_path}")

        if not (1 <= number <= self.n_segments):
            raise ValueError(
                f"Segment number must be between 1 and {self.n_segments} (received {number})"
            )

        seg_len = self.bundle.get("segment_len", SEGMENT_LEN)
        start = (number - 1) * seg_len
        end = start + seg_len

        rows = self.data.iloc[start:end]
        missing_cols = [c for c in SENSOR_COLUMNS if c not in rows.columns]
        if missing_cols:
            raise ValueError(f"Benchmark dataset is missing required sensor columns: {missing_cols}")

        sensors = rows[SENSOR_COLUMNS].values.astype(float)
        true_fraction = float(rows["Result"].mean()) if "Result" in rows.columns else 0.5

        return sensors, true_fraction, start, end

    def predict_segment(self, number: int) -> Dict[str, Any]:
        """
        Run inference on segment `number` from the benchmark dataset.
        Uses the out-of-sample cross-validation fold model corresponding to this
        segment to strictly ensure honest, non-memorized test inference.
        """
        sensors, true_fraction, start, end = self.get_segment_data(number)
        features = segment_windows(sensors)

        if len(features) == 0:
            raise ValueError(f"Insufficient sensor windows in segment {number}")

        edges = self.bundle["seg_edges"]
        fold = 0
        for k in range(len(edges) - 1):
            if edges[k] <= number - 1 < edges[k + 1]:
                fold = k
                break

        models = self.bundle["fold_models"][fold]

        et_score = float(np.mean(models["extra_trees"].predict_proba(features)[:, 1]))
        rf_score = float(np.mean(models["random_forest"].predict_proba(features)[:, 1]))
        gb_score = float(np.mean(models["hist_gb"].predict_proba(features)[:, 1]))

        combined = (et_score + rf_score + gb_score) / 3.0
        predicted_class = int(combined >= self.threshold)
        predicted_label = (
            "Parkinson's Tremor Pattern"
            if predicted_class == 1
            else "Healthy / Control (No Tremor)"
        )
        true_class = int(true_fraction >= 0.5)

        return {
            "dataset_number": number,
            "rows": f"{start + 1}-{end}",
            "source": f"Benchmark Segment #{number} (Rows {start + 1}–{end})",
            "windows_analysed": int(len(features)),
            "extra_trees_score": et_score,
            "random_forest_score": rf_score,
            "gradient_boosting_score": gb_score,
            "combined_score": combined,
            "threshold": self.threshold,
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "dataset_true_label": (
                "Parkinson's Tremor Pattern" if true_class == 1
                else "Healthy / Control (No Tremor)"
            ),
            "dataset_label_purity": float(max(true_fraction, 1 - true_fraction)),
            "match": predicted_class == true_class,
        }

    def predict_sensor_data(
        self, sensor_array: np.ndarray, source_name: str = "Uploaded Sensor Stream"
    ) -> Dict[str, Any]:
        """
        Run inference on an arbitrary sensor array of shape (N, 7).
        Ensembles across all cross-validation fold models for robust inference on novel data.
        """
        sensor_array = np.asarray(sensor_array, dtype=float)
        if sensor_array.ndim != 2 or sensor_array.shape[1] < len(SENSOR_COLUMNS):
            raise ValueError(
                f"Sensor data must have 2 dimensions with at least {len(SENSOR_COLUMNS)} columns ({SENSOR_COLUMNS}). "
                f"Received shape: {sensor_array.shape}"
            )

        if len(sensor_array) < WINDOW:
            raise ValueError(
                f"Sensor recording too short: must have at least {WINDOW} samples (received {len(sensor_array)})."
            )

        # Slice precisely to standard 7 sensor columns if extra columns exist
        sensor_subset = sensor_array[:, :len(SENSOR_COLUMNS)]
        features = segment_windows(sensor_subset)

        if len(features) == 0:
            raise ValueError(f"Could not construct valid analysis windows from sensor data.")

        # Ensemble across all K folds
        all_et_scores = []
        all_rf_scores = []
        all_gb_scores = []

        for models in self.bundle["fold_models"]:
            all_et_scores.append(float(np.mean(models["extra_trees"].predict_proba(features)[:, 1])))
            all_rf_scores.append(float(np.mean(models["random_forest"].predict_proba(features)[:, 1])))
            all_gb_scores.append(float(np.mean(models["hist_gb"].predict_proba(features)[:, 1])))

        et_score = float(np.mean(all_et_scores))
        rf_score = float(np.mean(all_rf_scores))
        gb_score = float(np.mean(all_gb_scores))

        combined = (et_score + rf_score + gb_score) / 3.0
        predicted_class = int(combined >= self.threshold)
        predicted_label = (
            "Parkinson's Tremor Pattern"
            if predicted_class == 1
            else "Healthy / Control (No Tremor)"
        )

        return {
            "dataset_number": None,
            "rows": f"1–{len(sensor_array)}",
            "source": source_name,
            "windows_analysed": int(len(features)),
            "extra_trees_score": et_score,
            "random_forest_score": rf_score,
            "gradient_boosting_score": gb_score,
            "combined_score": combined,
            "threshold": self.threshold,
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "dataset_true_label": "N/A (External Sensor Stream)",
            "dataset_label_purity": None,
            "match": None,
        }

    def predict_csv(
        self, csv_file_or_path: Union[str, Path, io.IOBase], source_name: str = "Uploaded CSV File"
    ) -> Dict[str, Any]:
        """Reads CSV sensor file and produces inference results."""
        df = pd.read_csv(csv_file_or_path)

        # Check column matching (case-insensitive fallback)
        df_cols_lower = {c.lower(): c for c in df.columns}
        extracted_data = {}

        for col in SENSOR_COLUMNS:
            if col in df.columns:
                extracted_data[col] = df[col].values
            elif col.lower() in df_cols_lower:
                extracted_data[col] = df[df_cols_lower[col.lower()]].values
            elif col == "mX":
                # Magnetometer channel fallback if absent
                extracted_data[col] = np.zeros(len(df), dtype=float)
            else:
                raise ValueError(
                    f"Uploaded sensor CSV missing required channel: '{col}'. "
                    f"Expected columns: {SENSOR_COLUMNS}"
                )

        sensor_array = np.column_stack([extracted_data[c] for c in SENSOR_COLUMNS]).astype(float)
        result = self.predict_sensor_data(sensor_array, source_name=source_name)

        if "Result" in df.columns or "result" in df_cols_lower:
            res_col = "Result" if "Result" in df.columns else df_cols_lower["result"]
            true_fraction = float(df[res_col].mean())
            true_class = int(true_fraction >= 0.5)
            result["dataset_true_label"] = (
                "Parkinson's Tremor Pattern" if true_class == 1 else "Healthy / Control (No Tremor)"
            )
            result["dataset_label_purity"] = float(max(true_fraction, 1 - true_fraction))
            result["match"] = result["predicted_class"] == true_class

        return result

    def predict(
        self,
        segment_number: Optional[int] = None,
        csv_path: Optional[Union[str, Path]] = None,
        sensor_data: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Convenient generic prediction entrypoint."""
        if csv_path is not None:
            return self.predict_csv(csv_path, source_name=Path(csv_path).name)
        elif sensor_data is not None:
            return self.predict_sensor_data(sensor_data)
        else:
            seg = segment_number if segment_number is not None else 1
            return self.predict_segment(seg)

    def get_segments_catalog(self) -> Dict[str, Any]:
        """Returns catalog of benchmark segments for frontend picker/presets."""
        sample_parkinsons = [1, 2, 3, 5, 10]
        sample_healthy = [12, 15, 16, 18, 20]

        return {
            "total_segments": self.n_segments,
            "segment_len": self.bundle.get("segment_len", SEGMENT_LEN),
            "window": self.bundle.get("window", WINDOW),
            "sensor_columns": SENSOR_COLUMNS,
            "metrics": self.bundle.get("metrics", {}),
            "presets": [
                {"id": 1, "name": "Segment #1 (Parkinson's Tremor Preset)", "class": "Parkinson's", "purity": "96.4%"},
                {"id": 12, "name": "Segment #12 (Healthy / Control Preset)", "class": "Healthy", "purity": "100.0%"},
                {"id": 15, "name": "Segment #15 (Healthy / Control Preset)", "class": "Healthy", "purity": "100.0%"},
                {"id": 5, "name": "Segment #5 (Parkinson's Tremor Preset)", "class": "Parkinson's", "purity": "98.8%"},
            ]
        }
