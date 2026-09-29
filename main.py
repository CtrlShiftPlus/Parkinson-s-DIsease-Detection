"""
===============================================================================
Parkinson's Disease ML Multi-Modal Pipeline Integration
===============================================================================
Integrates two independently trained Parkinson's Machine Learning models:
1. Voice Model: Predicts Parkinson's risk from audio speech feature extraction
   (Librosa acoustic summary stats -> Random Forest + Gradient Boosting ensemble).
2. Alcohol-Effect Model: Evaluates Parkinson's association based on Age &
   Alcohol Consumption (NIH PDBP Study -> StandardScaler + Logistic Regression).
===============================================================================
"""

import sys
import os
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import librosa

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONSTANTS AND PATH CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

# Voice Model Paths
VOICE_MODEL_DIR = BASE_DIR / "audio_model"
VOICE_MODEL_PATH = VOICE_MODEL_DIR / "parkinsons_audio_ensemble2.pkl"
VOICE_CONFIG_PATH = VOICE_MODEL_DIR / "audio_model_config2.json"

# Alcohol-Effect Model Paths (Located on D: drive)
ALCOHOL_MODEL_DIR = Path(r"D:\Major project Datasets")
ALCOHOL_MODEL_PATH = ALCOHOL_MODEL_DIR / "model.pkl"
ALCOHOL_SCALER_PATH = ALCOHOL_MODEL_DIR / "scaler.pkl"

DEFAULT_SAMPLE_WAV = BASE_DIR / "recordings" / "my_voice.wav"

# Default extraction parameters for voice
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_MFCC = 20


def fix_sklearn_compat(model_obj):
    """Compatibility patch for scikit-learn unpickling differences across versions."""
    if hasattr(model_obj, "named_steps"):
        for step in model_obj.named_steps.values():
            fix_sklearn_compat(step)
    if hasattr(model_obj, "_fit_dtype") and not hasattr(model_obj, "_fill_dtype"):
        setattr(model_obj, "_fill_dtype", getattr(model_obj, "_fit_dtype"))

# =============================================================================
# 2. VOICE MODEL PIPELINE
# =============================================================================

class VoiceModelRunner:
    def __init__(self, model_path=VOICE_MODEL_PATH, config_path=VOICE_CONFIG_PATH):
        self.model_path = model_path
        self.config_path = config_path
        self.bundle = None
        self.config = None
        self.load_artifacts()

    def load_artifacts(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Voice model file missing: {self.model_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Voice config file missing: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.bundle = joblib.load(self.model_path)
        
        required_keys = ["random_forest", "gradient_boosting", "rf_weight", "gb_weight", "feature_columns"]
        for k in required_keys:
            if k not in self.bundle:
                raise ValueError(f"Missing key '{k}' in voice model bundle.")

        fix_sklearn_compat(self.bundle["random_forest"])
        fix_sklearn_compat(self.bundle["gradient_boosting"])

    def extract_features(self, audio_path):
        sample_rate = self.config.get("sample_rate", DEFAULT_SAMPLE_RATE)
        n_mfcc = self.config.get("n_mfcc", DEFAULT_N_MFCC)

        audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        audio, _ = librosa.effects.trim(audio, top_db=30)

        if len(audio) < sr:
            raise ValueError("Audio recording is too short after silence removal (minimum 1 second required).")

        features = {}

        def add_summary(name, values):
            vals = np.asarray(values, dtype=float)
            finite = vals[np.isfinite(vals)]
            features[name + "_mean"] = float(np.mean(finite)) if len(finite) else np.nan
            features[name + "_std"] = float(np.std(finite)) if len(finite) else np.nan

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        for i in range(n_mfcc):
            add_summary(f"mfcc_{i + 1}", mfcc[i])

        delta = librosa.feature.delta(mfcc)
        for i in range(n_mfcc):
            add_summary(f"delta_mfcc_{i + 1}", delta[i])

        add_summary("rms", librosa.feature.rms(y=audio)[0])
        add_summary("zero_crossing_rate", librosa.feature.zero_crossing_rate(y=audio)[0])
        add_summary("spectral_centroid", librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
        add_summary("spectral_bandwidth", librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0])
        add_summary("spectral_rolloff", librosa.feature.spectral_rolloff(y=audio, sr=sr)[0])
        add_summary("spectral_flatness", librosa.feature.spectral_flatness(y=audio)[0])

        pitch, voiced_flag, _ = librosa.pyin(audio, sr=sr, fmin=65, fmax=600)
        add_summary("pitch", pitch)
        features["voiced_fraction"] = float(np.mean(voiced_flag))

        return features

    def predict(self, audio_path):
        features = self.extract_features(audio_path)
        expected_cols = self.bundle["feature_columns"]

        missing = [c for c in expected_cols if c not in features]
        if missing:
            raise ValueError(f"Missing extracted voice features: {missing}")

        X = pd.DataFrame([features]).reindex(columns=expected_cols)

        rf_prob = float(self.bundle["random_forest"].predict_proba(X)[0, 1])
        gb_prob = float(self.bundle["gradient_boosting"].predict_proba(X)[0, 1])

        rf_weight = self.bundle["rf_weight"]
        gb_weight = self.bundle["gb_weight"]
        combined_prob = (rf_weight * rf_prob) + (gb_weight * gb_prob)

        threshold = self.config.get("threshold", 0.5)
        pred_class = int(combined_prob >= threshold)
        pred_label = "Parkinson's (Acoustic Pattern)" if pred_class == 1 else "Healthy / Control (Acoustic Pattern)"

        return {
            "rf_score": rf_prob,
            "gb_score": gb_prob,
            "combined_score": combined_prob,
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "audio_file": str(audio_path)
        }

# =============================================================================
# 3. ALCOHOL-EFFECT MODEL PIPELINE
# =============================================================================

class AlcoholModelRunner:
    def __init__(self, model_path=ALCOHOL_MODEL_PATH, scaler_path=ALCOHOL_SCALER_PATH):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.load_artifacts()

    def load_artifacts(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Alcohol model file missing: {self.model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Alcohol scaler file missing: {self.scaler_path}")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        fix_sklearn_compat(self.model)
        fix_sklearn_compat(self.scaler)

    def predict(self, age, avg_drinks):
        # Input features expected: ['AgeYrs', 'AlcDrinkingDayAvgDrinks']
        input_df = pd.DataFrame([[age, avg_drinks]], columns=['AgeYrs', 'AlcDrinkingDayAvgDrinks'])
        input_scaled = self.scaler.transform(input_df)

        prob = float(self.model.predict_proba(input_scaled)[0, 1]) * 100.0

        if prob < 35.0:
            risk_level = "LOW"
        elif prob <= 65.0:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        pred_class = int(prob >= 50.0)
        pred_label = "Elevated Behavioral-Demographic Risk" if pred_class == 1 else "Standard Behavioral-Demographic Risk"

        return {
            "age": age,
            "avg_drinks": avg_drinks,
            "probability_pct": prob,
            "risk_level": risk_level,
            "predicted_class": pred_class,
            "predicted_label": pred_label
        }

# =============================================================================
# 4. INPUT VALIDATION HELPERS
# =============================================================================

def parse_age(age_input):
    try:
        val = float(age_input)
        if not (18 <= val <= 120):
            raise ValueError
        return val
    except ValueError:
        raise ValueError("Age must be a valid number between 18 and 120 years.")

def parse_alcohol(alcohol_input):
    try:
        val = float(alcohol_input)
        if val < 0 or val > 25:
            raise ValueError
        return val
    except ValueError:
        raise ValueError("Alcohol drinks/day must be a non-negative number (0 - 25).")

def validate_audio_file(file_path):
    path = Path(file_path.strip('"\''))
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() not in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
        raise ValueError("File must be a supported audio format (.wav, .mp3, .flac, .ogg, .m4a)")
    return path

# =============================================================================
# 5. CLI & PRESENTATION INTERFACE
# =============================================================================

def run_pipeline(voice_runner, alcohol_runner, age, alcohol_val, audio_path):
    print("\n" + "=" * 55)
    print("      RUNNING INTEGRATED PARKINSON'S PIPELINE")
    print("=" * 55)

    print("\n[Step 1/2] Processing Voice Model...")
    voice_res = voice_runner.predict(audio_path)
    print(f" -> Acoustic Features Extracted from: {audio_path.name}")
    print(f" -> Random Forest Score:       {voice_res['rf_score'] * 100:.2f}%")
    print(f" -> Gradient Boosting Score:   {voice_res['gb_score'] * 100:.2f}%")
    print(f" -> Voice Ensemble Prediction: {voice_res['combined_score'] * 100:.2f}% ({voice_res['predicted_label']})")

    print("\n[Step 2/2] Processing Alcohol-Effect Model...")
    alcohol_res = alcohol_runner.predict(age, alcohol_val)
    print(f" -> Age:                       {alcohol_res['age']} years")
    print(f" -> Avg Drinks per Day:        {alcohol_res['avg_drinks']}")
    print(f" -> Demographic Study Risk:    {alcohol_res['probability_pct']:.2f}% ({alcohol_res['risk_level']} RISK)")

    print("\n" + "=" * 55)
    print("               FINAL PIPELINE RESULTS")
    print("=" * 55)
    print(f"Voice Model Classification:    {voice_res['predicted_label']}")
    print(f"  * Voice Ensemble Confidence: {voice_res['combined_score'] * 100:.2f}%")
    print(f"  * Audio File Evaluated:      {voice_res['audio_file']}")
    print("-" * 55)
    print(f"Alcohol-Effect Model Risk:     {alcohol_res['risk_level']} RISK ({alcohol_res['probability_pct']:.2f}%)")
    print(f"  * Age Evaluated:             {alcohol_res['age']} years")
    print(f"  * Alcohol Daily Consumption: {alcohol_res['avg_drinks']} drinks/day")
    print("=" * 55)

    print("\n" + "-" * 55)
    print("       SCIENTIFIC MODEL RELATIONSHIP INTERPRETATION")
    print("-" * 55)
    print("1. Voice Model:")
    print("   - Evaluates vocal acoustic biomarkers (MFCCs, jitter/shimmer proxies,")
    print("     pitch stability, spectral flatness) from recorded speech.")
    print("   - Measures active neuro-motor articulation patterns.")
    print("2. Alcohol-Effect Model:")
    print("   - Evaluates statistical demographic risk based on participant age and")
    print("     alcohol consumption metrics from the NIH PDBP study dataset.")
    print("3. Pipeline Integration Note:")
    print("   - The outputs are kept distinct because they assess two separate")
    print("     dimensions: physiological vocal signal vs epidemiological history.")
    print("   - Combining them into a single merged probability is statistically")
    print("     unjustified as they were trained on independent dataset populations.")
    print("=" * 55 + "\n")

    return voice_res, alcohol_res


def interactive_cli(voice_runner, alcohol_runner):
    print("=" * 50)
    print("        Parkinson's Disease ML Demo")
    print("========================================")
    print("\nEnter patient information:")

    # Age input
    while True:
        try:
            raw_age = input("Age (18-120): ").strip()
            age = parse_age(raw_age)
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    # Alcohol input
    while True:
        try:
            raw_alc = input("Alcohol-related input (Average drinks/day, e.g. 0-10): ").strip()
            alcohol_val = parse_alcohol(raw_alc)
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    # Voice input
    default_str = f" [Press Enter for default: {DEFAULT_SAMPLE_WAV.name}]" if DEFAULT_SAMPLE_WAV.exists() else ""
    while True:
        try:
            raw_audio = input(f"Voice/audio file path{default_str}: ").strip()
            if not raw_audio and DEFAULT_SAMPLE_WAV.exists():
                audio_path = DEFAULT_SAMPLE_WAV
            else:
                audio_path = validate_audio_file(raw_audio)
            break
        except Exception as e:
            print(f"Invalid input: {e}")

    run_pipeline(voice_runner, alcohol_runner, age, alcohol_val, audio_path)


def main():
    parser = argparse.ArgumentParser(description="Parkinson's Disease Multi-Model ML Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run automated demonstration mode with sample inputs")
    parser.add_argument("--age", type=float, help="Patient age in years")
    parser.add_argument("--alcohol", type=float, help="Average alcoholic drinks per day")
    parser.add_argument("--audio", type=str, help="Path to audio file (.wav)")
    args = parser.parse_args()

    print("Initializing Multi-Modal Pipeline...")
    voice_runner = VoiceModelRunner()
    alcohol_runner = AlcoholModelRunner()
    print("Models & Scalers loaded successfully!\n")

    if args.demo:
        age = 65.0
        alcohol_val = 2.0
        audio_path = DEFAULT_SAMPLE_WAV
        if not audio_path.exists():
            print(f"Error: Sample audio file missing at {audio_path}")
            sys.exit(1)
        print(">>> Running non-interactive demo mode...")
        print(f"Input Age: {age} | Input Alcohol: {alcohol_val} drinks/day | Audio: {audio_path.name}")
        run_pipeline(voice_runner, alcohol_runner, age, alcohol_val, audio_path)
    elif args.age is not None and args.alcohol is not None and args.audio is not None:
        age = parse_age(args.age)
        alcohol_val = parse_alcohol(args.alcohol)
        audio_path = validate_audio_file(args.audio)
        run_pipeline(voice_runner, alcohol_runner, age, alcohol_val, audio_path)
    else:
        interactive_cli(voice_runner, alcohol_runner)


if __name__ == "__main__":
    main()
