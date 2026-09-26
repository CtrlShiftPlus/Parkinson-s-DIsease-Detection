
from pathlib import Path
import json
import warnings

import joblib
import librosa
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf


# ========================================
# 1. PROJECT SETTINGS
# ========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "audio_model"
RECORDINGS_DIR = BASE_DIR / "recordings"

MODEL_PATH = MODEL_DIR / "parkinsons_audio_ensemble2.pkl"
CONFIG_PATH = MODEL_DIR / "audio_model_config2.json"

RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
N_MFCC = 20
RECORD_SECONDS = 7


# ========================================
# 2. LOAD TRAINED MODELS
# ========================================

def load_models():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Extract your audio-model ZIP into the "
            "audio_model folder."
        )

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        bundle = joblib.load(MODEL_PATH)

    required_keys = [
        "random_forest",
        "gradient_boosting",
        "rf_weight",
        "gb_weight",
        "feature_columns"
    ]

    for key in required_keys:
        if key not in bundle:
            raise ValueError(
                f"Missing '{key}' in the saved model."
            )

    if config["feature_columns"] != bundle["feature_columns"]:
        raise ValueError(
            "Model and configuration feature lists differ."
        )

    return bundle, config


# ========================================
# 3. RECORD VOICE
# ========================================

def record_voice():
    print("\n" + "=" * 50)
    print("PARKINSON'S VOICE SCREENING PROTOTYPE")
    print("=" * 50)

    print("\nRecording instructions:")
    print("1. Sit in a quiet room.")
    print("2. Keep your microphone at a fixed distance.")
    print("3. Say 'aaaaah' steadily for 7 seconds.")
    print("4. Avoid speaking other words.")

    input("\nPress ENTER when ready to record...")

    print("\nRecording started. Say 'aaaaah' now!")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    # Convert the recording to a 1D mono signal.
    audio = audio.flatten()

    if np.max(np.abs(audio)) < 0.001:
        raise ValueError(
            "Recording is almost silent. "
            "Check your microphone and try again."
        )

    file_path = RECORDINGS_DIR / "my_voice.wav"

    # Save a standard mono, 16-bit PCM WAV file.
    sf.write(
        str(file_path),
        audio,
        SAMPLE_RATE,
        subtype="PCM_16"
    )

    print("\nRecording complete!")
    print("Saved to:", file_path)

    return file_path


# ========================================
# 4. AUDIO FEATURE EXTRACTION
# Must match the Colab training code.
# ========================================

def extract_audio_features(file_path):
    audio, sr = librosa.load(
        str(file_path),
        sr=SAMPLE_RATE,
        mono=True
    )

    audio, _ = librosa.effects.trim(
        audio,
        top_db=30
    )

    if len(audio) < sr:
        raise ValueError(
            "Recording is too short after silence removal. "
            "Please record again."
        )

    features = {}

    def add_summary(name, values):
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]

        features[name + "_mean"] = (
            float(np.mean(finite))
            if len(finite) else np.nan
        )

        features[name + "_std"] = (
            float(np.std(finite))
            if len(finite) else np.nan
        )

    # MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    for i in range(N_MFCC):
        add_summary(
            f"mfcc_{i + 1}",
            mfcc[i]
        )

    # Delta MFCC features
    delta = librosa.feature.delta(mfcc)

    for i in range(N_MFCC):
        add_summary(
            f"delta_mfcc_{i + 1}",
            delta[i]
        )

    # Energy
    add_summary(
        "rms",
        librosa.feature.rms(y=audio)[0]
    )

    # Zero-crossing rate
    add_summary(
        "zero_crossing_rate",
        librosa.feature.zero_crossing_rate(
            y=audio
        )[0]
    )

    # Spectral centroid
    add_summary(
        "spectral_centroid",
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )[0]
    )

    # Spectral bandwidth
    add_summary(
        "spectral_bandwidth",
        librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr
        )[0]
    )

    # Spectral rolloff
    add_summary(
        "spectral_rolloff",
        librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr
        )[0]
    )

    # Spectral flatness
    add_summary(
        "spectral_flatness",
        librosa.feature.spectral_flatness(
            y=audio
        )[0]
    )

    # Pitch estimation
    pitch, voiced_flag, _ = librosa.pyin(
        audio,
        sr=sr,
        fmin=65,
        fmax=600
    )

    add_summary(
        "pitch",
        pitch
    )

    features["voiced_fraction"] = float(
        np.mean(voiced_flag)
    )

    return features


# ========================================
# 5. RUN MODEL PREDICTION
# ========================================

def predict_voice(file_path, bundle):
    print("\nExtracting voice features...")
    features = extract_audio_features(file_path)

    expected_columns = bundle["feature_columns"]

    missing = [
        column for column in expected_columns
        if column not in features
    ]

    if missing:
        raise ValueError(
            "Missing features: " + str(missing)
        )

    X = pd.DataFrame(
        [features]
    ).reindex(columns=expected_columns)

    print("Running Random Forest...")
    rf_score = bundle[
        "random_forest"
    ].predict_proba(X)[0, 1]

    print("Running Gradient Boosting...")
    gb_score = bundle[
        "gradient_boosting"
    ].predict_proba(X)[0, 1]

    rf_weight = bundle["rf_weight"]
    gb_weight = bundle["gb_weight"]

    combined_score = (
        rf_weight * rf_score
        + gb_weight * gb_score
    )

    threshold = bundle.get(
        "threshold",
        0.5
    )

    predicted_class = int(
        combined_score >= threshold
    )

    predicted_label = (
        "Parkinson's dataset class"
        if predicted_class == 1
        else "Healthy-control dataset class"
    )

    result = {
        "random_forest_score": float(rf_score),
        "gradient_boosting_score": float(gb_score),
        "combined_score": float(combined_score),
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "recording": str(file_path)
    }

    return result


# ========================================
# 6. DISPLAY RESULTS
# ========================================

def display_results(result):
    print("\n" + "=" * 50)
    print("VOICE MODEL RESULTS")
    print("=" * 50)

    print(
        "\nRandom Forest score:",
        f"{result['random_forest_score'] * 100:.2f}%"
    )

    print(
        "Gradient Boosting score:",
        f"{result['gradient_boosting_score'] * 100:.2f}%"
    )

    print(
        "\nCombined model score:",
        f"{result['combined_score'] * 100:.2f}%"
    )

    print(
        "\nPredicted dataset class:",
        result["predicted_label"]
    )

    print(
        "\nNote: These are experimental model outputs, "
        "not a medical diagnosis or a validated "
        "probability of Parkinson's disease."
    )

    print("=" * 50)


# ========================================
# 7. MAIN PROGRAM
# ========================================

def main():
    print("Loading your trained audio models...")

    bundle, config = load_models()

    global SAMPLE_RATE, N_MFCC

    SAMPLE_RATE = config["sample_rate"]
    N_MFCC = config["n_mfcc"]

    print("Models loaded successfully.")
    print("Sample rate:", SAMPLE_RATE)
    print("Number of MFCCs:", N_MFCC)

    while True:
        print("\nChoose an option:")
        print("1. Record a new voice sample")
        print("2. Test an existing WAV file")
        print("3. Exit")

        choice = input("\nEnter 1, 2 or 3: ").strip()

        if choice == "3":
            print("Goodbye!")
            break

        try:
            if choice == "1":
                file_path = record_voice()

            elif choice == "2":
                path_text = input(
                    "Enter the WAV file path: "
                ).strip().strip('"')

                file_path = Path(path_text)

                if not file_path.is_file():
                    print("File not found:", file_path)
                    continue

            else:
                print("Invalid choice.")
                continue

            result = predict_voice(
                file_path,
                bundle
            )

            display_results(result)

            # Save the latest result as JSON.
            result_path = (
                RECORDINGS_DIR / "latest_result.json"
            )

            with open(
                result_path,
                "w",
                encoding="utf-8"
            ) as file:
                json.dump(
                    result,
                    file,
                    indent=2
                )

            print("\nResult saved to:", result_path)

        except Exception as error:
            print("\nAn error occurred:")
            print(error)


if __name__ == "__main__":
    main()