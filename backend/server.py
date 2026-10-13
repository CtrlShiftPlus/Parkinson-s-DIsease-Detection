import os
import sys
import shutil
import uuid
import datetime
from pathlib import Path
from typing import Optional, List
import soundfile as sf
import numpy as np

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from main import VoiceModelRunner, AlcoholModelRunner, DEFAULT_SAMPLE_WAV
from tremor_runner import TremorModelRunner
from backend.database import init_db, get_db, User, AnalysisRecord, Notification
from backend.auth import hash_password, verify_password, create_access_token, get_current_user, get_doctor_user
from backend.alert_engine import evaluate_alert_rules, get_alert_settings

init_db()

print("Initializing backend ML model instances...")
voice_runner = VoiceModelRunner()
alcohol_runner = AlcoholModelRunner()
tremor_runner = TremorModelRunner()
print("All 3 ML Models loaded successfully in FastAPI Backend!")

UPLOADS_DIR = BASE_DIR / "backend" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

def convert_to_wav(input_path: Path) -> Path:
    try:
        audio, sr = sf.read(input_path)
        wav_path = input_path.with_suffix('.wav')
        sf.write(wav_path, audio, sr)
        if input_path.suffix.lower() != '.wav':
            os.remove(input_path)
        return wav_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {str(e)}")

app = FastAPI(
    title="Parkinson's Disease Multi-Modal Screening Dashboard API",
    description="Clinical decision-support & research demonstration API integrating Voice & Alcohol ML models.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RegisterSchema(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: str = "patient"

class LoginSchema(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

def seed_demo_users():
    db = next(get_db())
    try:
        doc = db.query(User).filter(User.role == "doctor").first()
        if not doc:
            doc_user = User(
                email="doctor@hospital.org",
                password_hash=hash_password("DocPass123"),
                full_name="Dr. Sarah Jenkins (Neurology)",
                role="doctor"
            )
            db.add(doc_user)

        pat = db.query(User).filter(User.role == "patient").first()
        if not pat:
            pat_user = User(
                email="patient@example.com",
                password_hash=hash_password("PatPass123"),
                full_name="Arthur Pendelton",
                role="patient"
            )
            db.add(pat_user)
        db.commit()
    except Exception as e:
        db.rollback()
        print("Demo seed note:", e)

seed_demo_users()

@app.post("/api/auth/register", response_model=TokenResponse)
def register(data: RegisterSchema, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == data.email.lower()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")
    
    role = data.role.lower()
    if role not in ["doctor", "patient"]:
        role = "patient"

    user = User(
        email=data.email.lower(),
        password_hash=hash_password(data.password),
        full_name=data.full_name.strip(),
        role=role
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(data={"sub": str(user.id), "role": user.role})
    return TokenResponse(access_token=token, user=UserResponse.from_orm(user))

@app.post("/api/auth/login", response_model=TokenResponse)
def login(data: LoginSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email.lower()).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_access_token(data={"sub": str(user.id), "role": user.role})
    return TokenResponse(access_token=token, user=UserResponse.from_orm(user))

@app.get("/api/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse.from_orm(current_user)

@app.post("/api/auth/logout")
def logout():
    return {"message": "Successfully logged out."}

@app.post("/api/analysis/predict")
async def run_analysis(
    patient_name: str = Form(...),
    patient_age: float = Form(...),
    patient_gender: Optional[str] = Form("Unspecified"),
    avg_drinks: float = Form(...),
    audio_file: Optional[UploadFile] = File(None),
    use_sample_audio: Optional[bool] = Form(False),
    tremor_segment: Optional[int] = Form(1),
    tremor_file: Optional[UploadFile] = File(None),
    use_sample_tremor: Optional[bool] = Form(True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    print(f"=== Analysis Request ===")
    print(f"Patient: {patient_name}, Age: {patient_age}, Drinks: {avg_drinks}")
    print(f"Audio file: {audio_file.filename if audio_file else 'None'}, Use sample: {use_sample_audio}")
    print(f"Tremor file: {tremor_file.filename if tremor_file else 'None'}, Tremor Segment: {tremor_segment}")

    if not (18 <= patient_age <= 120):
        raise HTTPException(status_code=400, detail="Age must be between 18 and 120 years.")
    if avg_drinks < 0 or avg_drinks > 25:
        raise HTTPException(status_code=400, detail="Alcohol drinks/day must be between 0 and 25.")

    temp_audio_path = None
    temp_tremor_path = None
    saved_filename = "sample.wav"

    try:
        # 1. Voice Model Execution
        if audio_file and audio_file.filename:
            file_ext = Path(audio_file.filename).suffix.lower()
            print(f"Received audio file: {audio_file.filename}, extension: {file_ext}")
            if file_ext not in [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"]:
                raise HTTPException(status_code=400, detail="Unsupported audio format. Please upload .wav, .mp3, or .m4a")

            temp_filename = f"rec_{uuid.uuid4().hex[:8]}{file_ext}"
            temp_audio_path = UPLOADS_DIR / temp_filename
            saved_filename = audio_file.filename

            with open(temp_audio_path, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)

            if file_ext != ".wav":
                temp_audio_path = convert_to_wav(temp_audio_path)
                saved_filename = Path(saved_filename).stem + ".wav"
        elif use_sample_audio or DEFAULT_SAMPLE_WAV.exists():
            temp_audio_path = DEFAULT_SAMPLE_WAV
            saved_filename = "recorded_sample_voice.wav"
        else:
            raise HTTPException(status_code=400, detail="Audio recording or file upload is required.")

        voice_res = voice_runner.predict(temp_audio_path)

        # 2. Alcohol-Demographic Model Execution
        alcohol_res = alcohol_runner.predict(patient_age, avg_drinks)

        # 3. Kinematic Tremor Model Execution
        if tremor_file and tremor_file.filename:
            t_ext = Path(tremor_file.filename).suffix.lower()
            if t_ext != ".csv":
                raise HTTPException(status_code=400, detail="Uploaded tremor file must be in .csv format with sensor readings.")
            temp_t_filename = f"tremor_{uuid.uuid4().hex[:8]}.csv"
            temp_tremor_path = UPLOADS_DIR / temp_t_filename
            with open(temp_tremor_path, "wb") as buffer:
                shutil.copyfileobj(tremor_file.file, buffer)
            tremor_res = tremor_runner.predict_csv(temp_tremor_path, source_name=tremor_file.filename)
        elif tremor_segment is not None and 1 <= tremor_segment <= tremor_runner.n_segments:
            tremor_res = tremor_runner.predict_segment(int(tremor_segment))
        else:
            tremor_res = tremor_runner.predict_segment(1)

        # Persist all three model predictions in SQLite
        record = AnalysisRecord(
            user_id=current_user.id,
            patient_name=patient_name.strip(),
            patient_age=patient_age,
            patient_gender=patient_gender,
            avg_drinks=avg_drinks,
            voice_score=voice_res["combined_score"] * 100.0,
            voice_label=voice_res["predicted_label"],
            rf_score=voice_res["rf_score"] * 100.0,
            gb_score=voice_res["gb_score"] * 100.0,
            alcohol_prob=alcohol_res["probability_pct"],
            alcohol_risk=alcohol_res["risk_level"],
            tremor_score=tremor_res["combined_score"] * 100.0,
            tremor_label=tremor_res["predicted_label"],
            tremor_et_score=tremor_res["extra_trees_score"] * 100.0,
            tremor_rf_score=tremor_res["random_forest_score"] * 100.0,
            tremor_gb_score=tremor_res["gradient_boosting_score"] * 100.0,
            tremor_source=tremor_res.get("source", "Benchmark Segment"),
            alert_triggered=False,
            audio_filename=saved_filename
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        alert_triggered = evaluate_alert_rules(
            db=db,
            patient_user=current_user,
            patient_name=patient_name.strip(),
            patient_age=patient_age,
            voice_confidence=record.voice_score,
            analysis_id=record.id
        )

        if alert_triggered:
            record.alert_triggered = True
            db.commit()

        return {
            "analysis_id": record.id,
            "patient_name": record.patient_name,
            "patient_age": record.patient_age,
            "patient_gender": record.patient_gender,
            "avg_drinks": record.avg_drinks,
            "voice_analysis": {
                "prediction_label": voice_res["predicted_label"],
                "combined_score_pct": voice_res["combined_score"] * 100.0,
                "rf_score_pct": voice_res["rf_score"] * 100.0,
                "gb_score_pct": voice_res["gb_score"] * 100.0,
                "predicted_class": voice_res["predicted_class"],
                "audio_source": saved_filename
            },
            "alcohol_analysis": {
                "probability_pct": alcohol_res["probability_pct"],
                "risk_level": alcohol_res["risk_level"],
                "predicted_label": alcohol_res["predicted_label"]
            },
            "tremor_analysis": {
                "status": "Active / Verified",
                "available": True,
                "prediction_label": tremor_res["predicted_label"],
                "combined_score_pct": tremor_res["combined_score"] * 100.0,
                "et_score_pct": tremor_res["extra_trees_score"] * 100.0,
                "rf_score_pct": tremor_res["random_forest_score"] * 100.0,
                "gb_score_pct": tremor_res["gradient_boosting_score"] * 100.0,
                "predicted_class": tremor_res["predicted_class"],
                "windows_analysed": tremor_res["windows_analysed"],
                "source": tremor_res.get("source", "Benchmark Segment"),
                "rows": tremor_res.get("rows", "N/A"),
                "dataset_number": tremor_res.get("dataset_number"),
                "dataset_true_label": tremor_res.get("dataset_true_label"),
                "match": tremor_res.get("match")
            },
            "alert_triggered": alert_triggered,
            "timestamp": record.created_at.isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_audio_path and temp_audio_path != DEFAULT_SAMPLE_WAV and temp_audio_path.exists():
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass
        if temp_tremor_path and temp_tremor_path.exists():
            try:
                os.remove(temp_tremor_path)
            except Exception:
                pass
                pass

@app.get("/api/tremor/catalog")
def get_tremor_catalog():
    return tremor_runner.get_segments_catalog()

@app.post("/api/tremor/predict")
async def run_standalone_tremor(
    tremor_segment: Optional[int] = Form(1),
    tremor_file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user)
):
    temp_tremor_path = None
    try:
        if tremor_file and tremor_file.filename:
            t_ext = Path(tremor_file.filename).suffix.lower()
            if t_ext != ".csv":
                raise HTTPException(status_code=400, detail="Uploaded tremor file must be in .csv format.")
            temp_t_filename = f"tremor_standalone_{uuid.uuid4().hex[:8]}.csv"
            temp_tremor_path = UPLOADS_DIR / temp_t_filename
            with open(temp_tremor_path, "wb") as buffer:
                shutil.copyfileobj(tremor_file.file, buffer)
            return tremor_runner.predict_csv(temp_tremor_path, source_name=tremor_file.filename)
        else:
            seg = tremor_segment if tremor_segment else 1
            return tremor_runner.predict_segment(seg)
    finally:
        if temp_tremor_path and temp_tremor_path.exists():
            try:
                os.remove(temp_tremor_path)
            except Exception:
                pass

@app.get("/api/analysis/history")
def get_analysis_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role == "doctor":
        records = db.query(AnalysisRecord).order_by(AnalysisRecord.created_at.desc()).all()
    else:
        records = db.query(AnalysisRecord).filter(AnalysisRecord.user_id == current_user.id).order_by(AnalysisRecord.created_at.desc()).all()

    return [
        {
            "id": r.id,
            "user_id": r.user_id,
            "patient_name": r.patient_name,
            "patient_age": r.patient_age,
            "patient_gender": r.patient_gender,
            "avg_drinks": r.avg_drinks,
            "voice_score": r.voice_score,
            "voice_label": r.voice_label,
            "rf_score": r.rf_score,
            "gb_score": r.gb_score,
            "alcohol_prob": r.alcohol_prob,
            "alcohol_risk": r.alcohol_risk,
            "tremor_score": r.tremor_score,
            "tremor_label": r.tremor_label,
            "tremor_et_score": r.tremor_et_score,
            "tremor_rf_score": r.tremor_rf_score,
            "tremor_gb_score": r.tremor_gb_score,
            "tremor_source": r.tremor_source,
            "alert_triggered": r.alert_triggered,
            "audio_filename": r.audio_filename,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }
        for r in records
    ]

@app.get("/api/analysis/history/{record_id}")
def get_analysis_detail(
    record_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis record not found.")

    if current_user.role != "doctor" and record.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Unauthorized access to this patient record.")

    return {
        "id": record.id,
        "patient_name": record.patient_name,
        "patient_age": record.patient_age,
        "patient_gender": record.patient_gender,
        "avg_drinks": record.avg_drinks,
        "voice_score": record.voice_score,
        "voice_label": record.voice_label,
        "rf_score": record.rf_score,
        "gb_score": record.gb_score,
        "alcohol_prob": record.alcohol_prob,
        "alcohol_risk": record.alcohol_risk,
        "tremor_score": record.tremor_score,
        "tremor_label": record.tremor_label,
        "tremor_et_score": record.tremor_et_score,
        "tremor_rf_score": record.tremor_rf_score,
        "tremor_gb_score": record.tremor_gb_score,
        "tremor_source": record.tremor_source,
        "alert_triggered": record.alert_triggered,
        "audio_filename": record.audio_filename,
        "created_at": record.created_at.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/api/notifications")
def get_notifications(
    current_user: User = Depends(get_doctor_user),
    db: Session = Depends(get_db)
):
    notifs = db.query(Notification).order_by(Notification.created_at.desc()).all()
    unread_count = db.query(Notification).filter(Notification.is_read == False).count()

    return {
        "unread_count": unread_count,
        "notifications": [
            {
                "id": n.id,
                "patient_id": n.patient_id,
                "patient_name": n.patient_name,
                "patient_age": n.patient_age,
                "voice_confidence": n.voice_confidence,
                "analysis_id": n.analysis_id,
                "is_read": n.is_read,
                "created_at": n.created_at.strftime("%Y-%m-%d %H:%M:%S")
            }
            for n in notifs
        ]
    }

@app.patch("/api/notifications/{notif_id}/read")
def mark_notification_read(
    notif_id: int,
    current_user: User = Depends(get_doctor_user),
    db: Session = Depends(get_db)
):
    notif = db.query(Notification).filter(Notification.id == notif_id).first()
    if not notif:
        raise HTTPException(status_code=404, detail="Notification not found.")

    notif.is_read = True
    db.commit()
    return {"message": "Notification marked as read.", "id": notif.id}

@app.get("/api/patients")
def get_patients(
    current_user: User = Depends(get_doctor_user),
    db: Session = Depends(get_db)
):
    patients = db.query(User).filter(User.role == "patient").all()
    return [
        {
            "id": p.id,
            "full_name": p.full_name,
            "email": p.email,
            "total_analyses": db.query(AnalysisRecord).filter(AnalysisRecord.user_id == p.id).count(),
            "created_at": p.created_at.strftime("%Y-%m-%d")
        }
        for p in patients
    ]

@app.get("/api/settings")
def get_settings():
    return get_alert_settings()

FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def read_root():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Parkinson's Dashboard API Server is Running!"}
