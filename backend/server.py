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
from backend.database import (
    init_db, get_db, User, AnalysisRecord, Notification,
    DoctorPatientAssignment, ChatMessage
)
from backend.auth import (
    hash_password, verify_password, create_access_token,
    get_current_user, get_doctor_user, get_patient_user, verify_patient_doctor_access
)
from backend.alert_engine import evaluate_alert_rules, get_alert_settings, create_notification, get_assigned_doctor_id


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
            doc = User(
                email="doctor@hospital.org",
                password_hash=hash_password("DocPass123"),
                full_name="Dr. Sarah Jenkins (Neurology)",
                role="doctor"
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

        pat = db.query(User).filter(User.role == "patient").first()
        if not pat:
            pat = User(
                email="patient@example.com",
                password_hash=hash_password("PatPass123"),
                full_name="Arthur Pendelton",
                role="patient"
            )
            db.add(pat)
            db.commit()
            db.refresh(pat)

        # Seed primary doctor-patient assignment for demo users if none exists
        if doc and pat:
            assignment = db.query(DoctorPatientAssignment).filter(
                DoctorPatientAssignment.patient_id == pat.id
            ).first()
            if not assignment:
                assignment = DoctorPatientAssignment(
                    patient_id=pat.id,
                    doctor_id=doc.id,
                    status="active",
                    notes="Primary Neurologist"
                )
                db.add(assignment)
                db.commit()

            # Ensure legacy notifications are attributed to the primary doctor
            db.query(Notification).filter(Notification.doctor_id == None).update(
                {Notification.doctor_id: doc.id},
                synchronize_session=False
            )
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

        assigned_doc_id = get_assigned_doctor_id(db, current_user.id)

        alert_triggered = evaluate_alert_rules(
            db=db,
            patient_user=current_user,
            patient_name=patient_name.strip(),
            patient_age=patient_age,
            voice_confidence=record.voice_score,
            analysis_id=record.id,
            doctor_id=assigned_doc_id
        )

        if alert_triggered:
            record.alert_triggered = True
            db.commit()
        elif assigned_doc_id:
            # Generate assessment completed notification for the assigned doctor
            create_notification(
                db=db,
                patient_id=current_user.id,
                doctor_id=assigned_doc_id,
                patient_name=patient_name.strip(),
                patient_age=patient_age,
                voice_confidence=record.voice_score,
                analysis_id=record.id,
                notification_type="assessment_completed",
                title=f"New Assessment: {patient_name.strip()}",
                description=f"Assessment complete. Voice: {record.voice_label} ({record.voice_score:.1f}%), Tremor: {record.tremor_score:.1f}%, Alcohol Risk: {record.alcohol_risk}.",
                target_url=f"#analysis-{record.id}"
            )


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

class AssignmentRequestSchema(BaseModel):
    doctor_id: int
    auto_accept: Optional[bool] = True

class DirectAssignSchema(BaseModel):
    patient_id: Optional[int] = None
    patient_email: Optional[str] = None

class ReassignSchema(BaseModel):
    patient_id: int
    new_doctor_id: int

class ChatSendSchema(BaseModel):
    patient_id: Optional[int] = None
    message: str

class ConsultationRequestSchema(BaseModel):
    notes: Optional[str] = "Patient requested a follow-up consultation regarding Parkinson's screening results."

@app.get("/api/analysis/history")
def get_analysis_history(
    patient_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role == "doctor":
        if patient_id is not None:
            verify_patient_doctor_access(current_user, patient_id, db)
            records = db.query(AnalysisRecord).filter(AnalysisRecord.user_id == patient_id).order_by(AnalysisRecord.created_at.desc()).all()
        else:
            # Doctors ONLY see records for patients assigned to them
            assigned_patient_ids = [
                a.patient_id for a in db.query(DoctorPatientAssignment).filter(
                    DoctorPatientAssignment.doctor_id == current_user.id,
                    DoctorPatientAssignment.status == "active"
                ).all()
            ]
            records = db.query(AnalysisRecord).filter(AnalysisRecord.user_id.in_(assigned_patient_ids)).order_by(AnalysisRecord.created_at.desc()).all()
    else:
        # Patients can strictly ONLY see their own records
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

    # Strict server-side RBAC and IDOR validation
    verify_patient_doctor_access(current_user, record.user_id, db)

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

# =============================================================================
# NOTIFICATION SYSTEM (DOCTOR & PATIENT)
# =============================================================================

@app.get("/api/notifications")
def get_notifications(
    current_user: User = Depends(get_doctor_user),
    db: Session = Depends(get_db)
):
    # A doctor only receives notifications for their own assigned patients
    notifs = db.query(Notification).filter(
        Notification.doctor_id == current_user.id
    ).order_by(Notification.created_at.desc()).all()

    unread_count = db.query(Notification).filter(
        Notification.doctor_id == current_user.id,
        Notification.is_read == False
    ).count()

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
                "notification_type": n.notification_type or "clinical_alert",
                "title": n.title or f"Alert: {n.patient_name}",
                "description": n.description or f"Voice confidence: {n.voice_confidence:.1f}%",
                "target_url": n.target_url or f"#analysis-{n.analysis_id}",
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

    if notif.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Unauthorized: Notification belongs to a different clinician.")

    notif.is_read = True
    db.commit()
    return {"message": "Notification marked as read.", "id": notif.id}

# =============================================================================
# DOCTOR-PATIENT ASSIGNMENT MANAGEMENT
# =============================================================================

@app.get("/api/doctors")
def list_doctors(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    docs = db.query(User).filter(User.role == "doctor").all()
    return [
        {
            "id": d.id,
            "doctor_code": f"DOC-{d.id:04d}",
            "full_name": d.full_name,
            "email": d.email
        }
        for d in docs
    ]

@app.get("/api/patients")
def get_patients(
    current_user: User = Depends(get_doctor_user),
    db: Session = Depends(get_db)
):
    # Strictly return only patients actively assigned to this doctor
    assignments = db.query(DoctorPatientAssignment).filter(
        DoctorPatientAssignment.doctor_id == current_user.id,
        DoctorPatientAssignment.status == "active"
    ).all()

    assigned_patient_ids = [a.patient_id for a in assignments]
    patients = db.query(User).filter(User.id.in_(assigned_patient_ids)).all()
    assignment_map = {a.patient_id: a for a in assignments}

    result = []
    for p in patients:
        a = assignment_map.get(p.id)
        latest_analysis = db.query(AnalysisRecord).filter(AnalysisRecord.user_id == p.id).order_by(AnalysisRecord.created_at.desc()).first()
        unread_chat_count = db.query(ChatMessage).filter(
            ChatMessage.patient_id == p.id,
            ChatMessage.doctor_id == current_user.id,
            ChatMessage.sender_id == p.id,
            ChatMessage.is_read == False
        ).count()

        result.append({
            "id": p.id,
            "patient_code": f"PAT-{p.id:04d}",
            "full_name": p.full_name,
            "email": p.email,
            "assignment_status": a.status if a else "unassigned",
            "assigned_date": a.created_at.strftime("%Y-%m-%d") if a else "N/A",
            "total_analyses": db.query(AnalysisRecord).filter(AnalysisRecord.user_id == p.id).count(),
            "latest_prediction": latest_analysis.voice_label if latest_analysis else "No assessments yet",
            "latest_voice_score": latest_analysis.voice_score if latest_analysis else None,
            "latest_tremor_score": latest_analysis.tremor_score if latest_analysis else None,
            "latest_alcohol_risk": latest_analysis.alcohol_risk if latest_analysis else "N/A",
            "latest_date": latest_analysis.created_at.strftime("%Y-%m-%d %H:%M") if latest_analysis else "None",
            "has_high_risk": latest_analysis.alert_triggered if latest_analysis else False,
            "unread_chat_count": unread_chat_count,
            "created_at": p.created_at.strftime("%Y-%m-%d")
        })
    return result

@app.get("/api/patients/{patient_id}")
def get_patient_detail(
    patient_id: int,
    current_user: User = Depends(get_doctor_user),
    db: Session = Depends(get_db)
):
    verify_patient_doctor_access(current_user, patient_id, db)
    patient = db.query(User).filter(User.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    analyses = db.query(AnalysisRecord).filter(AnalysisRecord.user_id == patient_id).order_by(AnalysisRecord.created_at.desc()).all()
    assignment = db.query(DoctorPatientAssignment).filter(
        DoctorPatientAssignment.patient_id == patient_id,
        DoctorPatientAssignment.doctor_id == current_user.id,
        DoctorPatientAssignment.status == "active"
    ).first()

    return {
        "id": patient.id,
        "patient_code": f"PAT-{patient.id:04d}",
        "full_name": patient.full_name,
        "email": patient.email,
        "assignment_status": assignment.status if assignment else "active",
        "assigned_at": assignment.created_at.strftime("%Y-%m-%d %H:%M") if assignment else "N/A",
        "total_analyses": len(analyses),
        "analyses": [
            {
                "id": a.id,
                "created_at": a.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "voice_label": a.voice_label,
                "voice_score": a.voice_score,
                "rf_score": a.rf_score,
                "gb_score": a.gb_score,
                "alcohol_prob": a.alcohol_prob,
                "alcohol_risk": a.alcohol_risk,
                "tremor_label": a.tremor_label,
                "tremor_score": a.tremor_score,
                "tremor_source": a.tremor_source,
                "alert_triggered": a.alert_triggered,
                "audio_filename": a.audio_filename
            }
            for a in analyses
        ]
    }

@app.get("/api/assignment/my-doctor")
def get_my_doctor(
    current_user: User = Depends(get_patient_user),
    db: Session = Depends(get_db)
):
    assignment = db.query(DoctorPatientAssignment).filter(
        DoctorPatientAssignment.patient_id == current_user.id,
        DoctorPatientAssignment.status == "active"
    ).first()

    if not assignment:
        # Check pending request
        pending = db.query(DoctorPatientAssignment).filter(
            DoctorPatientAssignment.patient_id == current_user.id,
            DoctorPatientAssignment.status == "pending"
        ).first()
        if pending:
            doc = db.query(User).filter(User.id == pending.doctor_id).first()
            return {
                "has_doctor": False,
                "status": "pending",
                "doctor": {
                    "id": doc.id,
                    "doctor_code": f"DOC-{doc.id:04d}",
                    "full_name": doc.full_name,
                    "email": doc.email
                } if doc else None,
                "assignment_id": pending.id
            }
        return {"has_doctor": False, "status": "unassigned", "doctor": None, "assignment_id": None}

    doc = db.query(User).filter(User.id == assignment.doctor_id).first()
    unread_chat_count = db.query(ChatMessage).filter(
        ChatMessage.patient_id == current_user.id,
        ChatMessage.doctor_id == assignment.doctor_id,
        ChatMessage.sender_id == assignment.doctor_id,
        ChatMessage.is_read == False
    ).count()

    return {
        "has_doctor": True,
        "status": "active",
        "doctor": {
            "id": doc.id,
            "doctor_code": f"DOC-{doc.id:04d}",
            "full_name": doc.full_name,
            "email": doc.email
        } if doc else None,
        "assignment_id": assignment.id,
        "assigned_date": assignment.created_at.strftime("%Y-%m-%d"),
        "unread_chat_count": unread_chat_count
    }

@app.post("/api/assignment/request")
def request_doctor_assignment(
    data: AssignmentRequestSchema,
    current_user: User = Depends(get_patient_user),
    db: Session = Depends(get_db)
):
    doctor = db.query(User).filter(User.id == data.doctor_id, User.role == "doctor").first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found.")

    status_to_set = "active" if data.auto_accept else "pending"

    assignment = db.query(DoctorPatientAssignment).filter(
        DoctorPatientAssignment.patient_id == current_user.id
    ).first()

    if assignment:
        assignment.doctor_id = doctor.id
        assignment.status = status_to_set
        assignment.updated_at = datetime.datetime.utcnow()
    else:
        assignment = DoctorPatientAssignment(
            patient_id=current_user.id,
            doctor_id=doctor.id,
            status=status_to_set
        )
        db.add(assignment)

    db.commit()
    db.refresh(assignment)

    # Notify doctor of assignment or consultation link
    create_notification(
        db=db,
        patient_id=current_user.id,
        doctor_id=doctor.id,
        patient_name=current_user.full_name,
        notification_type="consultation_request" if status_to_set == "pending" else "clinical_alert",
        title=f"Patient Assignment: {current_user.full_name}",
        description=f"Patient {current_user.full_name} is now {'assigned to' if status_to_set == 'active' else 'requesting'} your clinical care.",
        target_url=f"#patient-{current_user.id}"
    )

    return {
        "message": f"Successfully {'assigned to' if status_to_set == 'active' else 'requested'} Dr. {doctor.full_name}.",
        "status": status_to_set,
        "assignment_id": assignment.id,
        "doctor": {
            "id": doctor.id,
            "full_name": doctor.full_name,
            "email": doctor.email
        }
    }

@app.get("/api/assignment/pending")
def get_pending_assignments(
    current_user: User = Depends(get_doctor_user),
    db: Session = Depends(get_db)
):
    pending = db.query(DoctorPatientAssignment).filter(
        DoctorPatientAssignment.doctor_id == current_user.id,
        DoctorPatientAssignment.status == "pending"
    ).all()

    patient_ids = [p.patient_id for p in pending]
    patients = {p.id: p for p in db.query(User).filter(User.id.in_(patient_ids)).all()}

    return [
        {
            "assignment_id": p.id,
            "patient_id": p.patient_id,
            "patient_name": patients[p.patient_id].full_name if p.patient_id in patients else "Unknown Patient",
            "patient_email": patients[p.patient_id].email if p.patient_id in patients else "",
            "requested_at": p.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }
        for p in pending
    ]

@app.post("/api/assignment/{assignment_id}/accept")
def accept_assignment(
    assignment_id: int,
    current_user: User = Depends(get_doctor_user),
    db: Session = Depends(get_db)
):
    assignment = db.query(DoctorPatientAssignment).filter(
        DoctorPatientAssignment.id == assignment_id
    ).first()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment record not found.")

    if assignment.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Unauthorized: Assignment request not directed to you.")

    assignment.status = "active"
    assignment.updated_at = datetime.datetime.utcnow()
    db.commit()
    return {"message": "Patient assignment accepted.", "assignment_id": assignment.id}

@app.post("/api/assignment/{assignment_id}/decline")
def decline_assignment(
    assignment_id: int,
    current_user: User = Depends(get_doctor_user),
    db: Session = Depends(get_db)
):
    assignment = db.query(DoctorPatientAssignment).filter(
        DoctorPatientAssignment.id == assignment_id
    ).first()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment record not found.")

    if assignment.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Unauthorized: Assignment request not directed to you.")

    assignment.status = "declined"
    assignment.updated_at = datetime.datetime.utcnow()
    db.commit()
    return {"message": "Patient assignment declined.", "assignment_id": assignment.id}

@app.post("/api/assignment/assign")
def direct_assign_patient(
    data: DirectAssignSchema,
    current_user: User = Depends(get_doctor_user),
    db: Session = Depends(get_db)
):
    if data.patient_id:
        patient = db.query(User).filter(User.id == data.patient_id, User.role == "patient").first()
    elif data.patient_email:
        patient = db.query(User).filter(User.email == data.patient_email.lower().strip(), User.role == "patient").first()
    else:
        raise HTTPException(status_code=400, detail="Provide patient_id or patient_email.")

    if not patient:
        raise HTTPException(status_code=404, detail="Patient account not found.")

    assignment = db.query(DoctorPatientAssignment).filter(
        DoctorPatientAssignment.patient_id == patient.id
    ).first()

    if assignment:
        assignment.doctor_id = current_user.id
        assignment.status = "active"
        assignment.updated_at = datetime.datetime.utcnow()
    else:
        assignment = DoctorPatientAssignment(
            patient_id=patient.id,
            doctor_id=current_user.id,
            status="active"
        )
        db.add(assignment)

    db.commit()
    db.refresh(assignment)
    return {"message": f"Patient {patient.full_name} assigned successfully.", "patient_id": patient.id}

@app.post("/api/assignment/reassign")
def reassign_patient(
    data: ReassignSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Authorization: Patient reassigning self, or current doctor reassigning
    if current_user.role == "patient" and current_user.id != data.patient_id:
        raise HTTPException(status_code=403, detail="Unauthorized: Cannot reassign another patient.")
    if current_user.role == "doctor":
        verify_patient_doctor_access(current_user, data.patient_id, db)

    new_doc = db.query(User).filter(User.id == data.new_doctor_id, User.role == "doctor").first()
    if not new_doc:
        raise HTTPException(status_code=404, detail="Target doctor not found.")

    assignment = db.query(DoctorPatientAssignment).filter(
        DoctorPatientAssignment.patient_id == data.patient_id
    ).first()

    if assignment:
        assignment.doctor_id = new_doc.id
        assignment.status = "active"
        assignment.updated_at = datetime.datetime.utcnow()
    else:
        assignment = DoctorPatientAssignment(
            patient_id=data.patient_id,
            doctor_id=new_doc.id,
            status="active"
        )
        db.add(assignment)

    db.commit()
    return {"message": f"Successfully reassigned to Dr. {new_doc.full_name}."}

# =============================================================================
# PRIVATE PATIENT-DOCTOR CHAT
# =============================================================================

@app.get("/api/chat/messages")
def get_chat_messages(
    patient_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role == "patient":
        target_patient_id = current_user.id
    else:
        if patient_id is None:
            raise HTTPException(status_code=400, detail="patient_id is required for doctor access.")
        target_patient_id = patient_id
        verify_patient_doctor_access(current_user, target_patient_id, db)

    # Active assignment check
    assignment = db.query(DoctorPatientAssignment).filter(
        DoctorPatientAssignment.patient_id == target_patient_id,
        DoctorPatientAssignment.status == "active"
    ).first()

    if not assignment:
        return {
            "assignment": None,
            "messages": [],
            "total": 0,
            "disclaimer": "This chat is for clinical consultation and symptoms follow-up only. It does not replace emergency medical care."
        }

    # Fetch chronologically ordered messages
    messages = db.query(ChatMessage).filter(
        ChatMessage.patient_id == target_patient_id,
        ChatMessage.doctor_id == assignment.doctor_id
    ).order_by(ChatMessage.created_at.asc()).all()

    patient_user = db.query(User).filter(User.id == target_patient_id).first()
    doctor_user = db.query(User).filter(User.id == assignment.doctor_id).first()

    return {
        "assignment": {
            "patient_id": target_patient_id,
            "patient_name": patient_user.full_name if patient_user else "Patient",
            "doctor_id": assignment.doctor_id,
            "doctor_name": doctor_user.full_name if doctor_user else "Doctor"
        },
        "messages": [
            {
                "id": m.id,
                "patient_id": m.patient_id,
                "doctor_id": m.doctor_id,
                "sender_id": m.sender_id,
                "sender_name": patient_user.full_name if m.sender_id == target_patient_id else (doctor_user.full_name if doctor_user else "Doctor"),
                "sender_role": "patient" if m.sender_id == target_patient_id else "doctor",
                "is_self": m.sender_id == current_user.id,
                "message": m.message,
                "is_read": m.is_read,
                "created_at": m.created_at.strftime("%Y-%m-%d %H:%M:%S")
            }
            for m in messages
        ],
        "total": len(messages),
        "disclaimer": "This chat is intended for discussing symptoms, follow-up questions, reports, and general care guidance. It does not replace professional medical judgment."
    }

@app.post("/api/chat/messages")
def send_chat_message(
    data: ChatSendSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    msg_text = data.message.strip()
    if not msg_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(msg_text) > 2500:
        raise HTTPException(status_code=400, detail="Message exceeds maximum length (2500 characters).")

    if current_user.role == "patient":
        target_patient_id = current_user.id
        assignment = db.query(DoctorPatientAssignment).filter(
            DoctorPatientAssignment.patient_id == current_user.id,
            DoctorPatientAssignment.status == "active"
        ).first()
        if not assignment:
            raise HTTPException(status_code=400, detail="You do not currently have an active assigned doctor.")
        target_doctor_id = assignment.doctor_id
    else:
        if not data.patient_id:
            raise HTTPException(status_code=400, detail="patient_id is required for doctors.")
        target_patient_id = data.patient_id
        verify_patient_doctor_access(current_user, target_patient_id, db)
        target_doctor_id = current_user.id

    # Strictly assign sender_id on server to prevent impersonation
    new_msg = ChatMessage(
        patient_id=target_patient_id,
        doctor_id=target_doctor_id,
        sender_id=current_user.id,
        message=msg_text,
        is_read=False
    )
    db.add(new_msg)
    db.commit()
    db.refresh(new_msg)

    # When patient sends message, trigger notification for assigned doctor
    if current_user.role == "patient":
        create_notification(
            db=db,
            patient_id=target_patient_id,
            doctor_id=target_doctor_id,
            patient_name=current_user.full_name,
            notification_type="chat_message",
            title=f"New Chat Message from {current_user.full_name}",
            description=msg_text[:120],
            target_url=f"#chat-{target_patient_id}"
        )

    return {
        "id": new_msg.id,
        "patient_id": new_msg.patient_id,
        "doctor_id": new_msg.doctor_id,
        "sender_id": new_msg.sender_id,
        "sender_name": current_user.full_name,
        "sender_role": current_user.role,
        "is_self": True,
        "message": new_msg.message,
        "is_read": new_msg.is_read,
        "created_at": new_msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.patch("/api/chat/read")
def mark_chat_read(
    patient_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role == "patient":
        target_patient_id = current_user.id
    else:
        if not patient_id:
            raise HTTPException(status_code=400, detail="patient_id is required.")
        target_patient_id = patient_id
        verify_patient_doctor_access(current_user, target_patient_id, db)

    # Mark incoming unread messages as read
    updated_count = db.query(ChatMessage).filter(
        ChatMessage.patient_id == target_patient_id,
        ChatMessage.sender_id != current_user.id,
        ChatMessage.is_read == False
    ).update({ChatMessage.is_read: True}, synchronize_session=False)

    db.commit()
    return {"message": "Chat messages marked as read.", "updated_count": updated_count}

@app.get("/api/chat/unread")
def get_unread_chat_count(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role == "patient":
        count = db.query(ChatMessage).filter(
            ChatMessage.patient_id == current_user.id,
            ChatMessage.sender_id != current_user.id,
            ChatMessage.is_read == False
        ).count()
    else:
        count = db.query(ChatMessage).filter(
            ChatMessage.doctor_id == current_user.id,
            ChatMessage.sender_id != current_user.id,
            ChatMessage.is_read == False
        ).count()
    return {"unread_chat_count": count}

# =============================================================================
# CONSULTATION FOLLOW-UP REQUEST & PATIENT STATUS
# =============================================================================

@app.post("/api/consultation/request")
def request_consultation(
    data: ConsultationRequestSchema,
    current_user: User = Depends(get_patient_user),
    db: Session = Depends(get_db)
):
    assignment = db.query(DoctorPatientAssignment).filter(
        DoctorPatientAssignment.patient_id == current_user.id,
        DoctorPatientAssignment.status == "active"
    ).first()

    if not assignment:
        raise HTTPException(status_code=400, detail="You do not have an active assigned doctor to request consultation from.")

    create_notification(
        db=db,
        patient_id=current_user.id,
        doctor_id=assignment.doctor_id,
        patient_name=current_user.full_name,
        notification_type="consultation_request",
        title=f"Consultation Request: {current_user.full_name}",
        description=data.notes.strip() if data.notes else "Patient requested a medical consultation.",
        target_url=f"#patient-{current_user.id}"
    )

    return {"message": "Consultation request sent to your assigned doctor successfully."}

@app.get("/api/patient/dashboard-status")
def get_patient_dashboard_status(
    current_user: User = Depends(get_patient_user),
    db: Session = Depends(get_db)
):
    assignment = db.query(DoctorPatientAssignment).filter(
        DoctorPatientAssignment.patient_id == current_user.id,
        DoctorPatientAssignment.status == "active"
    ).first()

    doctor_info = None
    if assignment:
        doc = db.query(User).filter(User.id == assignment.doctor_id).first()
        if doc:
            doctor_info = {
                "id": doc.id,
                "doctor_code": f"DOC-{doc.id:04d}",
                "full_name": doc.full_name,
                "email": doc.email
            }

    unread_chat = db.query(ChatMessage).filter(
        ChatMessage.patient_id == current_user.id,
        ChatMessage.sender_id != current_user.id,
        ChatMessage.is_read == False
    ).count()

    analyses_count = db.query(AnalysisRecord).filter(AnalysisRecord.user_id == current_user.id).count()
    latest_analysis = db.query(AnalysisRecord).filter(AnalysisRecord.user_id == current_user.id).order_by(AnalysisRecord.created_at.desc()).first()

    return {
        "has_assigned_doctor": assignment is not None,
        "assignment_status": assignment.status if assignment else "unassigned",
        "doctor": doctor_info,
        "unread_chat_count": unread_chat,
        "analyses_count": analyses_count,
        "latest_analysis": {
            "id": latest_analysis.id,
            "created_at": latest_analysis.created_at.strftime("%Y-%m-%d %H:%M"),
            "voice_label": latest_analysis.voice_label,
            "voice_score": latest_analysis.voice_score,
            "tremor_label": latest_analysis.tremor_label,
            "tremor_score": latest_analysis.tremor_score,
            "alcohol_risk": latest_analysis.alcohol_risk,
            "alert_triggered": latest_analysis.alert_triggered
        } if latest_analysis else None
    }

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
