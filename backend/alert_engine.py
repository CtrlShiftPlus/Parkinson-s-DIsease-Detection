import os
from typing import Optional
from sqlalchemy.orm import Session
from backend.database import Notification, User, DoctorPatientAssignment

# Configurable clinical alert thresholds (Loaded from environment variables)
# Documented as DEMO / CONFIGURATION thresholds — not medical guidance.
PARKINSON_THRESHOLD = float(os.getenv("PARKINSON_THRESHOLD", "0.60"))
AGE_THRESHOLD = float(os.getenv("AGE_THRESHOLD", "60.0"))

def get_assigned_doctor_id(db: Session, patient_id: int) -> Optional[int]:
    assignment = db.query(DoctorPatientAssignment).filter(
        DoctorPatientAssignment.patient_id == patient_id,
        DoctorPatientAssignment.status == "active"
    ).first()
    return assignment.doctor_id if assignment else None

def create_notification(
    db: Session,
    patient_id: int,
    doctor_id: Optional[int],
    patient_name: str,
    notification_type: str,
    title: str,
    description: str,
    analysis_id: int = 0,
    patient_age: float = 0.0,
    voice_confidence: float = 0.0,
    target_url: Optional[str] = None
) -> Notification:
    """Helper to persist a patient-specific notification targeted to their assigned doctor."""
    if not doctor_id:
        doctor_id = get_assigned_doctor_id(db, patient_id)

    notification = Notification(
        doctor_id=doctor_id,
        patient_id=patient_id,
        patient_name=patient_name,
        patient_age=patient_age,
        voice_confidence=voice_confidence,
        analysis_id=analysis_id,
        notification_type=notification_type,
        title=title,
        description=description,
        target_url=target_url,
        is_read=False
    )
    db.add(notification)
    db.commit()
    db.refresh(notification)
    return notification

def evaluate_alert_rules(
    db: Session,
    patient_user: User,
    patient_name: str,
    patient_age: float,
    voice_confidence: float,  # Percentage 0 - 100
    analysis_id: int,
    doctor_id: Optional[int] = None
) -> bool:
    """
    Evaluates configurable clinical alert rules:
    IF voice_confidence >= (PARKINSON_THRESHOLD * 100) AND patient_age >= AGE_THRESHOLD
    THEN triggers doctor notification linked directly to the assigned doctor.
    """
    conf_fraction = voice_confidence / 100.0

    if conf_fraction >= PARKINSON_THRESHOLD and patient_age >= AGE_THRESHOLD:
        target_doctor_id = doctor_id if doctor_id is not None else get_assigned_doctor_id(db, patient_user.id)
        
        # Rule satisfied -> Create clinical alert notification record for the assigned doctor
        create_notification(
            db=db,
            patient_id=patient_user.id,
            doctor_id=target_doctor_id,
            patient_name=patient_name,
            patient_age=patient_age,
            voice_confidence=voice_confidence,
            analysis_id=analysis_id,
            notification_type="clinical_alert",
            title=f"HIGH RISK ALERT: {patient_name} ({patient_age:.0f}y)",
            description=f"Clinical rule triggered: Voice Confidence {voice_confidence:.1f}% >= {PARKINSON_THRESHOLD*100:.0f}% and Age {patient_age:.0f} >= {AGE_THRESHOLD:.0f}.",
            target_url=f"#analysis-{analysis_id}"
        )
        return True
    
    return False

def get_alert_settings():
    return {
        "pariknson_threshold_pct": PARKINSON_THRESHOLD * 100.0,
        "pariknson_threshold_fraction": PARKINSON_THRESHOLD,
        "age_threshold_years": AGE_THRESHOLD,
        "disclaimer": "These configuration thresholds are research/demo values and do NOT constitute formal medical guidance."
    }

