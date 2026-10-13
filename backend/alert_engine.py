import os
from sqlalchemy.orm import Session
from backend.database import Notification, User

# Configurable clinical alert thresholds (Loaded from environment variables)
# Documented as DEMO / CONFIGURATION thresholds — not medical guidance.
PARKINSON_THRESHOLD = float(os.getenv("PARKINSON_THRESHOLD", "0.60"))
AGE_THRESHOLD = float(os.getenv("AGE_THRESHOLD", "60.0"))

def evaluate_alert_rules(
    db: Session,
    patient_user: User,
    patient_name: str,
    patient_age: float,
    voice_confidence: float,  # Percentage 0 - 100
    analysis_id: int
) -> bool:
    """
    Evaluates configurable clinical alert rules:
    IF voice_confidence >= (PARKINSON_THRESHOLD * 100) AND patient_age >= AGE_THRESHOLD
    THEN triggers doctor notification.
    """
    conf_fraction = voice_confidence / 100.0

    if conf_fraction >= PARKINSON_THRESHOLD and patient_age >= AGE_THRESHOLD:
        # Rule satisfied -> Create notification record for doctors
        notification = Notification(
            patient_id=patient_user.id,
            patient_name=patient_name,
            patient_age=patient_age,
            voice_confidence=voice_confidence,
            analysis_id=analysis_id,
            is_read=False
        )
        db.add(notification)
        db.commit()
        db.refresh(notification)
        return True
    
    return False

def get_alert_settings():
    return {
        "pariknson_threshold_pct": PARKINSON_THRESHOLD * 100.0,
        "pariknson_threshold_fraction": PARKINSON_THRESHOLD,
        "age_threshold_years": AGE_THRESHOLD,
        "disclaimer": "These configuration thresholds are research/demo values and do NOT constitute formal medical guidance."
    }
