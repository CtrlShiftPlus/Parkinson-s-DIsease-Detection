import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./parkinsons_dashboard.db")

# Enable WAL mode for SQLite for safe concurrent reads/writes
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    role = Column(String, nullable=False, default="patient")  # "doctor" or "patient"
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    analyses = relationship("AnalysisRecord", back_populates="owner", cascade="all, delete-orphan")

class AnalysisRecord(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    patient_name = Column(String, nullable=False)
    patient_age = Column(Float, nullable=False)
    patient_gender = Column(String, nullable=True, default="Unspecified")
    avg_drinks = Column(Float, nullable=False)
    
    voice_score = Column(Float, nullable=False)  # Ensemble percentage (0 - 100)
    voice_label = Column(String, nullable=False)  # "Parkinson's (Acoustic Pattern)" or "Healthy"
    rf_score = Column(Float, nullable=False)
    gb_score = Column(Float, nullable=False)
    
    alcohol_prob = Column(Float, nullable=False)  # Percentage (0 - 100)
    alcohol_risk = Column(String, nullable=False)  # "LOW", "MEDIUM", "HIGH"

    # Kinematic Tremor Model Fields
    tremor_score = Column(Float, nullable=True)    # Percentage (0 - 100)
    tremor_label = Column(String, nullable=True)    # Classification label
    tremor_et_score = Column(Float, nullable=True) # Extra Trees Score (%)
    tremor_rf_score = Column(Float, nullable=True) # Random Forest Score (%)
    tremor_gb_score = Column(Float, nullable=True) # Hist Gradient Boosting Score (%)
    tremor_source = Column(String, nullable=True)   # Dataset Segment or CSV name
    
    alert_triggered = Column(Boolean, default=False)
    audio_filename = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="analyses")

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    patient_name = Column(String, nullable=False)
    patient_age = Column(Float, nullable=False)
    voice_confidence = Column(Float, nullable=False)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)
    # Lightweight schema migration for existing SQLite databases
    from sqlalchemy import text
    with engine.connect() as conn:
        new_columns = [
            ("tremor_score", "FLOAT"),
            ("tremor_label", "VARCHAR"),
            ("tremor_et_score", "FLOAT"),
            ("tremor_rf_score", "FLOAT"),
            ("tremor_gb_score", "FLOAT"),
            ("tremor_source", "VARCHAR"),
        ]
        for col_name, col_type in new_columns:
            try:
                conn.execute(text(f"ALTER TABLE analyses ADD COLUMN {col_name} {col_type}"))
                conn.commit()
            except Exception:
                pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

