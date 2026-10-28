import os
import datetime
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
import bcrypt
from sqlalchemy.orm import Session

from backend.database import get_db, User

JWT_SECRET = os.getenv("JWT_SECRET", "parkinsons_clinical_demo_secret_key_2026_x89f")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def hash_password(password: str) -> str:
    # Use bcrypt directly to avoid passlib compatibility issues
    safe_pass = str(password).encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(safe_pass, salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Use bcrypt directly for verification
    safe_pass = str(plain_password).encode('utf-8')[:72]
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(safe_pass, hashed_bytes)

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception
    return user

def get_doctor_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access restricted to Doctor role only"
        )
    return current_user

def get_patient_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access restricted to Patient role only"
        )
    return current_user

def verify_patient_doctor_access(current_user: User, patient_id: int, db: Session) -> bool:
    """
    Enforces strict role-based access control and patient isolation:
    - If current_user is a Patient: MUST be their own ID. Cannot access other patients' data.
    - If current_user is a Doctor: Patient MUST be actively assigned to this doctor.
    - Otherwise raises 403 Forbidden.
    """
    if current_user.role == "patient":
        if current_user.id != patient_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Unauthorized: Patients cannot access other patients' records, chat, or notifications."
            )
        return True
    
    if current_user.role == "doctor":
        from backend.database import DoctorPatientAssignment
        assignment = db.query(DoctorPatientAssignment).filter(
            DoctorPatientAssignment.doctor_id == current_user.id,
            DoctorPatientAssignment.patient_id == patient_id,
            DoctorPatientAssignment.status == "active"
        ).first()
        if not assignment:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Unauthorized: This patient is not assigned to you."
            )
        return True

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Unauthorized: Access denied."
    )

