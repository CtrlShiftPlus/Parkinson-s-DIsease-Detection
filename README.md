# Parkinson's Disease Multi-Modal ML Screening Dashboard

A secure web dashboard integrating trained ML models for Parkinson's disease screening research and clinical decision-support.

## ⚠️ IMPORTANT DISCLAIMER

**This is a research/demo decision-support system, NOT a medical diagnosis tool.** Results should be reviewed by qualified clinicians. The ML models are for screening evaluation purposes only and do not provide definitive medical diagnoses.

---

## Architecture Overview

### Project Structure

```
Major Project/
├── backend/
│   ├── server.py              # FastAPI application with all API endpoints
│   ├── auth.py                # JWT authentication & password hashing
│   ├── database.py            # SQLAlchemy models (User, AnalysisRecord, Notification)
│   ├── alert_engine.py        # Configurable clinical alert rules
│   └── uploads/               # Temporary audio file storage
├── frontend/
│   ├── index.html             # Single-page application UI
│   ├── app.js                 # Client-side logic (auth, recording, API calls)
│   └── style.css              # Medical/clinical styling
├── ml/
│   ├── audio_model/           # Voice acoustic model artifacts
│   │   ├── parkinsons_audio_ensemble2.pkl
│   │   └── audio_model_config2.json
│   ├── alcohol_model/         # Alcohol demographic model artifacts
│   │   ├── model.pkl
│   │   └── scaler.pkl
│   └── main.py                # ML model runners (VoiceModelRunner, AlcoholModelRunner)
├── recordings/
│   └── my_voice.wav           # Sample audio for testing
├── .env                       # Environment configuration
├── requirements.txt           # Python dependencies
└── parkinsons_dashboard.db    # SQLite database
```

### ML Model Integration

**1. Voice Model (Acoustic Biomarker)**
- Located: `audio_model/parkinsons_audio_ensemble2.pkl`
- Ensemble: Random Forest (50%) + Gradient Boosting (50%)
- Features: MFCCs, pitch stability, spectral characteristics
- Input: Audio file (.wav, .mp3, .m4a)
- Output: Parkinson's probability (0-100%) + classification

**2. Alcohol-Effect Model (Demographic Risk)**
- Located: `alcohol_model/model.pkl` + `scaler.pkl`
- Algorithm: Logistic Regression with StandardScaler
- Features: Age, Average drinks/day
- Output: Risk probability (0-100%) + risk level (LOW/MEDIUM/HIGH)

**Note:** Models are kept separate because they were trained on different datasets and measure different dimensions (physiological vs epidemiological).

---

## Authentication & Security

### User Roles
- **Patient**: Can record voice, run analysis, view own history
- **Doctor**: Can view all patient analyses, receive high-risk notifications

### Security Features
- JWT-based session authentication
- Bcrypt password hashing (never plain text)
- Role-based access control (RBAC)
- Protected API endpoints
- Input validation and sanitization
- Configurable secrets in `.env`

### Demo Accounts
- **Doctor**: `doctor@hospital.org` / `DocPass123`
- **Patient**: `patient@example.com` / `PatPass123`

---

## Clinical Alert System

### Configurable Thresholds (in `.env`)

```env
PARKINSON_THRESHOLD=0.60    # Voice confidence threshold (0-1)
AGE_THRESHOLD=60.0           # Age threshold in years
```

### Alert Rule
IF:
- Voice confidence ≥ PARKINSON_THRESHOLD (60%)
- AND patient age ≥ AGE_THRESHOLD (60 years)

THEN:
- Create notification for all doctors
- Display high-risk alert in patient dashboard
- Store alert flag in analysis record

**These are demo/configuration values, NOT medical guidance.**

---

## API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login (returns JWT)
- `POST /api/auth/logout` - Session logout
- `GET /api/auth/me` - Get current user info

### Analysis
- `POST /api/analysis/predict` - Run ML pipeline (requires audio + patient data)
- `GET /api/analysis/history` - Get analysis history (role-filtered)
- `GET /api/analysis/history/{record_id}` - Get specific analysis details

### Doctor Notifications
- `GET /api/notifications` - Get all doctor notifications (doctor only)
- `PATCH /api/notifications/{id}/read` - Mark notification as read (doctor only)

### Patients
- `GET /api/patients` - List all patients (doctor only)

### Settings
- `GET /api/settings` - Get alert threshold configuration

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` file (already configured with defaults):

```env
HOST=127.0.0.1
PORT=8000
JWT_SECRET=parkinsons_clinical_demo_secret_key_2026_x89f
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
PARKINSON_THRESHOLD=0.60
AGE_THRESHOLD=60.0
DATABASE_URL=sqlite:///./parkinsons_dashboard.db
```

### 3. Start Backend Server

```bash
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Open Dashboard

Open browser to: **http://127.0.0.1:8000**

---

## End-to-End Testing

### Test 1: Patient Flow

1. Open dashboard at http://127.0.0.1:8000
2. Click "Login as Patient" (or register new patient)
3. Enter patient information:
   - Name: Test Patient
   - Age: 67 (above threshold)
   - Gender: Male
   - Alcohol: 2.0 drinks/day
4. Click "Use Pre-recorded Sample WAV" or upload audio
5. Click "Run Integrated ML Pipeline"
6. View results:
   - Voice model prediction (actual ML output)
   - Alcohol model risk (actual ML output)
   - Tremor section (disabled/greyed out)
7. Check "Patient Analysis History" table

### Test 2: High-Risk Alert

1. Use patient age ≥ 60 and voice confidence ≥ 60%
2. Run analysis
3. Verify "HIGH CLINICAL RISK DETECTED" banner appears
4. Logout as patient

### Test 3: Doctor Notification

1. Login as doctor (use "Login as Doctor" button)
2. Check "Doctor Risk Alert Center" section
3. Verify notification appears with patient details
4. Click "View Analysis" to see full details
5. Click "Mark Read" to dismiss notification

### Test 4: Unauthorized Access

1. Login as patient
2. Try to access `/api/notifications` directly (should return 403)
3. Try to view another patient's analysis (should return 403)

---

## Features Implemented

### ✅ User Accounts & Security
- [x] Multi-user registration/login
- [x] Doctor and Patient roles
- [x] JWT-based authentication
- [x] Bcrypt password hashing
- [x] Role-based access control
- [x] Protected routes

### ✅ Dashboard UI
- [x] Clean medical/clinical design
- [x] Patient information form
- [x] Live voice recording (Web Audio API)
- [x] Audio file upload
- [x] Sample audio option

### ✅ ML Integration
- [x] Voice model (actual trained model)
- [x] Alcohol model (actual trained model)
- [x] Real probability outputs (no fake values)
- [x] Ensemble score display
- [x] Individual model scores (RF, GB)

### ✅ Tremor Section
- [x] Disabled/greyed out state
- [x] "Coming Soon" label
- [x] No fake data generation

### ✅ High-Risk Detection
- [x] Configurable thresholds
- [x] Environment variable configuration
- [x] Alert rule evaluation
- [x] Doctor notification generation

### ✅ Doctor Notifications
- [x] Notification center in doctor dashboard
- [x] Unread count badge
- [x] Mark as read functionality
- [x] View analysis details
- [x] Notification history

### ✅ Patient History
- [x] Analysis record storage
- [x] Historical table view
- [x] Detail modal
- [x] Role-filtered access

### ✅ Backend API
- [x] RESTful endpoints
- [x] Authentication middleware
- [x] Input validation
- [x] Error handling
- [x] CORS configuration

### ✅ Privacy & Security
- [x] Password hashing
- [x] JWT tokens
- [x] Role-based access
- [x] Input validation
- [x] File type restrictions
- [x] Temporary file cleanup
- [x] Disclaimer banner

---

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **SQLAlchemy**: ORM for database
- **SQLite**: Embedded database
- **JWT**: Token-based authentication
- **Bcrypt**: Password hashing
- **Librosa**: Audio feature extraction
- **Scikit-learn**: ML model execution
- **Joblib**: Model serialization

### Frontend
- **Vanilla JavaScript**: No framework dependencies
- **Web Audio API**: Browser-based recording
- **Fetch API**: HTTP requests
- **CSS3**: Responsive styling
- **HTML5**: Semantic markup

---

## Troubleshooting

### Backend won't start
- Check if port 8000 is already in use
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check `.env` file exists and is valid

### ML model loading errors
- Verify `audio_model/parkinsons_audio_ensemble2.pkl` exists
- Verify `alcohol_model/model.pkl` and `alcohol_model/scaler.pkl` exist
- Check that scikit-learn version is compatible

### Audio recording not working
- Check browser permissions for microphone
- Try using file upload instead
- Verify browser supports Web Audio API

### Database errors
- Delete `parkinsons_dashboard.db` and restart server to reinitialize
- Check write permissions in project directory

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| HOST | 127.0.0.1 | Server bind address |
| PORT | 8000 | Server port |
| JWT_SECRET | (generated) | JWT signing secret |
| JWT_ALGORITHM | HS256 | JWT algorithm |
| ACCESS_TOKEN_EXPIRE_MINUTES | 1440 | Token expiration (24 hours) |
| PARKINSON_THRESHOLD | 0.60 | Voice confidence threshold (0-1) |
| AGE_THRESHOLD | 60.0 | Age threshold for alerts (years) |
| DATABASE_URL | sqlite:///./parkinsons_dashboard.db | Database connection string |

---

## Future Enhancements

- [ ] Email/SMS notifications for doctors
- [ ] Tremor sensor hardware integration
- [ ] Multi-language support
- [ ] Export analysis reports (PDF)
- [ ] Patient-doctor assignment system
- [ ] Additional ML models integration
- [ ] PostgreSQL/MongoDB support option
- [ ] Docker containerization
- [ ] Automated testing suite

---

## License

This is a research/educational project. The ML models and dataset references should be cited appropriately in any academic or clinical use.

---

## Contact & Support

For issues or questions about this dashboard, please refer to the project documentation or contact the development team.
