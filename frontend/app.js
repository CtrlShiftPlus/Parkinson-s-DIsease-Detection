/* =============================================================================
   Parkinson's Multi-Modal ML Dashboard — Client Application Logic (app.js)
   Integrated Voice, Alcohol, Kinematic Tremor, One-to-One Patient-Doctor Assignment,
   Private Consultation Chat, and Doctor Notification System.
   ============================================================================= */

document.addEventListener('DOMContentLoaded', () => {

    // =========================================================================
    // GLOBAL APPLICATION STATE
    // =========================================================================
    let authToken = localStorage.getItem('parkinsons_jwt_token') || null;
    let currentUser = null;
    let assignedDoctor = null;
    let assignedPatientsList = [];
    let activeChatPatientId = null;
    let chatPollingTimer = null;
    let globalPollingTimer = null;

    // Audio & Recording State
    let mediaRecorder = null;
    let audioChunks = [];
    let recordedAudioBlob = null;
    let recordTimerInterval = null;
    let recordSeconds = 0;
    let selectedUploadFile = null;
    let useSampleAudioFlag = false;

    // Tremor Model State
    let useTremorSegmentMode = true;
    let selectedTremorUploadFile = null;

    // =========================================================================
    // DOM ELEMENTS REFERENCES
    // =========================================================================
    // Auth & Navigation
    const authScreen = document.getElementById('auth-screen');
    const dashboardScreen = document.getElementById('dashboard-screen');
    const navUserControls = document.getElementById('nav-user-controls');
    const userDisplayName = document.getElementById('user-display-name');
    const userRoleTag = document.getElementById('user-role-tag');
    const logoutBtn = document.getElementById('logout-btn');

    // Nav Actions
    const navChatBtn = document.getElementById('nav-chat-btn');
    const navChatUnreadBadge = document.getElementById('nav-chat-unread-badge');
    const doctorNotifBell = document.getElementById('doctor-notif-bell');
    const notifBtn = document.getElementById('notif-btn');
    const notifCountBadge = document.getElementById('notif-count');

    // Auth Forms
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    const tabLoginBtn = document.getElementById('tab-login-btn');
    const tabRegisterBtn = document.getElementById('tab-register-btn');
    const authError = document.getElementById('auth-error');
    const demoDoctorBtn = document.getElementById('demo-doctor-btn');
    const demoPatientBtn = document.getElementById('demo-patient-btn');

    // Patient Dashboard Elements
    const patientDoctorBanner = document.getElementById('patient-doctor-banner');
    const patDoctorName = document.getElementById('pat-doctor-name');
    const patDoctorEmail = document.getElementById('pat-doctor-email');
    const patAssignedDate = document.getElementById('pat-assigned-date');
    const patDoctorStatusBadge = document.getElementById('pat-doctor-status-badge');
    const btnPatientOpenChat = document.getElementById('btn-patient-open-chat');
    const patChatUnreadBadge = document.getElementById('pat-chat-unread-badge');
    const btnRequestConsultation = document.getElementById('btn-request-consultation');
    const btnChangeDoctor = document.getElementById('btn-change-doctor');

    // Doctor Navigation & Assigned Patients Section
    const doctorNavBar = document.getElementById('doctor-nav-bar');
    const tabDocPatients = document.getElementById('tab-doc-patients');
    const tabDocPipeline = document.getElementById('tab-doc-pipeline');
    const tabDocAlerts = document.getElementById('tab-doc-alerts');
    const tabDocChat = document.getElementById('tab-doc-chat');
    const docPatientsCountBadge = document.getElementById('doc-patients-count-badge');
    const docAlertsCountBadge = document.getElementById('doc-alerts-count-badge');
    const docChatCountBadge = document.getElementById('doc-chat-count-badge');

    const doctorPatientsSection = document.getElementById('doctor-patients-section');
    const docStatTotalPatients = document.getElementById('doc-stat-total-patients');
    const docStatAlerts = document.getElementById('doc-stat-alerts');
    const docStatUnreadMessages = document.getElementById('doc-stat-unread-messages');
    const pendingRequestsCard = document.getElementById('pending-requests-card');
    const pendingRequestsList = document.getElementById('pending-requests-list');
    const assignedPatientsTbody = document.getElementById('assigned-patients-tbody');
    const btnRefreshPatients = document.getElementById('btn-refresh-patients');
    const btnOpenAssignModal = document.getElementById('btn-open-assign-modal');

    // Screening Pipeline & Alerts Containers
    const screeningPipelineContainer = document.getElementById('screening-pipeline-container');
    const doctorNotifsSection = document.getElementById('doctor-notifications-section');
    const unreadCountBadge = document.getElementById('unread-count-badge');
    const notificationsContainer = document.getElementById('notifications-container');
    const refreshNotifsBtn = document.getElementById('refresh-notifs-btn');

    // Audio Subtabs & Live Recording
    const subtabRecord = document.getElementById('subtab-record');
    const subtabUpload = document.getElementById('subtab-upload');
    const modeRecordPanel = document.getElementById('mode-record-panel');
    const modeUploadPanel = document.getElementById('mode-upload-panel');
    const btnStartRecord = document.getElementById('btn-start-record');
    const btnStopRecord = document.getElementById('btn-stop-record');
    const visualizerBox = document.getElementById('visualizer-box');
    const recordTimer = document.getElementById('record-timer');
    const audioPreviewBox = document.getElementById('audio-preview-box');
    const recordedAudioElement = document.getElementById('recorded-audio-element');
    const dropzone = document.getElementById('dropzone');
    const audioFileInput = document.getElementById('audio-file-input');
    const selectedFileName = document.getElementById('selected-file-name');
    const btnUseSample = document.getElementById('btn-use-sample');
    const btnRunPipeline = document.getElementById('btn-run-pipeline');

    // Tremor Input Elements
    const subtabTremorSegment = document.getElementById('subtab-tremor-segment');
    const subtabTremorUpload = document.getElementById('subtab-tremor-upload');
    const modeTremorSegmentPanel = document.getElementById('mode-tremor-segment-panel');
    const modeTremorUploadPanel = document.getElementById('mode-tremor-upload-panel');
    const tremorSegmentInput = document.getElementById('tremor-segment-input');
    const tremorSegmentStatus = document.getElementById('tremor-segment-status');
    const tremorDropzone = document.getElementById('tremor-dropzone');
    const tremorFileInput = document.getElementById('tremor-file-input');
    const selectedTremorFileName = document.getElementById('selected-tremor-file-name');

    // Results Display Elements
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsContent = document.getElementById('results-content');
    const highRiskBanner = document.getElementById('high-risk-banner');

    // Model 1: Voice
    const voiceClassBadge = document.getElementById('voice-class-badge');
    const voiceGaugeFill = document.getElementById('voice-gauge-fill');
    const voiceScoreVal = document.getElementById('voice-score-val');
    const rfScoreVal = document.getElementById('rf-score-val');
    const gbScoreVal = document.getElementById('gb-score-val');
    const audioFilenameVal = document.getElementById('audio-filename-val');

    // Model 2: Alcohol
    const alcoholRiskBadge = document.getElementById('alcohol-risk-badge');
    const alcoholGaugeFill = document.getElementById('alcohol-gauge-fill');
    const alcoholProbVal = document.getElementById('alcohol-prob-val');
    const resPatientAge = document.getElementById('res-patient-age');
    const resDrinksVal = document.getElementById('res-drinks-val');

    // Model 3: Tremor
    const tremorClassBadge = document.getElementById('tremor-class-badge');
    const tremorGaugeFill = document.getElementById('tremor-gauge-fill');
    const tremorScoreVal = document.getElementById('tremor-score-val');
    const tremorEtVal = document.getElementById('tremor-et-val');
    const tremorRfVal = document.getElementById('tremor-rf-val');
    const tremorGbVal = document.getElementById('tremor-gb-val');
    const tremorWindowsVal = document.getElementById('tremor-windows-val');
    const tremorSourceVal = document.getElementById('tremor-source-val');
    const tremorCvVal = document.getElementById('tremor-cv-val');

    // Tri-Modal Synthesis Statuses
    const synthVoiceStatus = document.getElementById('synth-voice-status');
    const synthAlcoholStatus = document.getElementById('synth-alcohol-status');
    const synthTremorStatus = document.getElementById('synth-tremor-status');

    // History & Modals
    const historyTbody = document.getElementById('history-tbody');
    const btnRefreshHistory = document.getElementById('btn-refresh-history');
    const detailModal = document.getElementById('detail-modal');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const modalBodyContent = document.getElementById('modal-body-content');
    const loadingSpinner = document.getElementById('loading-spinner');

    // Chat Modal Elements
    const chatModal = document.getElementById('chat-modal');
    const chatModalCloseBtn = document.getElementById('chat-modal-close-btn');
    const chatCounterpartName = document.getElementById('chat-counterpart-name');
    const chatCounterpartRole = document.getElementById('chat-counterpart-role');
    const chatSidebar = document.getElementById('chat-sidebar');
    const chatPatientList = document.getElementById('chat-patient-list');
    const chatMessagesContainer = document.getElementById('chat-messages-container');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');

    // Patient Detail Modal (Doctor View)
    const patientDetailModal = document.getElementById('patient-detail-modal');
    const patModalTitle = document.getElementById('pat-modal-title');
    const patModalSub = document.getElementById('pat-modal-sub');
    const patModalCloseBtn = document.getElementById('pat-modal-close-btn');
    const patModalTbody = document.getElementById('pat-modal-tbody');
    const btnModalOpenPatientChat = document.getElementById('btn-modal-open-patient-chat');
    let currentSelectedPatientIdForModal = null;

    // Doctor Select Modal (Patient View)
    const doctorSelectModal = document.getElementById('doctor-select-modal');
    const docSelectCloseBtn = document.getElementById('doc-select-close-btn');
    const doctorSelectList = document.getElementById('doctor-select-list');

    // Consultation Modal (Patient View)
    const consultationModal = document.getElementById('consultation-modal');
    const consultationCloseBtn = document.getElementById('consultation-close-btn');
    const consultationForm = document.getElementById('consultation-form');
    const consultationNotes = document.getElementById('consultation-notes');

    // Direct Assign Modal (Doctor View)
    const directAssignModal = document.getElementById('direct-assign-modal');
    const assignCloseBtn = document.getElementById('assign-close-btn');
    const directAssignForm = document.getElementById('direct-assign-form');
    const assignPatientEmail = document.getElementById('assign-patient-email');

    // Helper: Escape HTML to prevent XSS
    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // =========================================================================
    // 1. INITIALIZATION & SESSION MANAGEMENT
    // =========================================================================

    async function initApp() {
        if (authToken) {
            try {
                const res = await fetch('/api/auth/me', {
                    headers: { 'Authorization': `Bearer ${authToken}` }
                });
                if (res.ok) {
                    currentUser = await res.json();
                    setupDashboardForUser();
                    return;
                }
            } catch (err) {
                console.error("Session verification error:", err);
            }
        }
        showAuthScreen();
    }

    function showAuthScreen() {
        authToken = null;
        currentUser = null;
        assignedDoctor = null;
        assignedPatientsList = [];
        activeChatPatientId = null;
        clearInterval(chatPollingTimer);
        clearInterval(globalPollingTimer);
        localStorage.removeItem('parkinsons_jwt_token');

        authScreen.style.display = 'flex';
        dashboardScreen.style.display = 'none';
        navUserControls.style.display = 'none';
    }

    function setupDashboardForUser() {
        authScreen.style.display = 'none';
        dashboardScreen.style.display = 'block';
        navUserControls.style.display = 'flex';

        userDisplayName.textContent = currentUser.full_name;
        userRoleTag.textContent = currentUser.role.toUpperCase();

        if (currentUser.role === 'doctor') {
            doctorNotifBell.style.display = 'block';
            doctorNavBar.style.display = 'flex';
            patientDoctorBanner.style.display = 'none';
            switchDoctorTab('patients');
            loadDoctorNotifications();
            loadAssignedPatients();
            loadPendingAssignments();
        } else {
            doctorNotifBell.style.display = 'none';
            doctorNavBar.style.display = 'none';
            doctorPatientsSection.style.display = 'none';
            doctorNotifsSection.style.display = 'none';
            screeningPipelineContainer.style.display = 'block';
            loadPatientDoctorStatus();
        }

        loadHistory();
        loadUnreadChatCount();
        startGlobalPolling();

        // Animate entrance
        dashboardScreen.style.opacity = '0';
        dashboardScreen.style.transform = 'translateY(20px)';
        setTimeout(() => {
            dashboardScreen.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            dashboardScreen.style.opacity = '1';
            dashboardScreen.style.transform = 'translateY(0)';
        }, 50);
    }

    function startGlobalPolling() {
        clearInterval(globalPollingTimer);
        globalPollingTimer = setInterval(() => {
            if (!authToken) return;
            loadUnreadChatCount();
            if (currentUser && currentUser.role === 'doctor') {
                loadDoctorNotifications();
                loadPendingAssignments();
            } else if (currentUser && currentUser.role === 'patient') {
                loadPatientDoctorStatus();
            }
        }, 8000);
    }

    // =========================================================================
    // 2. AUTHENTICATION HANDLERS
    // =========================================================================

    tabLoginBtn.addEventListener('click', () => {
        tabLoginBtn.classList.add('active');
        tabRegisterBtn.classList.remove('active');
        loginForm.classList.add('active');
        registerForm.classList.remove('active');
        hideAuthError();
    });

    tabRegisterBtn.addEventListener('click', () => {
        tabRegisterBtn.classList.add('active');
        tabLoginBtn.classList.remove('active');
        registerForm.classList.add('active');
        loginForm.classList.remove('active');
        hideAuthError();
    });

    function showAuthError(msg) {
        authError.textContent = msg;
        authError.style.display = 'block';
    }

    function hideAuthError() {
        authError.style.display = 'none';
    }

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideAuthError();
        const email = document.getElementById('login-email').value;
        const password = document.getElementById('login-password').value;

        try {
            const res = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Login failed');

            authToken = data.access_token;
            currentUser = data.user;
            localStorage.setItem('parkinsons_jwt_token', authToken);
            setupDashboardForUser();
        } catch (err) {
            showAuthError(err.message);
        }
    });

    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideAuthError();
        const full_name = document.getElementById('reg-name').value;
        const email = document.getElementById('reg-email').value;
        const password = document.getElementById('reg-password').value;
        const role = document.getElementById('reg-role').value;

        try {
            const res = await fetch('/api/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ full_name, email, password, role })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Registration failed');

            authToken = data.access_token;
            currentUser = data.user;
            localStorage.setItem('parkinsons_jwt_token', authToken);
            setupDashboardForUser();
        } catch (err) {
            showAuthError(err.message);
        }
    });

    demoDoctorBtn.addEventListener('click', () => {
        document.getElementById('login-email').value = 'doctor@hospital.org';
        document.getElementById('login-password').value = 'DocPass123';
        loginForm.requestSubmit();
    });

    demoPatientBtn.addEventListener('click', () => {
        document.getElementById('login-email').value = 'patient@example.com';
        document.getElementById('login-password').value = 'PatPass123';
        loginForm.requestSubmit();
    });

    logoutBtn.addEventListener('click', () => {
        showAuthScreen();
    });

    // =========================================================================
    // 3. DOCTOR NAVIGATION & ASSIGNED PATIENT MANAGEMENT
    // =========================================================================

    function switchDoctorTab(tab) {
        tabDocPatients.classList.toggle('active', tab === 'patients');
        tabDocPipeline.classList.toggle('active', tab === 'pipeline');
        tabDocAlerts.classList.toggle('active', tab === 'alerts');
        tabDocChat.classList.toggle('active', tab === 'chat');

        if (tab === 'patients') {
            doctorPatientsSection.style.display = 'block';
            screeningPipelineContainer.style.display = 'none';
            doctorNotifsSection.style.display = 'none';
            loadAssignedPatients();
            loadPendingAssignments();
        } else if (tab === 'pipeline') {
            doctorPatientsSection.style.display = 'none';
            screeningPipelineContainer.style.display = 'block';
            doctorNotifsSection.style.display = 'none';
        } else if (tab === 'alerts') {
            doctorPatientsSection.style.display = 'none';
            screeningPipelineContainer.style.display = 'none';
            doctorNotifsSection.style.display = 'block';
            loadDoctorNotifications();
        } else if (tab === 'chat') {
            openChatModal(assignedPatientsList.length > 0 ? assignedPatientsList[0].id : null);
        }
    }

    if (tabDocPatients) tabDocPatients.addEventListener('click', () => switchDoctorTab('patients'));
    if (tabDocPipeline) tabDocPipeline.addEventListener('click', () => switchDoctorTab('pipeline'));
    if (tabDocAlerts) tabDocAlerts.addEventListener('click', () => switchDoctorTab('alerts'));
    if (tabDocChat) tabDocChat.addEventListener('click', () => switchDoctorTab('chat'));

    if (notifBtn) {
        notifBtn.addEventListener('click', () => {
            if (currentUser && currentUser.role === 'doctor') {
                switchDoctorTab('alerts');
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        });
    }

    if (navChatBtn) {
        navChatBtn.addEventListener('click', () => {
            openChatModal();
        });
    }

    async function loadAssignedPatients() {
        try {
            const res = await fetch('/api/patients', {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!res.ok) return;
            assignedPatientsList = await res.json();

            docStatTotalPatients.textContent = assignedPatientsList.length;
            docPatientsCountBadge.textContent = assignedPatientsList.length;

            if (assignedPatientsList.length === 0) {
                assignedPatientsTbody.innerHTML = `<tr><td colspan="9" class="text-center" style="padding:2rem;">No patients assigned to your clinical panel yet. Use "Direct Assign Patient" or accept pending assignment requests.</td></tr>`;
                return;
            }

            assignedPatientsTbody.innerHTML = assignedPatientsList.map(p => `
                <tr>
                    <td><strong>${p.patient_code}</strong></td>
                    <td>
                        <div><strong>${escapeHtml(p.full_name)}</strong></div>
                        <div style="font-size:0.8rem; color:var(--text-muted);">${escapeHtml(p.email)}</div>
                    </td>
                    <td>${p.assigned_date}</td>
                    <td><strong>${p.total_analyses}</strong> tests</td>
                    <td>${p.latest_prediction ? (p.latest_prediction.includes("Parkinson") ? '<span class="badge badge-positive">Parkinson\'s</span>' : '<span class="badge badge-negative">Healthy</span>') : '<span class="badge badge-grey">None</span>'}</td>
                    <td>${p.latest_voice_score !== null ? `${p.latest_voice_score.toFixed(1)}%` : '—'}</td>
                    <td>${p.latest_tremor_score !== null ? `${p.latest_tremor_score.toFixed(1)}%` : '—'}</td>
                    <td><span class="badge ${p.latest_alcohol_risk === 'HIGH' ? 'badge-risk-high' : (p.latest_alcohol_risk === 'MEDIUM' ? 'badge-risk-med' : 'badge-risk-low')}">${p.latest_alcohol_risk}</span></td>
                    <td>
                        <div style="display:flex; gap:0.4rem;">
                            <button class="btn btn-primary btn-xs patient-chat-btn" data-id="${p.id}" data-name="${escapeHtml(p.full_name)}">
                                💬 Chat ${p.unread_chat_count > 0 ? `<span class="tab-badge" style="background:var(--danger-color);">${p.unread_chat_count}</span>` : ''}
                            </button>
                            <button class="btn btn-secondary btn-xs patient-detail-btn" data-id="${p.id}">
                                📋 Records
                            </button>
                        </div>
                    </td>
                </tr>
            `).join('');

            document.querySelectorAll('.patient-chat-btn').forEach(b => {
                b.addEventListener('click', (e) => {
                    const pid = parseInt(b.dataset.id);
                    openChatModal(pid);
                });
            });

            document.querySelectorAll('.patient-detail-btn').forEach(b => {
                b.addEventListener('click', (e) => {
                    const pid = parseInt(b.dataset.id);
                    openPatientDetailModal(pid);
                });
            });

        } catch (err) {
            console.error("Failed to load assigned patients:", err);
        }
    }

    if (btnRefreshPatients) btnRefreshPatients.addEventListener('click', loadAssignedPatients);

    async function loadPendingAssignments() {
        if (!currentUser || currentUser.role !== 'doctor') return;
        try {
            const res = await fetch('/api/assignment/pending', {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!res.ok) return;
            const pending = await res.json();

            if (pending.length === 0) {
                pendingRequestsCard.style.display = 'none';
                return;
            }

            pendingRequestsCard.style.display = 'block';
            pendingRequestsList.innerHTML = pending.map(req => `
                <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(255,255,255,0.03); padding:0.75rem 1rem; border-radius:var(--radius-md); border:1px solid rgba(255,255,255,0.08);">
                    <div>
                        <strong>${escapeHtml(req.patient_name)}</strong> (${escapeHtml(req.patient_email)})
                        <div style="font-size:0.75rem; color:var(--text-secondary);">Requested: ${req.requested_at}</div>
                    </div>
                    <div style="display:flex; gap:0.5rem;">
                        <button class="btn btn-primary btn-xs accept-req-btn" data-id="${req.assignment_id}">Accept Patient</button>
                        <button class="btn btn-outline btn-xs decline-req-btn" data-id="${req.assignment_id}">Decline</button>
                    </div>
                </div>
            `).join('');

            document.querySelectorAll('.accept-req-btn').forEach(b => {
                b.addEventListener('click', async (e) => {
                    const id = b.dataset.id;
                    const res = await fetch(`/api/assignment/${id}/accept`, {
                        method: 'POST',
                        headers: { 'Authorization': `Bearer ${authToken}` }
                    });
                    if (res.ok) {
                        loadPendingAssignments();
                        loadAssignedPatients();
                    }
                });
            });

            document.querySelectorAll('.decline-req-btn').forEach(b => {
                b.addEventListener('click', async (e) => {
                    const id = b.dataset.id;
                    const res = await fetch(`/api/assignment/${id}/decline`, {
                        method: 'POST',
                        headers: { 'Authorization': `Bearer ${authToken}` }
                    });
                    if (res.ok) {
                        loadPendingAssignments();
                    }
                });
            });

        } catch (err) {
            console.error("Failed to load pending assignments:", err);
        }
    }

    async function openPatientDetailModal(patientId) {
        currentSelectedPatientIdForModal = patientId;
        patModalTbody.innerHTML = `<tr><td colspan="8" class="text-center">Loading patient clinical records...</td></tr>`;
        patientDetailModal.style.display = 'flex';

        try {
            const res = await fetch(`/api/patients/${patientId}`, {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!res.ok) throw new Error("Could not load patient clinical details.");
            const data = await res.json();

            patModalTitle.textContent = `${data.full_name} (${data.patient_code})`;
            patModalSub.textContent = `Email: ${data.email} | Total Analyses: ${data.total_analyses} | Status: ${data.assignment_status.toUpperCase()}`;

            if (data.analyses.length === 0) {
                patModalTbody.innerHTML = `<tr><td colspan="8" class="text-center">No assessments recorded for this patient yet.</td></tr>`;
                return;
            }

            patModalTbody.innerHTML = data.analyses.map(a => `
                <tr>
                    <td>${a.created_at}</td>
                    <td>${a.voice_label.includes("Parkinson") ? '<span class="badge badge-positive">Parkinson\'s</span>' : '<span class="badge badge-negative">Healthy</span>'}</td>
                    <td><strong>${a.voice_score.toFixed(1)}%</strong></td>
                    <td>${a.tremor_label ? (a.tremor_label.includes("Parkinson") ? '<span class="badge badge-positive">Tremor Pattern</span>' : '<span class="badge badge-negative">Healthy</span>') : '—'}</td>
                    <td><strong>${a.tremor_score !== null && a.tremor_score !== undefined ? `${a.tremor_score.toFixed(1)}%` : '—'}</strong></td>
                    <td><span class="badge ${a.alcohol_risk === 'HIGH' ? 'badge-risk-high' : (a.alcohol_risk === 'MEDIUM' ? 'badge-risk-med' : 'badge-risk-low')}">${a.alcohol_risk}</span></td>
                    <td>${a.alert_triggered ? '🔔 HIGH' : '—'}</td>
                    <td><button class="btn btn-secondary btn-xs pat-analysis-detail-btn" data-id="${a.id}">Full Report</button></td>
                </tr>
            `).join('');

            document.querySelectorAll('.pat-analysis-detail-btn').forEach(b => {
                b.addEventListener('click', (e) => {
                    openAnalysisDetailModal(b.dataset.id);
                });
            });

        } catch (err) {
            patModalTbody.innerHTML = `<tr><td colspan="8" class="text-center" style="color:var(--danger-color);">${err.message}</td></tr>`;
        }
    }

    if (patModalCloseBtn) {
        patModalCloseBtn.addEventListener('click', () => {
            patientDetailModal.style.display = 'none';
        });
    }

    if (btnModalOpenPatientChat) {
        btnModalOpenPatientChat.addEventListener('click', () => {
            patientDetailModal.style.display = 'none';
            if (currentSelectedPatientIdForModal) {
                openChatModal(currentSelectedPatientIdForModal);
            }
        });
    }

    // Direct Assign Modal
    if (btnOpenAssignModal) {
        btnOpenAssignModal.addEventListener('click', () => {
            directAssignModal.style.display = 'flex';
        });
    }

    if (assignCloseBtn) {
        assignCloseBtn.addEventListener('click', () => {
            directAssignModal.style.display = 'none';
        });
    }

    if (directAssignForm) {
        directAssignForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = assignPatientEmail.value.trim();
            if (!email) return;

            try {
                const res = await fetch('/api/assignment/assign', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}`
                    },
                    body: JSON.stringify({ patient_email: email })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || "Assignment failed");

                alert(data.message || "Patient assigned successfully.");
                directAssignModal.style.display = 'none';
                assignPatientEmail.value = '';
                loadAssignedPatients();

            } catch (err) {
                alert(`Error: ${err.message}`);
            }
        });
    }

    // =========================================================================
    // 4. PATIENT DASHBOARD: DOCTOR PROFILE, CONSULTATION & SELECT MODAL
    // =========================================================================

    async function loadPatientDoctorStatus() {
        if (!currentUser || currentUser.role !== 'patient') return;

        try {
            const res = await fetch('/api/assignment/my-doctor', {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!res.ok) return;
            const data = await res.json();
            assignedDoctor = data;

            patientDoctorBanner.style.display = 'flex';

            if (data.has_doctor && data.doctor) {
                patDoctorName.textContent = data.doctor.full_name;
                patDoctorEmail.textContent = data.doctor.email;
                patAssignedDate.textContent = `Assigned Primary Physician (${data.doctor.doctor_code || 'Verified'})`;
                patDoctorStatusBadge.textContent = "Active Care";
                patDoctorStatusBadge.className = "badge badge-success";
                btnPatientOpenChat.style.display = 'inline-flex';
                btnRequestConsultation.style.display = 'inline-flex';
                btnChangeDoctor.textContent = "🔄 Select / Change Doctor";

                if (data.unread_chat_count > 0) {
                    patChatUnreadBadge.textContent = data.unread_chat_count;
                    patChatUnreadBadge.style.display = 'inline-block';
                } else {
                    patChatUnreadBadge.style.display = 'none';
                }
            } else if (data.status === 'pending' && data.doctor) {
                patDoctorName.textContent = data.doctor.full_name;
                patDoctorEmail.textContent = data.doctor.email;
                patAssignedDate.textContent = "Assignment Request Pending Doctor Confirmation";
                patDoctorStatusBadge.textContent = "Pending Approval";
                patDoctorStatusBadge.className = "badge badge-warning";
                btnPatientOpenChat.style.display = 'none';
                btnRequestConsultation.style.display = 'none';
                btnChangeDoctor.textContent = "🔄 Select Another Doctor";
            } else {
                patDoctorName.textContent = "No Assigned Primary Care Neurologist";
                patDoctorEmail.textContent = "Connect to an authorized clinician to enable secure medical chat & reports review";
                patAssignedDate.textContent = "Direct care link inactive";
                patDoctorStatusBadge.textContent = "Not Linked";
                patDoctorStatusBadge.className = "badge badge-alert";
                btnPatientOpenChat.style.display = 'none';
                btnRequestConsultation.style.display = 'none';
                btnChangeDoctor.textContent = "🏥 Select & Link Doctor";
            }
        } catch (err) {
            console.error("Failed to load patient doctor status:", err);
        }
    }

    if (btnPatientOpenChat) {
        btnPatientOpenChat.addEventListener('click', () => {
            openChatModal(currentUser.id);
        });
    }

    if (btnChangeDoctor) {
        btnChangeDoctor.addEventListener('click', () => {
            openDoctorSelectModal();
        });
    }

    async function openDoctorSelectModal() {
        doctorSelectModal.style.display = 'flex';
        doctorSelectList.innerHTML = `<div class="empty-state">Loading registered neurologists...</div>`;

        try {
            const res = await fetch('/api/doctors', {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!res.ok) throw new Error("Could not load doctors directory.");
            const doctors = await res.json();

            if (doctors.length === 0) {
                doctorSelectList.innerHTML = `<div class="empty-state">No registered doctors found in directory.</div>`;
                return;
            }

            doctorSelectList.innerHTML = doctors.map(doc => `
                <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(255,255,255,0.03); padding:0.85rem 1rem; border-radius:var(--radius-md); border:1px solid var(--border-color);">
                    <div>
                        <strong style="color:var(--text-primary); font-size:0.95rem;">${escapeHtml(doc.full_name)}</strong>
                        <div style="font-size:0.78rem; color:var(--text-secondary);">${doc.doctor_code} • ${escapeHtml(doc.email)}</div>
                    </div>
                    <button class="btn btn-primary btn-xs select-doc-btn" data-id="${doc.id}" data-name="${escapeHtml(doc.full_name)}">
                        Link as My Doctor
                    </button>
                </div>
            `).join('');

            document.querySelectorAll('.select-doc-btn').forEach(b => {
                b.addEventListener('click', async (e) => {
                    const docId = parseInt(b.dataset.id);
                    const docName = b.dataset.name;
                    await linkDoctorToPatient(docId, docName);
                });
            });

        } catch (err) {
            doctorSelectList.innerHTML = `<div class="empty-state" style="color:var(--danger-color);">${err.message}</div>`;
        }
    }

    async function linkDoctorToPatient(docId, docName) {
        try {
            const res = await fetch('/api/assignment/request', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${authToken}`
                },
                body: JSON.stringify({ doctor_id: docId, auto_accept: true })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Doctor linking failed");

            alert(data.message || `Successfully linked to Dr. ${docName}!`);
            doctorSelectModal.style.display = 'none';
            loadPatientDoctorStatus();

        } catch (err) {
            alert(`Error: ${err.message}`);
        }
    }

    if (docSelectCloseBtn) {
        docSelectCloseBtn.addEventListener('click', () => {
            doctorSelectModal.style.display = 'none';
        });
    }

    // Consultation Request Modal
    if (btnRequestConsultation) {
        btnRequestConsultation.addEventListener('click', () => {
            consultationModal.style.display = 'flex';
        });
    }

    if (consultationCloseBtn) {
        consultationCloseBtn.addEventListener('click', () => {
            consultationModal.style.display = 'none';
        });
    }

    if (consultationForm) {
        consultationForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const notes = consultationNotes.value.trim();

            try {
                const res = await fetch('/api/consultation/request', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}`
                    },
                    body: JSON.stringify({ notes })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || "Consultation request failed");

                alert("Medical follow-up consultation request sent to your assigned doctor!");
                consultationModal.style.display = 'none';
                consultationNotes.value = '';

            } catch (err) {
                alert(`Error: ${err.message}`);
            }
        });
    }

    // =========================================================================
    // 5. PRIVATE ONE-TO-ONE CHAT ENGINE
    // =========================================================================

    async function openChatModal(patientId = null) {
        chatModal.style.display = 'flex';

        if (currentUser.role === 'doctor') {
            chatSidebar.style.display = 'flex';
            if (!patientId && assignedPatientsList.length > 0) {
                patientId = assignedPatientsList[0].id;
            }
            activeChatPatientId = patientId;
            renderChatPatientSidebar(activeChatPatientId);

            if (activeChatPatientId) {
                const pat = assignedPatientsList.find(p => p.id === activeChatPatientId);
                chatCounterpartName.textContent = pat ? `Patient: ${pat.full_name} (${pat.patient_code})` : "Patient Consultation";
                chatCounterpartRole.textContent = "Assigned Clinical Patient";
            } else {
                chatCounterpartName.textContent = "No Patient Selected";
                chatCounterpartRole.textContent = "Assign a patient to begin consultation";
            }
        } else {
            chatSidebar.style.display = 'none';
            activeChatPatientId = currentUser.id;
            chatCounterpartName.textContent = assignedDoctor && assignedDoctor.doctor ? assignedDoctor.doctor.full_name : "Assigned Physician";
            chatCounterpartRole.textContent = "Primary Clinical Neurologist";
        }

        if (activeChatPatientId) {
            await loadChatMessages(activeChatPatientId, false);
        } else {
            chatMessagesContainer.innerHTML = `<div class="empty-state">No assigned patient selected for consultation.</div>`;
        }

        // Start periodic chat refresh (every 3 seconds)
        clearInterval(chatPollingTimer);
        chatPollingTimer = setInterval(() => {
            if (chatModal.style.display === 'flex' && activeChatPatientId) {
                loadChatMessages(activeChatPatientId, true);
            }
        }, 3000);
    }

    function renderChatPatientSidebar(selectedId) {
        if (!chatPatientList) return;
        if (assignedPatientsList.length === 0) {
            chatPatientList.innerHTML = `<div class="empty-state" style="font-size:0.8rem;">No assigned patients.</div>`;
            return;
        }

        chatPatientList.innerHTML = assignedPatientsList.map(p => `
            <div class="chat-patient-item ${p.id === selectedId ? 'active' : ''}" data-id="${p.id}">
                <div>
                    <div style="font-weight:600; font-size:0.88rem; color:var(--text-primary);">${escapeHtml(p.full_name)}</div>
                    <div style="font-size:0.75rem; color:var(--text-muted);">${p.patient_code}</div>
                </div>
                ${p.unread_chat_count > 0 ? `<span class="tab-badge" style="background:var(--danger-color);">${p.unread_chat_count}</span>` : ''}
            </div>
        `).join('');

        document.querySelectorAll('.chat-patient-item').forEach(item => {
            item.addEventListener('click', () => {
                const pid = parseInt(item.dataset.id);
                activeChatPatientId = pid;
                renderChatPatientSidebar(pid);
                const pat = assignedPatientsList.find(p => p.id === pid);
                if (pat) {
                    chatCounterpartName.textContent = `Patient: ${pat.full_name} (${pat.patient_code})`;
                    chatCounterpartRole.textContent = "Assigned Clinical Patient";
                }
                loadChatMessages(pid, false);
            });
        });
    }

    async function loadChatMessages(patientId, silent = false) {
        if (!patientId || !authToken) return;

        try {
            const res = await fetch(`/api/chat/messages?patient_id=${patientId}`, {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!res.ok) return;
            const data = await res.json();

            if (data.messages.length === 0) {
                chatMessagesContainer.innerHTML = `<div class="empty-state">No messages exchanged yet. Send a message to start this consultation.</div>`;
                return;
            }

            const wasScrolledToBottom = (chatMessagesContainer.scrollHeight - chatMessagesContainer.clientHeight) <= (chatMessagesContainer.scrollTop + 60);

            chatMessagesContainer.innerHTML = data.messages.map(m => `
                <div class="chat-bubble-row ${m.is_self ? 'outgoing' : 'incoming'}">
                    <span class="chat-sender-name">${m.is_self ? 'You' : escapeHtml(m.sender_name)} • ${m.sender_role.toUpperCase()}</span>
                    <div class="chat-bubble">
                        ${escapeHtml(m.message)}
                        <div class="chat-bubble-meta">
                            <span>${m.created_at.split(' ')[1] || m.created_at}</span>
                            ${m.is_self ? `<span>${m.is_read ? '✓✓' : '✓'}</span>` : ''}
                        </div>
                    </div>
                </div>
            `).join('');

            if (wasScrolledToBottom || !silent) {
                chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
            }

            // Mark conversation as read
            await fetch(`/api/chat/read?patient_id=${patientId}`, {
                method: 'PATCH',
                headers: { 'Authorization': `Bearer ${authToken}` }
            });

            loadUnreadChatCount();

        } catch (err) {
            console.error("Failed to load chat messages:", err);
        }
    }

    if (chatForm) {
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = chatInput.value.trim();
            if (!text) return;
            if (!activeChatPatientId) {
                alert("No active patient selected for consultation.");
                return;
            }

            chatInput.value = '';

            try {
                const body = {
                    message: text,
                    patient_id: currentUser.role === 'doctor' ? activeChatPatientId : undefined
                };

                const res = await fetch('/api/chat/messages', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}`
                    },
                    body: JSON.stringify(body)
                });

                if (!res.ok) {
                    const errData = await res.json();
                    throw new Error(errData.detail || "Failed to send message");
                }

                await loadChatMessages(activeChatPatientId, false);
                if (currentUser.role === 'doctor') {
                    loadAssignedPatients();
                }

            } catch (err) {
                alert(`Chat Error: ${err.message}`);
            }
        });
    }

    if (chatModalCloseBtn) {
        chatModalCloseBtn.addEventListener('click', () => {
            chatModal.style.display = 'none';
            clearInterval(chatPollingTimer);
            loadUnreadChatCount();
        });
    }

    async function loadUnreadChatCount() {
        if (!authToken) return;
        try {
            const res = await fetch('/api/chat/unread', {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!res.ok) return;
            const data = await res.json();
            const count = data.unread_chat_count;

            if (navChatUnreadBadge) {
                if (count > 0) {
                    navChatUnreadBadge.textContent = count;
                    navChatUnreadBadge.style.display = 'inline-block';
                } else {
                    navChatUnreadBadge.style.display = 'none';
                }
            }

            if (docChatCountBadge) {
                if (count > 0) {
                    docChatCountBadge.textContent = count;
                    docChatCountBadge.style.display = 'inline-block';
                } else {
                    docChatCountBadge.style.display = 'none';
                }
            }

            if (docStatUnreadMessages) {
                docStatUnreadMessages.textContent = count;
            }

        } catch (err) {
            console.error("Failed to load unread chat count:", err);
        }
    }

    // =========================================================================
    // 6. DOCTOR NOTIFICATIONS & CLINICAL ALERT CENTER
    // =========================================================================

    async function loadDoctorNotifications() {
        if (!currentUser || currentUser.role !== 'doctor') return;
        try {
            const res = await fetch('/api/notifications', {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!res.ok) return;

            const data = await res.json();
            notifCountBadge.textContent = data.unread_count;
            unreadCountBadge.textContent = `${data.unread_count} New`;
            docAlertsCountBadge.textContent = `${data.unread_count} New`;
            docStatAlerts.textContent = data.unread_count;

            if (data.notifications.length === 0) {
                notificationsContainer.innerHTML = `<div class="empty-state">No clinical risk alerts or consultation notices generated yet.</div>`;
                return;
            }

            notificationsContainer.innerHTML = data.notifications.map(n => {
                const typeClass = `notif-type-${n.notification_type || 'clinical_alert'}`;
                const typeLabel = (n.notification_type || 'clinical_alert').replace('_', ' ');
                return `
                <div class="notif-card ${n.is_read ? 'read' : ''}">
                    <div>
                        <div class="notif-title">
                            <span class="notif-type-tag ${typeClass}">${typeLabel}</span>
                            <strong>${escapeHtml(n.title)}</strong>
                        </div>
                        <div class="notif-meta">
                            ${escapeHtml(n.description)} | Generated: ${n.created_at}
                        </div>
                    </div>
                    <div style="display:flex; gap:0.5rem; align-items:center;">
                        ${!n.is_read ? `<button class="btn btn-secondary btn-xs mark-read-btn" data-id="${n.id}">Mark Read</button>` : ''}
                        ${n.notification_type === 'chat_message' ? 
                            `<button class="btn btn-primary btn-xs notif-chat-action-btn" data-patient="${n.patient_id}" data-notif="${n.id}">💬 Chat</button>` :
                          (n.analysis_id && n.analysis_id > 0 ? 
                            `<button class="btn btn-primary btn-xs notif-analysis-action-btn" data-id="${n.analysis_id}" data-notif="${n.id}">View Analysis</button>` :
                            `<button class="btn btn-primary btn-xs notif-patient-action-btn" data-patient="${n.patient_id}" data-notif="${n.id}">Patient File</button>`)
                        }
                    </div>
                </div>
            `}).join('');

            // Notification Actions
            document.querySelectorAll('.mark-read-btn').forEach(b => {
                b.addEventListener('click', async (e) => {
                    const id = b.dataset.id;
                    await fetch(`/api/notifications/${id}/read`, {
                        method: 'PATCH',
                        headers: { 'Authorization': `Bearer ${authToken}` }
                    });
                    loadDoctorNotifications();
                });
            });

            document.querySelectorAll('.notif-chat-action-btn').forEach(b => {
                b.addEventListener('click', async (e) => {
                    const pid = parseInt(b.dataset.patient);
                    const notifId = b.dataset.notif;
                    await fetch(`/api/notifications/${notifId}/read`, {
                        method: 'PATCH',
                        headers: { 'Authorization': `Bearer ${authToken}` }
                    });
                    loadDoctorNotifications();
                    openChatModal(pid);
                });
            });

            document.querySelectorAll('.notif-analysis-action-btn').forEach(b => {
                b.addEventListener('click', async (e) => {
                    const aid = b.dataset.id;
                    const notifId = b.dataset.notif;
                    await fetch(`/api/notifications/${notifId}/read`, {
                        method: 'PATCH',
                        headers: { 'Authorization': `Bearer ${authToken}` }
                    });
                    loadDoctorNotifications();
                    openAnalysisDetailModal(aid);
                });
            });

            document.querySelectorAll('.notif-patient-action-btn').forEach(b => {
                b.addEventListener('click', async (e) => {
                    const pid = parseInt(b.dataset.patient);
                    const notifId = b.dataset.notif;
                    await fetch(`/api/notifications/${notifId}/read`, {
                        method: 'PATCH',
                        headers: { 'Authorization': `Bearer ${authToken}` }
                    });
                    loadDoctorNotifications();
                    openPatientDetailModal(pid);
                });
            });

        } catch (err) {
            console.error("Failed to load doctor notifications:", err);
        }
    }

    if (refreshNotifsBtn) refreshNotifsBtn.addEventListener('click', loadDoctorNotifications);

    // =========================================================================
    // 7. PREVIOUS ANALYSES HISTORY TABLE
    // =========================================================================

    async function loadHistory() {
        try {
            const res = await fetch('/api/analysis/history', {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!res.ok) return;

            const records = await res.json();

            if (records.length === 0) {
                historyTbody.innerHTML = `<tr><td colspan="9" class="text-center">No analysis records found.</td></tr>`;
                return;
            }

            historyTbody.innerHTML = records.map(r => `
                <tr>
                    <td>${r.created_at}</td>
                    <td><strong>${escapeHtml(r.patient_name)}</strong> (${r.patient_age}y)</td>
                    <td>${r.voice_label.includes("Parkinson") ? '<span class="badge badge-positive">Parkinson\'s</span>' : '<span class="badge badge-negative">Healthy</span>'}</td>
                    <td><strong>${r.voice_score.toFixed(1)}%</strong></td>
                    <td><span class="badge ${r.alcohol_risk === 'HIGH' ? 'badge-risk-high' : (r.alcohol_risk === 'MEDIUM' ? 'badge-risk-med' : 'badge-risk-low')}">${r.alcohol_risk} (${r.alcohol_prob.toFixed(1)}%)</span></td>
                    <td>${r.tremor_label ? (r.tremor_label.includes("Parkinson") ? '<span class="badge badge-positive">Tremor Pattern</span>' : '<span class="badge badge-negative">Healthy</span>') : '<span class="badge badge-grey">N/A</span>'}</td>
                    <td><strong>${r.tremor_score !== null && r.tremor_score !== undefined ? `${r.tremor_score.toFixed(1)}%` : '—'}</strong></td>
                    <td>${r.alert_triggered ? '🔔 HIGH' : '—'}</td>
                    <td><button class="btn btn-secondary btn-xs history-detail-btn" data-id="${r.id}">Details</button></td>
                </tr>
            `).join('');

            document.querySelectorAll('.history-detail-btn').forEach(b => {
                b.addEventListener('click', (e) => {
                    openAnalysisDetailModal(e.target.dataset.id);
                });
            });

        } catch (err) {
            console.error("Failed to load analysis history:", err);
        }
    }

    btnRefreshHistory.addEventListener('click', loadHistory);

    async function openAnalysisDetailModal(analysisId) {
        try {
            const res = await fetch(`/api/analysis/history/${analysisId}`, {
                headers: { 'Authorization': `Bearer ${authToken}` }
            });
            if (!res.ok) throw new Error("Could not fetch analysis record details.");

            const r = await res.json();

            modalBodyContent.innerHTML = `
                <div class="sub-metrics-grid" style="font-size:0.9rem; gap:0.75rem;">
                    <div><strong>Patient Name:</strong> ${escapeHtml(r.patient_name)}</div>
                    <div><strong>Age / Gender:</strong> ${r.patient_age} yrs | ${escapeHtml(r.patient_gender)}</div>
                    <div><strong>Avg Alcohol Input:</strong> ${r.avg_drinks} drinks/day</div>
                    <div><strong>Date & Time:</strong> ${r.created_at}</div>
                    <div><strong>Voice Ensemble Score:</strong> ${r.voice_score.toFixed(2)}%</div>
                    <div><strong>Voice Prediction:</strong> ${r.voice_label}</div>
                    <div><strong>Random Forest Score:</strong> ${r.rf_score.toFixed(2)}%</div>
                    <div><strong>Gradient Boosting Score:</strong> ${r.gb_score.toFixed(2)}%</div>
                    <div><strong>Alcohol Model Probability:</strong> ${r.alcohol_prob.toFixed(2)}% (${r.alcohol_risk} RISK)</div>
                    <div><strong>Tremor Model Score:</strong> ${r.tremor_score !== null && r.tremor_score !== undefined ? `${r.tremor_score.toFixed(2)}%` : 'N/A'}</div>
                    <div><strong>Tremor Prediction:</strong> ${r.tremor_label || 'N/A'}</div>
                    <div><strong>Tremor Source:</strong> ${r.tremor_source || 'N/A'}</div>
                    <div><strong>Clinical Alert Triggered:</strong> ${r.alert_triggered ? 'Yes (🔔 Doctor Notified)' : 'No'}</div>
                    <div><strong>Audio Source File:</strong> ${r.audio_filename}</div>
                </div>
            `;
            detailModal.style.display = 'flex';

        } catch (err) {
            alert(err.message);
        }
    }

    modalCloseBtn.addEventListener('click', () => {
        detailModal.style.display = 'none';
    });

    detailModal.addEventListener('click', (e) => {
        if (e.target === detailModal) {
            detailModal.style.display = 'none';
        }
    });

    // Close modals on clicking overlay background
    [chatModal, patientDetailModal, doctorSelectModal, consultationModal, directAssignModal].forEach(m => {
        if (m) {
            m.addEventListener('click', (e) => {
                if (e.target === m) {
                    m.style.display = 'none';
                    if (m === chatModal) clearInterval(chatPollingTimer);
                }
            });
        }
    });

    // =========================================================================
    // 8. AUDIO SUBTABS & LIVE WEB RECORDER (PRESERVED 100%)
    // =========================================================================

    subtabRecord.addEventListener('click', () => {
        subtabRecord.classList.add('active');
        subtabUpload.classList.remove('active');
        modeRecordPanel.classList.add('active');
        modeUploadPanel.classList.remove('active');
    });

    subtabUpload.addEventListener('click', () => {
        subtabUpload.classList.add('active');
        subtabRecord.classList.remove('active');
        modeUploadPanel.classList.add('active');
        modeRecordPanel.classList.remove('active');
    });

    btnStartRecord.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioChunks = [];
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                recordedAudioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(recordedAudioBlob);
                recordedAudioElement.src = audioUrl;
                audioPreviewBox.style.display = 'block';
                selectedUploadFile = null;
                useSampleAudioFlag = false;
            };

            mediaRecorder.start();
            btnStartRecord.style.display = 'none';
            btnStopRecord.style.display = 'inline-flex';
            visualizerBox.classList.add('recording');

            recordSeconds = 0;
            recordTimer.textContent = '00:00';
            recordTimerInterval = setInterval(() => {
                recordSeconds++;
                const mins = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
                const secs = String(recordSeconds % 60).padStart(2, '0');
                recordTimer.textContent = `${mins}:${secs}`;
            }, 1000);

        } catch (err) {
            alert("Microphone permission denied or unsupported device. Please use File Upload instead.");
            console.error("Microphone access error:", err);
        }
    });

    btnStopRecord.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        clearInterval(recordTimerInterval);
        btnStartRecord.style.display = 'inline-flex';
        btnStopRecord.style.display = 'none';
        visualizerBox.classList.remove('recording');
    });

    dropzone.addEventListener('click', () => audioFileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--primary-color)';
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.style.borderColor = 'var(--border-color)';
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--border-color)';
        if (e.dataTransfer.files.length) {
            handleFileSelected(e.dataTransfer.files[0]);
        }
    });

    audioFileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelected(e.target.files[0]);
        }
    });

    function handleFileSelected(file) {
        selectedUploadFile = file;
        recordedAudioBlob = null;
        useSampleAudioFlag = false;
        selectedFileName.textContent = `Selected File: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        selectedFileName.style.color = 'var(--accent-color)';
    }

    btnUseSample.addEventListener('click', () => {
        useSampleAudioFlag = true;
        selectedUploadFile = null;
        recordedAudioBlob = null;
        selectedFileName.textContent = `Preset Selected: Pre-recorded Sample WAV (my_voice.wav)`;
        selectedFileName.style.color = 'var(--success-color)';
    });

    // =========================================================================
    // 9. TREMOR INPUT HANDLERS (PRESERVED 100%)
    // =========================================================================

    if (subtabTremorSegment && subtabTremorUpload) {
        subtabTremorSegment.addEventListener('click', () => {
            subtabTremorSegment.classList.add('active');
            subtabTremorUpload.classList.remove('active');
            modeTremorSegmentPanel.style.display = 'block';
            modeTremorUploadPanel.style.display = 'none';
            useTremorSegmentMode = true;
        });

        subtabTremorUpload.addEventListener('click', () => {
            subtabTremorUpload.classList.add('active');
            subtabTremorSegment.classList.remove('active');
            modeTremorUploadPanel.style.display = 'block';
            modeTremorSegmentPanel.style.display = 'none';
            useTremorSegmentMode = false;
        });
    }

    if (tremorSegmentInput) {
        tremorSegmentInput.addEventListener('input', (e) => {
            const val = parseInt(e.target.value) || 1;
            const start = (val - 1) * 250 + 1;
            const end = val * 250;
            tremorSegmentStatus.textContent = `Segment #${val} (Rows ${start}–${end}, 9 analysis windows)`;
        });
    }

    document.querySelectorAll('.preset-segment-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const seg = btn.dataset.seg;
            if (tremorSegmentInput) {
                tremorSegmentInput.value = seg;
                const val = parseInt(seg);
                const start = (val - 1) * 250 + 1;
                const end = val * 250;
                tremorSegmentStatus.textContent = `Segment #${val} (Rows ${start}–${end}, 9 analysis windows)`;
                if (subtabTremorSegment) subtabTremorSegment.click();
            }
        });
    });

    if (tremorDropzone && tremorFileInput) {
        tremorDropzone.addEventListener('click', () => tremorFileInput.click());

        tremorDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            tremorDropzone.style.borderColor = 'var(--primary-color)';
        });

        tremorDropzone.addEventListener('dragleave', () => {
            tremorDropzone.style.borderColor = 'var(--border-color)';
        });

        tremorDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            tremorDropzone.style.borderColor = 'var(--border-color)';
            if (e.dataTransfer.files.length) {
                handleTremorFileSelected(e.dataTransfer.files[0]);
            }
        });

        tremorFileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleTremorFileSelected(e.target.files[0]);
            }
        });
    }

    function handleTremorFileSelected(file) {
        selectedTremorUploadFile = file;
        useTremorSegmentMode = false;
        selectedTremorFileName.textContent = `Selected CSV: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        selectedTremorFileName.style.color = 'var(--accent-color)';
    }

    // =========================================================================
    // 10. ML PIPELINE SUBMISSION (PRESERVED 100%)
    // =========================================================================

    btnRunPipeline.addEventListener('click', async () => {
        const patient_name = document.getElementById('patient-name').value.trim();
        const patient_age = parseFloat(document.getElementById('patient-age').value);
        const patient_gender = document.getElementById('patient-gender').value;
        const avg_drinks = parseFloat(document.getElementById('avg-drinks').value);

        if (!patient_name) {
            alert("Please enter patient name / ID.");
            return;
        }
        if (isNaN(patient_age) || patient_age < 18 || patient_age > 120) {
            alert("Please enter a valid age between 18 and 120.");
            return;
        }
        if (isNaN(avg_drinks) || avg_drinks < 0 || avg_drinks > 25) {
            alert("Please enter valid alcohol drinks per day (0 - 25).");
            return;
        }

        const formData = new FormData();
        formData.append('patient_name', patient_name);
        formData.append('patient_age', patient_age);
        formData.append('patient_gender', patient_gender);
        formData.append('avg_drinks', avg_drinks);

        if (recordedAudioBlob) {
            formData.append('audio_file', recordedAudioBlob, 'live_recording.wav');
        } else if (selectedUploadFile) {
            formData.append('audio_file', selectedUploadFile);
        } else {
            formData.append('use_sample_audio', 'true');
        }

        if (!useTremorSegmentMode && selectedTremorUploadFile) {
            formData.append('tremor_file', selectedTremorUploadFile);
        } else {
            const segVal = tremorSegmentInput ? (parseInt(tremorSegmentInput.value) || 1) : 1;
            formData.append('tremor_segment', segVal);
            formData.append('use_sample_tremor', 'true');
        }

        loadingSpinner.style.display = 'flex';
        const spinnerText = document.getElementById('spinner-text');
        if (spinnerText) spinnerText.textContent = "Executing Voice, Alcohol, and Tremor ML Models...";

        try {
            const res = await fetch('/api/analysis/predict', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${authToken}` },
                body: formData
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Prediction execution failed');

            displayAnalysisResults(data);
            loadHistory();
            if (currentUser.role === 'doctor') {
                loadDoctorNotifications();
                loadAssignedPatients();
            } else {
                loadPatientDoctorStatus();
            }

        } catch (err) {
            alert(`Execution Error: ${err.message}`);
        } finally {
            loadingSpinner.style.display = 'none';
        }
    });

    function displayAnalysisResults(data) {
        resultsPlaceholder.style.display = 'none';
        resultsContent.style.display = 'flex';

        resultsContent.style.opacity = '0';
        resultsContent.style.transform = 'translateY(20px)';
        setTimeout(() => {
            resultsContent.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            resultsContent.style.opacity = '1';
            resultsContent.style.transform = 'translateY(0)';
        }, 50);

        if (data.alert_triggered) {
            highRiskBanner.style.display = 'flex';
        } else {
            highRiskBanner.style.display = 'none';
        }

        // Model 1: Voice
        const v = data.voice_analysis;
        const vScore = v.combined_score_pct.toFixed(1);
        voiceScoreVal.textContent = `${vScore}%`;
        voiceGaugeFill.style.width = `${vScore}%`;
        rfScoreVal.textContent = `${v.rf_score_pct.toFixed(1)}%`;
        gbScoreVal.textContent = `${v.gb_score_pct.toFixed(1)}%`;
        audioFilenameVal.textContent = v.audio_source;

        if (v.predicted_class === 1) {
            voiceClassBadge.textContent = "Parkinson's (Acoustic Pattern)";
            voiceClassBadge.className = 'result-class-badge badge-positive';
        } else {
            voiceClassBadge.textContent = "Healthy / Control";
            voiceClassBadge.className = 'result-class-badge badge-negative';
        }

        // Model 2: Alcohol
        const a = data.alcohol_analysis;
        const aProb = a.probability_pct.toFixed(1);
        alcoholProbVal.textContent = `${aProb}%`;
        alcoholGaugeFill.style.width = `${aProb}%`;
        resPatientAge.textContent = `${data.patient_age} yrs`;
        resDrinksVal.textContent = `${data.avg_drinks} drinks/day`;

        alcoholRiskBadge.textContent = `${a.risk_level} RISK`;
        if (a.risk_level === 'HIGH') {
            alcoholRiskBadge.className = 'result-risk-badge badge-risk-high';
        } else if (a.risk_level === 'MEDIUM') {
            alcoholRiskBadge.className = 'result-risk-badge badge-risk-med';
        } else {
            alcoholRiskBadge.className = 'result-risk-badge badge-risk-low';
        }

        // Model 3: Tremor
        if (data.tremor_analysis && data.tremor_analysis.available) {
            const t = data.tremor_analysis;
            const tScore = t.combined_score_pct.toFixed(1);
            tremorScoreVal.textContent = `${tScore}%`;
            tremorGaugeFill.style.width = `${tScore}%`;
            tremorEtVal.textContent = `${t.et_score_pct.toFixed(1)}%`;
            tremorRfVal.textContent = `${t.rf_score_pct.toFixed(1)}%`;
            tremorGbVal.textContent = `${t.gb_score_pct.toFixed(1)}%`;
            tremorWindowsVal.textContent = `${t.windows_analysed} windows`;
            tremorSourceVal.textContent = t.source || `Segment #${t.dataset_number || 1}`;

            if (t.predicted_class === 1) {
                tremorClassBadge.textContent = "Parkinson's Tremor Pattern";
                tremorClassBadge.className = 'result-class-badge badge-positive';
            } else {
                tremorClassBadge.textContent = "Healthy / Control (No Tremor)";
                tremorClassBadge.className = 'result-class-badge badge-negative';
            }

            if (t.dataset_true_label && t.match !== null) {
                tremorCvVal.textContent = `Time-Blocked CV (${t.match ? 'Agreement: YES' : 'Agreement: NO'})`;
            } else {
                tremorCvVal.textContent = "Out-of-Fold Ensemble (Honest Test)";
            }

            // Tri-Modal Synthesis Statuses
            if (synthVoiceStatus) {
                synthVoiceStatus.textContent = v.predicted_class === 1 ? "⚠️ Dysphonic Articulation" : "✅ Normal Phonation";
                synthVoiceStatus.style.color = v.predicted_class === 1 ? "var(--danger-color)" : "var(--success-color)";
            }
            if (synthAlcoholStatus) {
                synthAlcoholStatus.textContent = `${a.risk_level} Risk (${a.probability_pct.toFixed(0)}%)`;
                synthAlcoholStatus.style.color = a.risk_level === 'HIGH' ? "var(--danger-color)" : (a.risk_level === 'MEDIUM' ? "var(--warning-color)" : "var(--success-color)");
            }
            if (synthTremorStatus) {
                synthTremorStatus.textContent = t.predicted_class === 1 ? "⚠️ Tremor Biomarker (+)" : "✅ Normal Biomechanics (-)";
                synthTremorStatus.style.color = t.predicted_class === 1 ? "var(--danger-color)" : "var(--success-color)";
            }
        }
    }

    // Initialize application
    initApp();
});
