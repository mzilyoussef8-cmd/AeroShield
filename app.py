import streamlit as st
import numpy as np
import pandas as pd
import time
from datetime import datetime
from scipy.linalg import inv
import json
import plotly.graph_objects as go

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AeroShield | Cyber Defense System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CLASSE DRONE
# ============================================================
class DroneSimulation:
    def __init__(self, shield_active=False):
        self.shield_active = shield_active
        self.reset()
    
    def reset(self):
        self.start = np.array([0.0, 0.0])
        self.target = np.array([18.0, 0.0])
        self.state = self.start.copy()
        self.velocity = np.array([0.0, 0.0])
        self.time = 0
        self.dt = 0.25
        
        self.gps_reading = self.state.copy()
        self.imu_reading = self.state.copy()
        self.imu_velocity = np.array([0.0, 0.0])
        
        self.kalman_state = self.state.copy()
        self.kalman_covariance = np.eye(2) * 0.01
        self.Q = np.eye(2) * 0.001
        self.R_gps = np.eye(2) * 0.0001
        self.R_imu = np.eye(2) * 0.001
        
        self.history = []
        self.logs = []
        self.attacks_detected = 0
        self.attacks_mitigated = 0
        self.detection_time = None
        self.shield_activated_time = None
        self.max_deviation = 0
        self.arrived = False
        
    def add_log(self, level, message):
        timestamp = f"[{self.time:05.1f}s]"
        self.logs.append(f"{timestamp} [{level}] {message}")
        if len(self.logs) > 200:
            self.logs.pop(0)
    
    def predict_kalman(self):
        self.kalman_covariance = self.kalman_covariance + self.Q
    
    def update_kalman(self, measurement, R):
        K = self.kalman_covariance @ inv(self.kalman_covariance + R)
        self.kalman_state = self.kalman_state + K @ (measurement - self.kalman_state)
        self.kalman_covariance = (np.eye(2) - K) @ self.kalman_covariance
    
    def step(self, attack_type="none", emitter_pos=None):
        self.time += self.dt
        
        error = self.target - self.state
        dist = np.linalg.norm(error)
        
        if dist > 0.2:
            direction = error / dist
            speed = 0.6
            self.velocity = direction * speed
            self.state = self.state + self.velocity * self.dt
        else:
            self.velocity = np.array([0.0, 0.0])
            self.arrived = True
        
        gps_noise = np.random.normal(0, 0.01, 2)
        self.gps_reading = self.state + gps_noise
        
        imu_noise = np.random.normal(0, 0.005, 2)
        self.imu_velocity = self.velocity + imu_noise
        self.imu_reading = self.imu_reading + self.imu_velocity * self.dt + np.random.normal(0, 0.002, 2)
        
        threat_level = 0
        gps_deviation = 0
        
        if attack_type == "spoofing" and self.time > 4 and emitter_pos is not None:
            dist_to_emitter = np.linalg.norm(self.state - emitter_pos)
            spoof_strength = max(0, 1 - dist_to_emitter / 15) * 4
            
            direction_to_emitter = (emitter_pos - self.state)
            if np.linalg.norm(direction_to_emitter) > 0.1:
                direction_to_emitter = direction_to_emitter / np.linalg.norm(direction_to_emitter)
            
            spoof_offset = direction_to_emitter * spoof_strength * (1 - np.exp(-(self.time - 4) * 0.4))
            spoof_offset += np.random.normal(0, 0.05, 2)
            self.gps_reading = self.state + spoof_offset
            gps_deviation = np.linalg.norm(spoof_offset)
            threat_level = 85
            
            if self.detection_time is None and gps_deviation > 0.5:
                self.detection_time = self.time
                self.add_log("CRIT", f"SPOOFING DETECTE - derive vers emetteur: {gps_deviation:.2f} km")
                self.attacks_detected += 1
        
        elif attack_type == "jamming" and self.time > 4:
            jam_intensity = 1 - np.exp(-(self.time - 4) * 0.5)
            jam_noise = np.random.normal(0, 2.5 * jam_intensity, 2)
            self.gps_reading = self.state + jam_noise
            gps_deviation = np.linalg.norm(jam_noise)
            threat_level = 70
            
            if self.detection_time is None and gps_deviation > 0.5:
                self.detection_time = self.time
                self.add_log("CRIT", f"JAMMING DETECTE - bruit GPS: {gps_deviation:.2f} km")
                self.attacks_detected += 1
        
        elif attack_type == "replay" and self.time > 4:
            replay_delay = 3.0
            replay_idx = max(0, len(self.history) - int(replay_delay / self.dt))
            if replay_idx < len(self.history):
                old_pos = np.array([self.history[replay_idx]['real_x'], self.history[replay_idx]['real_y']])
            else:
                old_pos = self.start.copy()
            
            self.gps_reading = old_pos + np.random.normal(0, 0.02, 2)
            gps_deviation = np.linalg.norm(self.state - self.gps_reading)
            threat_level = 90
            
            if self.detection_time is None and gps_deviation > 0.5:
                self.detection_time = self.time
                self.add_log("CRIT", f"REPLAY DETECTE - position obsolete, ecart: {gps_deviation:.2f} km")
                self.attacks_detected += 1
        
        self.predict_kalman()
        
        gps_imu_residual = np.linalg.norm(self.gps_reading - self.imu_reading)
        
        if gps_imu_residual > 0.5 and not self.shield_activated_time:
            self.shield_activated_time = self.time
            if self.shield_active:
                self.add_log("INFO", "AEROSHIELD AUTO-ACTIVE - basculement sur fusion IMU")
                self.attacks_mitigated += 1
        
        if self.shield_active and self.shield_activated_time:
            self.update_kalman(self.imu_reading, self.R_imu)
            display_state = self.kalman_state.copy()
        else:
            self.update_kalman(self.gps_reading, self.R_gps)
            display_state = self.kalman_state.copy()
        
        self.max_deviation = max(self.max_deviation, gps_deviation)
        trust_score = max(0, 100 - gps_imu_residual * 25 - threat_level * 0.3)
        
        self.history.append({
            'time': self.time,
            'real_x': self.state[0], 'real_y': self.state[1],
            'gps_x': self.gps_reading[0], 'gps_y': self.gps_reading[1],
            'imu_x': self.imu_reading[0], 'imu_y': self.imu_reading[1],
            'kalman_x': self.kalman_state[0], 'kalman_y': self.kalman_state[1],
            'display_x': display_state[0], 'display_y': display_state[1],
            'threat': threat_level,
            'trust': trust_score,
            'gps_deviation': gps_deviation,
            'gps_imu_residual': gps_imu_residual,
            'shield_active': self.shield_activated_time is not None
        })
        
        return threat_level, trust_score

# ============================================================
# INITIALISATION
# ============================================================
if 'drone_protected' not in st.session_state:
    st.session_state.drone_protected = DroneSimulation(shield_active=True)
if 'drone_unprotected' not in st.session_state:
    st.session_state.drone_unprotected = DroneSimulation(shield_active=False)
if 'running' not in st.session_state:
    st.session_state.running = False
if 'attack_type' not in st.session_state:
    st.session_state.attack_type = "none"

EMITTER_POS = np.array([11.5, -2.0])

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .stApp {
        background: #0a0e1a;
        font-family: 'Rajdhani', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, div, label, li {
        color: #e2e8f0 !important;
    }
    
    .top-bar {
        background: linear-gradient(90deg, #0a0e1a 0%, #0f1729 50%, #0a0e1a 100%);
        border-bottom: 2px solid #00e5ff;
        padding: 15px 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.3);
    }
    
    .logo {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8em;
        font-weight: 900;
        color: #00e5ff !important;
        text-shadow: 0 0 15px rgba(0, 229, 255, 0.8);
        letter-spacing: 3px;
    }
    
    .logo-sub {
        color: #64748b !important;
        font-size: 0.9em;
        margin-left: 15px;
    }
    
    .panel {
        background: linear-gradient(135deg, #0f1729 0%, #1a2332 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .panel-title {
        font-family: 'Orbitron', sans-serif;
        color: #00e5ff !important;
        font-size: 1.05em;
        font-weight: 700;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid #1e3a5f;
        letter-spacing: 1px;
    }
    
    .gauge-container {
        text-align: center;
        padding: 15px;
    }
    
    .gauge-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.8em;
        font-weight: 900;
        color: #00e5ff !important;
        text-shadow: 0 0 20px rgba(0, 229, 255, 0.8);
        margin-top: -15px;
    }
    
    .gauge-value.danger { color: #ff1744 !important; text-shadow: 0 0 20px rgba(255, 23, 68, 0.8); }
    .gauge-value.warning { color: #ffab00 !important; text-shadow: 0 0 20px rgba(255, 171, 0, 0.8); }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #1e3a5f;
    }
    
    .metric-label { color: #94a3b8 !important; font-size: 0.9em; }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.2em;
    }
    
    .metric-value.good { color: #00e676 !important; }
    .metric-value.bad { color: #ff1744 !important; }
    .metric-value.warn { color: #ffab00 !important; }
    .metric-value.info { color: #00e5ff !important; }
    
    .status-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        font-size: 1em;
    }
    
    .status-label { color: #94a3b8 !important; }
    .status-value { font-weight: 700; }
    .status-value.nominal { color: #00e676 !important; }
    .status-value.critical { color: #ff1744 !important; }
    
    .terminal {
        background: #000000;
        border: 1px solid #00e5ff;
        border-radius: 8px;
        padding: 12px;
        height: 400px;
        overflow-y: auto;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8em;
    }
    
    .terminal-line { margin: 3px 0; }
    .terminal-line.crit { color: #ff1744; }
    .terminal-line.warn { color: #ffab00; }
    .terminal-line.info { color: #00e676; }
    .terminal-line.normal { color: #64748b; }
    
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f1729 100%);
        color: #00e5ff !important;
        border: 2px solid #00e5ff;
        border-radius: 8px;
        padding: 10px 20px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        font-size: 1em;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #00e5ff;
        color: #000 !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.6);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00e676 0%, #00c853 100%);
        color: #000 !important;
        border-color: #00e676;
    }
    
    .alert-banner {
        background: linear-gradient(90deg, rgba(255,23,68,0.2) 0%, rgba(255,23,68,0.1) 100%);
        border: 2px solid #ff1744;
        border-radius: 10px;
        padding: 12px 20px;
        margin: 10px 0;
        color: #ff5252 !important;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 1.1em;
        text-align: center;
        animation: alertPulse 1.5s infinite;
    }
    
    @keyframes alertPulse {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 23, 68, 0.5); }
        50% { box-shadow: 0 0 40px rgba(255, 23, 68, 0.9); }
    }
    
    .success-banner {
        background: linear-gradient(90deg, rgba(0,230,118,0.2) 0%, rgba(0,230,118,0.1) 100%);
        border: 2px solid #00e676;
        border-radius: 10px;
        padding: 12px 20px;
        margin: 10px 0;
        color: #69f0ae !important;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 1.1em;
        text-align: center;
    }
    
    .attack-info {
        background: rgba(255, 171, 0, 0.1);
        border-left: 4px solid #ffab00;
        padding: 12px 15px;
        border-radius: 6px;
        margin: 10px 0;
        color: #ffab00 !important;
        font-size: 0.95em;
    }
    
    .legend-box {
        position: absolute;
        top: 15px;
        right: 15px;
        background: rgba(15, 23, 41, 0.95);
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 12px;
        z-index: 10;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        margin: 6px 0;
        color: #e2e8f0 !important;
        font-size: 0.9em;
    }
    
    .legend-color {
        width: 14px;
        height: 14px;
        border-radius: 3px;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="top-bar">
    <div>
        <span class="logo">🛡️ AEROSHIELD</span>
        <span class="logo-sub">Cyber Defense System • EMI AEROTECH 2026</span>
    </div>
    <div style="color:#64748b; font-family:'JetBrains Mono',monospace;">
        STATUS: <span style="color:#00e676;">● ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# FONCTION BOUTONS D'ATTAQUE
# ============================================================
def create_attack_buttons():
    attacks = [
        ("none", "🟢", "Aucune", "Vol nominal"),
        ("spoofing", "🔴", "GPS Spoofing", "Dérive vers zone interdite"),
        ("jamming", "🟠", "RF Jamming", "Brouillage du signal GPS"),
        ("replay", "🟣", "Replay Attack", "Position obsolète renvoyée")
    ]
    
    cols = st.columns(4)
    for i, (key, emoji, title, desc) in enumerate(attacks):
        with cols[i]:
            is_active = st.session_state.attack_type == key
            btn_label = f"{emoji} {title}"
            
            if st.button(btn_label, key=f"attack_{key}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.attack_type = key
                st.rerun()

# ============================================================
# CONTROLES
# ============================================================
ctrl_cols = st.columns([1, 3, 1])

with ctrl_cols[0]:
    if st.button("▶ DÉMARRER", use_container_width=True, type="primary"):
        st.session_state.running = True

with ctrl_cols[1]:
    st.markdown('<div style="color:#ffab00; font-weight:700; margin-bottom:8px; font-family:Rajdhani;">⚠️ Type de menace à simuler :</div>', unsafe_allow_html=True)
    create_attack_buttons()

with ctrl_cols[2]:
    if st.button("🔄 RESET", use_container_width=True):
        st.session_state.drone_protected = DroneSimulation(shield_active=True)
        st.session_state.drone_unprotected = DroneSimulation(shield_active=False)
        st.session_state.running = False
        st.rerun()

# Info sur l'attaque
attack = st.session_state.attack_type
attack_descriptions = {
    "none": "🟢 **Vol nominal** : Les deux drones suivent la trajectoire A→B sans perturbation.",
    "spoofing": "🔴 **GPS Spoofing** : Un émetteur dans la zone interdite envoie de fausses coordonnées GPS. Le drone **non-protégé** dérive vers la zone interdite, tandis que le drone **protégé** (AeroShield) détecte l'incohérence GPS-IMU et bascule automatiquement sur la navigation inertielle.",
    "jamming": "🟠 **RF Jamming** : Brouillage intense du signal GPS. Le drone **non-protégé** reçoit des positions aléatoires et erre, tandis que le drone **protégé** maintient sa trajectoire via l'IMU.",
    "replay": "🟣 **Replay Attack** : Le GPS renvoie une position obsolète. Le drone **non-protégé** revient en arrière, tandis que le drone **protégé** continue vers la cible."
}
st.markdown(f'<div class="attack-info">{attack_descriptions[attack]}</div>', unsafe_allow_html=True)

# ============================================================
# ALERTES
# ============================================================
drone_p = st.session_state.drone_protected
drone_u = st.session_state.drone_unprotected

shield_p_active = drone_p.shield_activated_time is not None
shield_u_active = drone_u.shield_activated_time is not None

if attack != "none" and not shield_p_active and len(drone_p.history) > 10:
    st.markdown('<div class="alert-banner">🚨 ALERTE : Attaque en cours — Les deux drones sont vulnérables, en attente de détection...</div>', unsafe_allow_html=True)
elif shield_p_active and not shield_u_active and attack != "none":
    st.markdown('<div class="success-banner">✅ AEROSHIELD ACTIF sur le drone protégé — Le drone non-protégé est en dérive</div>', unsafe_allow_html=True)
elif shield_p_active and shield_u_active:
    st.markdown('<div class="success-banner">✅ Les deux drones ont détecté l\'attaque — Comparaison des trajectoires de récupération</div>', unsafe_allow_html=True)

# ============================================================
# LAYOUT PRINCIPAL
# ============================================================
main_col1, main_col2 = st.columns([2.2, 1])

# ============================================================
# COLONNE GAUCHE : CARTE TACTIQUE
# ============================================================
with main_col1:
    st.markdown('<div class="panel" style="position:relative;">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="legend-box">
        <div class="legend-item"><div class="legend-color" style="background:#00e676;"></div>Drone avec AeroShield</div>
        <div class="legend-item"><div class="legend-color" style="background:#ff1744;"></div>Drone sans protection</div>
        <div class="legend-item"><div class="legend-color" style="background:#ffab00;"></div>Estimation IMU</div>
        <div class="legend-item"><div class="legend-color" style="background:#ffffff;"></div>Position GPS reçue</div>
    </div>
    """, unsafe_allow_html=True)
    
    data_p = drone_p.history[-1] if drone_p.history else None
    data_u = drone_u.history[-1] if drone_u.history else None
    hist_p = drone_p.history[-100:] if len(drone_p.history) > 0 else []
    hist_u = drone_u.history[-100:] if len(drone_u.history) > 0 else []
    
    canvas_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ margin: 0; padding: 0; background: #0a0e1a; overflow: hidden; font-family: Arial; }}
        canvas {{ display: block; }}
    </style>
    </head>
    <body>
    <canvas id="mapCanvas" width="950" height="560"></canvas>
    <script>
    const canvas = document.getElementById('mapCanvas');
    const ctx = canvas.getContext('2d');
    
    const dataP = {json.dumps(data_p) if data_p else 'null'};
    const dataU = {json.dumps(data_u) if data_u else 'null'};
    const histP = {json.dumps(hist_p)};
    const histU = {json.dumps(hist_u)};
    const attack = "{attack}";
    const emitterX = {EMITTER_POS[0]};
    const emitterY = {EMITTER_POS[1]};
    
    const scaleX = 900 / 20;
    const scaleY = 500 / 8;
    const offsetX = 50;
    const offsetY = 280;
    
    function toScreen(x, y) {{
        return {{
            x: offsetX + x * scaleX,
            y: offsetY - y * scaleY
        }};
    }}
    
    function drawGrid() {{
        ctx.strokeStyle = 'rgba(30, 58, 95, 0.3)';
        ctx.lineWidth = 1;
        for (let x = 0; x <= 20; x += 2) {{
            const p = toScreen(x, 0);
            ctx.beginPath();
            ctx.moveTo(p.x, 0);
            ctx.lineTo(p.x, canvas.height);
            ctx.stroke();
            ctx.fillStyle = '#64748b';
            ctx.font = '10px Arial';
            ctx.fillText(x + ' km', p.x - 12, canvas.height - 8);
        }}
        for (let y = -4; y <= 4; y += 1) {{
            const p = toScreen(0, y);
            ctx.beginPath();
            ctx.moveTo(0, p.y);
            ctx.lineTo(canvas.width, p.y);
            ctx.stroke();
        }}
    }}
    
    function drawRoute() {{
        const A = toScreen(0, 0);
        const B = toScreen(18, 0);
        
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(0, 229, 255, 0.3)';
        ctx.lineWidth = 2;
        ctx.setLineDash([10, 8]);
        ctx.moveTo(A.x, A.y);
        ctx.lineTo(B.x, B.y);
        ctx.stroke();
        ctx.setLineDash([]);
        
        [A, B].forEach((p, i) => {{
            const label = i === 0 ? 'A' : 'B';
            ctx.beginPath();
            ctx.arc(p.x, p.y, 12, 0, Math.PI * 2);
            ctx.fillStyle = '#0a0e1a';
            ctx.fill();
            ctx.strokeStyle = '#00e5ff';
            ctx.lineWidth = 3;
            ctx.stroke();
            
            ctx.fillStyle = '#00e5ff';
            ctx.font = 'bold 20px Arial';
            ctx.fillText(label, p.x - 30, p.y - 22);
        }});
    }}
    
    function drawForbiddenZone() {{
        const x1 = toScreen(9, -1).x;
        const y1 = toScreen(9, -1).y;
        const x2 = toScreen(14, -3).x;
        const y2 = toScreen(14, -3).y;
        const w = x2 - x1;
        const h = y2 - y1;
        
        ctx.fillStyle = 'rgba(255, 23, 68, 0.12)';
        ctx.fillRect(x1, y1, w, h);
        
        ctx.strokeStyle = '#ff1744';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, w, h);
        
        ctx.strokeStyle = 'rgba(255, 23, 68, 0.5)';
        ctx.lineWidth = 1;
        for (let i = 0; i < w + h; i += 12) {{
            ctx.beginPath();
            ctx.moveTo(x1 + i, y1);
            ctx.lineTo(x1 + i - h, y2);
            ctx.stroke();
        }}
        
        ctx.fillStyle = '#ff1744';
        ctx.font = 'bold 13px Arial';
        ctx.fillText('ZONE INTERDITE · relief / mur', x1 + 10, y1 + 22);
    }}
    
    function drawEmitter() {{
        if (attack !== 'spoofing') return;
        const p = toScreen(emitterX, emitterY);
        const time = Date.now() / 1000;
        
        for (let i = 0; i < 3; i++) {{
            const radius = 20 + ((time * 30 + i * 20) % 60);
            ctx.beginPath();
            ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(255, 171, 0, ${{0.5 - radius/120}})`;
            ctx.lineWidth = 2;
            ctx.stroke();
        }}
        
        ctx.beginPath();
        ctx.moveTo(p.x, p.y - 15);
        ctx.lineTo(p.x - 12, p.y + 10);
        ctx.lineTo(p.x + 12, p.y + 10);
        ctx.closePath();
        ctx.fillStyle = 'rgba(255, 171, 0, 0.4)';
        ctx.fill();
        ctx.strokeStyle = '#ffab00';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        ctx.fillStyle = '#ffab00';
        ctx.font = 'bold 12px Arial';
        ctx.fillText('émetteur', p.x + 18, p.y + 5);
    }}
    
    function drawJammingEffect() {{
        if (attack !== 'jamming') return;
        for (let i = 0; i < 30; i++) {{
            const x = Math.random() * canvas.width;
            const y = Math.random() * canvas.height;
            const size = Math.random() * 3 + 1;
            ctx.fillStyle = `rgba(255, 171, 0, ${{Math.random() * 0.3}})`;
            ctx.fillRect(x, y, size, size);
        }}
    }}
    
    function drawReplayEffect() {{
        if (attack !== 'replay') return;
        if (dataU) {{
            const p = toScreen(dataU.real_x, dataU.real_y);
            const ghost = toScreen(dataU.gps_x, dataU.gps_y);
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(ghost.x, ghost.y);
            ctx.strokeStyle = 'rgba(156, 39, 176, 0.5)';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.stroke();
            ctx.setLineDash([]);
        }}
    }}
    
    function drawTrajectory(history, color, dashed=false, glow=true) {{
        if (history.length < 2) return;
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        if (dashed) ctx.setLineDash([8, 5]);
        if (glow) {{
            ctx.shadowBlur = 8;
            ctx.shadowColor = color;
        }}
        for (let i = 0; i < history.length; i++) {{
            const p = toScreen(history[i].display_x, history[i].display_y);
            if (i === 0) ctx.moveTo(p.x, p.y);
            else ctx.lineTo(p.x, p.y);
        }}
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.shadowBlur = 0;
    }}
    
    function drawDrone(data, color, label, isProtected) {{
        if (!data) return;
        const p = toScreen(data.display_x, data.display_y);
        const pulse = Math.sin(Date.now() / 300) * 4 + 18;
        
        ctx.beginPath();
        ctx.arc(p.x, p.y, pulse, 0, Math.PI * 2);
        ctx.fillStyle = color === '#00e676' ? 'rgba(0, 230, 118, 0.2)' : 'rgba(255, 23, 68, 0.2)';
        ctx.fill();
        
        ctx.beginPath();
        ctx.moveTo(p.x + 14, p.y);
        ctx.lineTo(p.x - 10, p.y - 9);
        ctx.lineTo(p.x - 10, p.y + 9);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 11px Arial';
        ctx.fillText(label, p.x + 18, p.y + 4);
        
        if (isProtected && data.shield_active) {{
            ctx.fillStyle = '#00e676';
            ctx.font = '10px Arial';
            ctx.fillText('🛡️', p.x - 8, p.y - 18);
        }}
    }}
    
    function drawGPSMarker(data, color) {{
        if (!data) return;
        const p = toScreen(data.gps_x, data.gps_y);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }}
    
    function drawIMUMarker(data) {{
        if (!data) return;
        const p = toScreen(data.imu_x, data.imu_y);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#ffab00';
        ctx.fill();
    }}
    
    function animate() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawGrid();
        drawRoute();
        drawForbiddenZone();
        drawEmitter();
        drawJammingEffect();
        drawReplayEffect();
        drawTrajectory(histU, '#ff1744', true, true);
        drawTrajectory(histP, '#00e676', false, true);
        drawGPSMarker(dataU, '#ff1744');
        drawGPSMarker(dataP, '#ffffff');
        drawIMUMarker(dataP);
        drawDrone(dataU, '#ff1744', 'Sans protection', false);
        drawDrone(dataP, '#00e676', 'AeroShield', true);
        requestAnimationFrame(animate);
    }}
    
    animate();
    </script>
    </body>
    </html>
    """
    
    st.components.v1.html(canvas_html, height=560)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# COLONNE DROITE : PANNEAUX
# ============================================================
with main_col2:
    
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Confiance GPS · drone protégé</div>', unsafe_allow_html=True)
    
    trust_p = drone_p.history[-1]['trust'] if drone_p.history else 100
    gauge_color = '#00e5ff' if trust_p > 70 else ('#ffab00' if trust_p > 40 else '#ff1744')
    gauge_class = '' if trust_p > 70 else ('warning' if trust_p > 40 else 'danger')
    
    angle = (trust_p / 100) * 180
    svg_gauge = f"""
    <div class="gauge-container">
        <svg width="220" height="130" viewBox="0 0 220 130">
            <defs>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>
            <path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="#1e3a5f" stroke-width="15" stroke-linecap="round"/>
            <path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="{gauge_color}" stroke-width="15" stroke-linecap="round"
                  stroke-dasharray="{angle * 1.57} 565" filter="url(#glow)" style="transition: all 0.5s ease;"/>
        </svg>
        <div class="gauge-value {gauge_class}">{trust_p:.0f} %</div>
    </div>
    """
    st.markdown(svg_gauge, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    
    dev_p = drone_p.history[-1]['gps_deviation'] if drone_p.history else 0
    dev_u = drone_u.history[-1]['gps_deviation'] if drone_u.history else 0
    residual = drone_p.history[-1]['gps_imu_residual'] if drone_p.history else 0
    detection_t = drone_p.detection_time
    
    status_p = "Nominal" if dev_p < 0.5 else ("Critique" if dev_p > 2 else "Attention")
    status_u = "Nominal" if dev_u < 0.5 else ("Critique" if dev_u > 2 else "Attention")
    
    st.markdown(f"""
    <div class="metric-row">
        <div>
            <div class="metric-label">Écart protégé</div>
            <div class="metric-value good">{dev_p:.2f} km</div>
        </div>
        <div>
            <div class="metric-label">Écart non protégé</div>
            <div class="metric-value {'good' if dev_u < 0.5 else 'bad'}">{dev_u:.2f} km</div>
        </div>
    </div>
    <div class="metric-row">
        <div>
            <div class="metric-label">Résidu GPS – IMU</div>
            <div class="metric-value info">{residual:.2f} km</div>
        </div>
        <div>
            <div class="metric-label">Temps de détection</div>
            <div class="metric-value info">{f'{detection_t:.1f}s' if detection_t else '—'}</div>
        </div>
    </div>
    <div style="margin-top:12px;">
        <div class="status-row">
            <span class="status-label">🛡️ Protégé (AeroShield)</span>
            <span class="status-value {'nominal' if status_p=='Nominal' else 'critical'}">{status_p}</span>
        </div>
        <div class="status-row">
            <span class="status-label">❌ Non protégé</span>
            <span class="status-value {'nominal' if status_u=='Nominal' else 'critical'}">{status_u}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Résidu GPS – IMU (seuil 0,5 km)</div>', unsafe_allow_html=True)
    
    if drone_p.history:
        df_p = pd.DataFrame(drone_p.history)
        df_u = pd.DataFrame(drone_u.history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_u['time'], y=df_u['gps_imu_residual'], 
                                mode='lines', name='Non protégé', 
                                line=dict(color='#ff1744', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=df_p['time'], y=df_p['gps_imu_residual'], 
                                mode='lines', name='Protégé', 
                                line=dict(color='#00e676', width=3)))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#ffab00", 
                     annotation_text="Seuil 0.5 km", annotation_position="top right")
        fig.update_layout(
            template='plotly_dark',
            height=200,
            margin=dict(l=40, r=20, t=20, b=30),
            xaxis_title='Temps (s)',
            yaxis_title='Résidu (km)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', size=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # JOURNAL D'ÉVÉNEMENTS AGRANDI
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Journal d\'événements</div>', unsafe_allow_html=True)
    
    logs_html = '<div class="terminal">'
    # Récupère les 100 derniers logs au lieu de 25
    all_logs = sorted(drone_p.logs + drone_u.logs, key=lambda x: x[:10])[-100:]
    
    if not all_logs:
        logs_html += '<div class="terminal-line normal">[000.0s] En attente de démarrage...</div>'
    else:
        for log in all_logs:
            if "CRIT" in log:
                logs_html += f'<div class="terminal-line crit">{log}</div>'
            elif "WARN" in log:
                logs_html += f'<div class="terminal-line warn">{log}</div>'
            elif "INFO" in log:
                logs_html += f'<div class="terminal-line info">{log}</div>'
            else:
                logs_html += f'<div class="terminal-line normal">{log}</div>'
    
    logs_html += '</div>'
    st.markdown(logs_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# BOUCLE DE SIMULATION
# ============================================================
if st.session_state.running:
    drone_p.step(attack_type=attack, emitter_pos=EMITTER_POS)
    drone_u.step(attack_type=attack, emitter_pos=EMITTER_POS)
    time.sleep(0.2)
    st.rerun()