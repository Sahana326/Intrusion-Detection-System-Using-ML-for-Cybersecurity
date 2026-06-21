import os
import sys
import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Add the parent directory to sys.path so we can import preprocess and sniffer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocess import FEATURES
from train import train_models
from sniffer import HybridIDSDetector, TrafficSimulator

# Set page config for a premium, wide-screen dashboard layout
st.set_page_config(
    page_title="Cognitive NIDS - ML CyberSecurity Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism styling, clean fonts, card animations and alert badges
st.markdown("""
<style>
    /* Styling headers and fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3f51b5, #00bcd4, #f44336);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #8888aa;
        margin-bottom: 2rem;
    }
    
    /* Metrics panel cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 5px solid #00bcd4;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #aaaaaa;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 0.3rem;
    }
    
    /* Alert Status indicators */
    .risk-low {
        color: #4caf50;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffeb3b;
        font-weight: bold;
    }
    .risk-high {
        color: #ff9800;
        font-weight: bold;
    }
    .risk-critical {
        color: #f44336;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "detector" not in st.session_state:
    st.session_state.detector = HybridIDSDetector(models_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models")))
if "simulator" not in st.session_state:
    st.session_state.simulator = TrafficSimulator()
if "flow_history" not in st.session_state:
    st.session_state.flow_history = []
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "selected_flow_index" not in st.session_state:
    st.session_state.selected_flow_index = None

detector = st.session_state.detector
simulator = st.session_state.simulator

# Header Area
st.markdown("<h1 class='main-title'>🛡️ Cognitive Network Intrusion Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time machine learning anomaly detection and explainable security intelligence engine.</p>", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.image("https://img.shields.io/badge/Security-NIDS%20AI-blueviolet?style=for-the-badge&logo=analyzer", use_container_width=True)
    st.markdown("### 🎛️ Control Panel")
    
    # Engine Settings
    scan_mode = st.radio("Scanning Engine", ["Simulation Mode (Offline)", "Live Capture Mode (Requires Admin)"])
    
    # Run Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Scan", type="primary", use_container_width=True):
            st.session_state.monitoring = True
    with col2:
        if st.button("⏸️ Stop Scan", use_container_width=True):
            st.session_state.monitoring = False
            
    st.markdown("---")
    
    # Force Cyber Attacks
    st.markdown("### 💥 Threat Generator (Simulated)")
    attack_to_trigger = st.selectbox(
        "Inject Specific Attack Profile",
        ["None (Random Traffic)", "DoS", "PortScan", "BruteForce", "Infiltration"]
    )
    
    if st.button("Inject Cyber Attack", use_container_width=True, disabled=not st.session_state.monitoring):
        label = None if attack_to_trigger == "None (Random Traffic)" else attack_to_trigger
        flow_data, true_label = simulator.generate_flow(forced_label=label)
        result = detector.predict_flow(flow_data)
        st.session_state.flow_history.append({**flow_data, **result, "true_label": true_label})
        st.toast(f"Injected {attack_to_trigger} Threat Flow!", icon="⚠️")
        
    st.markdown("---")
    
    # Clear and Model management
    if st.button("🗑️ Clear Flow Logs", use_container_width=True):
        st.session_state.flow_history = []
        st.session_state.selected_flow_index = None
        st.rerun()

    st.markdown("### 🧠 Model Optimization")
    if st.button("🔄 Retrain AI Models", use_container_width=True):
        with st.spinner("Re-training ML pipelines..."):
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
            dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/network_traffic.csv"))
            train_models(dataset_path=dataset_path, models_dir=models_dir)
            st.session_state.detector.load_models()
            st.success("Models retrained and reloaded successfully!")
            time.sleep(1)
            st.rerun()

# ----------------- REAL-TIME SIMULATION LOOP -----------------
# Append flow logs when monitoring is active
if st.session_state.monitoring:
    if scan_mode == "Simulation Mode (Offline)":
        # Simulate 1 to 3 network flows on this iteration loop
        for _ in range(random.randint(1, 3)):
            # If a forced attack is selected in dropdown, we occasionally inject it, otherwise random
            label = None
            if attack_to_trigger != "None (Random Traffic)" and random.random() < 0.3:
                label = attack_to_trigger
                
            flow_data, true_label = simulator.generate_flow(forced_label=label)
            result = detector.predict_flow(flow_data)
            st.session_state.flow_history.append({**flow_data, **result, "true_label": true_label})
            
        # Bound history limit to prevent browser memory growth
        if len(st.session_state.flow_history) > 100:
            st.session_state.flow_history = st.session_state.flow_history[-100:]
    else:
        # Live mode alert
        st.warning("Live Capture requires terminal execution with administrative rights. Run 'python src/sniffer.py --interface <IF>' directly in an Administrator shell.")
        st.session_state.monitoring = False

# ----------------- TOP METRIC PANELS -----------------
total_flows = len(st.session_state.flow_history)
anomalies = [f for f in st.session_state.flow_history if f.get("is_anomaly", False)]
critical_alerts = [f for f in st.session_state.flow_history if f.get("risk_level") in ["High", "Critical"]]

anomaly_rate = (len(anomalies) / total_flows * 100) if total_flows > 0 else 0.0

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.markdown(f"""
    <div class='metric-card' style='border-left-color: #00bcd4;'>
        <div class='metric-title'>Total Network Flows</div>
        <div class='metric-value'>{total_flows}</div>
    </div>
    """, unsafe_allow_html=True)
with m_col2:
    st.markdown(f"""
    <div class='metric-card' style='border-left-color: #ff9800;'>
        <div class='metric-title'>Anomaly Rate</div>
        <div class='metric-value'>{anomaly_rate:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
with m_col3:
    st.markdown(f"""
    <div class='metric-card' style='border-left-color: #f44336;'>
        <div class='metric-title'>Critical Alerts</div>
        <div class='metric-value'>{len(critical_alerts)}</div>
    </div>
    """, unsafe_allow_html=True)
with m_col4:
    status_color = "#4caf50" if st.session_state.monitoring else "#888888"
    status_text = "ACTIVE" if st.session_state.monitoring else "IDLE"
    st.markdown(f"""
    <div class='metric-card' style='border-left-color: {status_color};'>
        <div class='metric-title'>Scanner Status</div>
        <div class='metric-value' style='color: {status_color};'>{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- MAIN LAYOUT SECTION -----------------
layout_col1, layout_col2 = st.columns([1.6, 1.0])

with layout_col1:
    st.markdown("### 📊 Network Flow Analysis Console")
    
    if total_flows == 0:
        st.info("No flow data captured yet. Press 'Start Scan' in the control panel to generate network traffic.")
    else:
        # Convert history list to pandas DataFrame for plotting and displaying
        df_history = pd.DataFrame(st.session_state.flow_history)
        
        # Display Network throughput (Packets/s) and Byte Rate line chart
        chart_df = df_history[["timestamp", "Flow Packets/s", "Flow Bytes/s"]].copy()
        # Downsample to last 20 for cleaner timeline visualization
        chart_df = chart_df.tail(20)
        
        st.line_chart(
            chart_df.set_index("timestamp"),
            y=["Flow Packets/s"],
            use_container_width=True,
            height=180
        )
        
        # Render flow log interactive table
        st.markdown("#### Flow Traffic Logs (Select a row to analyze threat detail)")
        
        # Build display-friendly dataframe
        display_df = df_history[[
            "timestamp", "src_ip", "dst_ip", "protocol", "Destination Port", "prediction", "risk_level"
        ]].copy()
        
        # Reorder and format column names
        display_df.columns = ["Timestamp", "Source IP", "Dest IP", "Protocol", "Port", "Prediction", "Risk Level"]
        display_df = display_df.iloc[::-1]  # Show newest first
        
        # Show interactive grid
        # Streamlit 1.25+ supports st.dataframe with key selections
        st.dataframe(
            display_df,
            use_container_width=True,
            height=300,
            hide_index=False
        )
        
        # Manual flow index selector to inspect
        selected_index_repr = st.number_input(
            "Enter flow index to inspect (Row number from table above):",
            min_value=0,
            max_value=total_flows - 1,
            value=total_flows - 1,
            step=1
        )
        st.session_state.selected_flow_index = int(selected_index_repr)

with layout_col2:
    st.markdown("### 🎯 Explainable Threat Intelligence")
    
    if st.session_state.selected_flow_index is None or total_flows == 0:
        st.info("Select a network flow to run explainable threat diagnostics.")
    else:
        idx = st.session_state.selected_flow_index
        if idx < len(st.session_state.flow_history):
            selected_flow = st.session_state.flow_history[idx]
            
            # Threat Card Title
            risk = selected_flow["risk_level"]
            risk_class = f"risk-{risk.lower()}"
            pred = selected_flow["prediction"]
            conf = selected_flow["confidence"] * 100
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;">
                <h4 style="margin-top: 0; color: #aaaaaa;">Flow Diagnostics</h4>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.4rem; font-weight: 700;">{pred}</span>
                    <span class="{risk_class}" style="font-size: 1.2rem;">{risk} Risk</span>
                </div>
                <div style="font-size: 0.9rem; color: #888888; margin-top: 0.5rem;">
                    Classification Confidence: <b>{conf:.1f}%</b> | Anomaly Score: <b>{selected_flow['anomaly_likelihood']*100:.1f}%</b>
                </div>
                <hr style="border: 0.5px solid rgba(255,255,255,0.1); margin: 1rem 0;"/>
                <table style="width: 100%; font-size: 0.9rem; color: #dddddd;">
                    <tr><td><b>Source Address</b></td><td style="text-align:right;">{selected_flow['src_ip']}</td></tr>
                    <tr><td><b>Destination Host</b></td><td style="text-align:right;">{selected_flow['dst_ip']}:{selected_flow['Destination Port']}</td></tr>
                    <tr><td><b>Protocol</b></td><td style="text-align:right;">{selected_flow['protocol']}</td></tr>
                    <tr><td><b>Flow Duration</b></td><td style="text-align:right;">{selected_flow['Flow Duration']:.2f} us</td></tr>
                    <tr><td><b>Packets (Fwd/Bwd)</b></td><td style="text-align:right;">{selected_flow['Total Fwd Packets']} / {selected_flow['Total Bwd Packets']}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            # XAI Explanation Chart
            st.markdown("#### 🧠 Local AI Explanation (SHAP Proxy)")
            st.caption("Identifies network metrics driving the threat score.")
            
            explanations = selected_flow["explanations"]
            
            # Render a horizontal bar chart of the top features using matplotlib
            features_xai = [item["feature"] for item in explanations][::-1]
            attributions_xai = [item["attribution"] for item in explanations][::-1]
            colors_xai = ["#f44336" if x > 0 else "#4caf50" for x in attributions_xai]
            
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            
            bars = ax.barh(features_xai, attributions_xai, color=colors_xai, height=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#555555')
            ax.spines['left'].set_color('#555555')
            ax.tick_params(colors='#aaaaaa', labelsize=8)
            ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#555555')
            
            # Add center line
            ax.axvline(0, color='#aaaaaa', linewidth=0.8)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Text based explanation summary
            st.markdown("##### Detailed Risk Insights:")
            for exp in explanations:
                bullet_color = "🔴" if exp['attribution'] > 0 else "🟢"
                st.markdown(f"- {bullet_color} **{exp['feature']}** is *{exp['direction']}* ({exp['percentage']:.1f}% decision influence).")

# Continuous UI rerun loop when monitoring is active
if st.session_state.monitoring:
    time.sleep(1.0)
    st.rerun()
