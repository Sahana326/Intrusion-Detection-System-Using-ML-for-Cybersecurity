import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import joblib

# Check if Scapy is available (for live sniffing)
try:
    from scapy.all import sniff, IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

from preprocess import FEATURES

class HybridIDSDetector:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.scaler = None
        self.anomaly_detector = None
        self.classifier = None
        self.metadata = None
        self.is_loaded = False
        
        self.load_models()

    def load_models(self):
        try:
            self.scaler = joblib.load(os.path.join(self.models_dir, "scaler.pkl"))
            self.anomaly_detector = joblib.load(os.path.join(self.models_dir, "anomaly_detector.pkl"))
            self.classifier = joblib.load(os.path.join(self.models_dir, "classifier.pkl"))
            self.metadata = joblib.load(os.path.join(self.models_dir, "metadata.pkl"))
            self.is_loaded = True
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models from '{self.models_dir}': {e}")
            print("Please run train.py first to train the models.")
            self.is_loaded = False

    def predict_flow(self, flow_features):
        """
        Runs the hybrid detection logic:
        1. Scales features.
        2. Detects anomalies using Isolation Forest (anomaly score).
        3. Identifies the attack type using the Multi-class Random Forest.
        4. Calculates a quick explainability profile (feature attribution).
        """
        if not self.is_loaded:
            return {"status": "error", "message": "Models not loaded."}

        # Convert dictionary to DataFrame with feature names
        features_dict = {f: [flow_features.get(f, 0.0)] for f in FEATURES}
        features_df = pd.DataFrame(features_dict)
        
        # Scale
        scaled_arr = self.scaler.transform(features_df)
        scaled_df = pd.DataFrame(scaled_arr, columns=FEATURES)

        # Stage 1: Isolation Forest Anomaly Detection
        # Isolation Forest predict returns -1 for outliers (anomalies) and 1 for inliers.
        # anomaly_score measures abnormality (lower score = more anomalous)
        anomaly_pred = self.anomaly_detector.predict(scaled_df)[0]
        anomaly_score = self.anomaly_detector.score_samples(scaled_df)[0]
        # Invert score so higher = more anomalous (for visualization)
        anomaly_likelihood = float(np.clip(-anomaly_score * 2.0, 0.0, 1.0))
        is_anomaly = bool(anomaly_pred == -1)

        # Stage 2: Classifier Prediction
        probabilities = self.classifier.predict_proba(scaled_df)[0]
        pred_class = self.classifier.predict(scaled_df)[0]
        class_prob = float(probabilities[list(self.classifier.classes_).index(pred_class)])

        # Explainability (Local Feature Attribution)
        # Compute how much each scaled feature deviates from its normal standard scaled mean (0)
        # Weight this deviation by the Random Forest classifier's global feature importances
        deviations = scaled_arr[0]
        importances = self.classifier.feature_importances_
        attributions = deviations * importances
        
        # Format XAI explanations
        xai_metrics = {}
        for feature, attr in zip(FEATURES, attributions):
            xai_metrics[feature] = float(attr)

        # Sort and get top factors
        top_factors = sorted(xai_metrics.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
        explanations = []
        for feat, val in top_factors:
            direction = "Elevated" if val > 0 else "Suppressed"
            importance_pct = abs(val) / (sum(abs(v) for v in xai_metrics.values()) + 1e-9) * 100
            explanations.append({
                "feature": feat,
                "attribution": val,
                "direction": direction,
                "percentage": float(np.clip(importance_pct, 5.0, 95.0))
            })

        # Override predictions if normal
        # A robust hybrid NIDS may treat flows as BENIGN if anomaly score is high enough (non-anomalous)
        # or if the classifier strongly labels it BENIGN.
        final_threat_label = pred_class
        risk_level = "Low"
        
        if final_threat_label == "BENIGN":
            risk_level = "Low"
        else:
            if class_prob > 0.8:
                risk_level = "Critical"
            elif class_prob > 0.5 or is_anomaly:
                risk_level = "High"
            else:
                risk_level = "Medium"

        return {
            "is_anomaly": is_anomaly,
            "anomaly_likelihood": anomaly_likelihood,
            "prediction": final_threat_label,
            "confidence": class_prob,
            "risk_level": risk_level,
            "explanations": explanations,
            "raw_features": flow_features
        }

class TrafficSimulator:
    """
    Simulates network flows for Benign, DoS, PortScan, BruteForce, and Infiltration,
    delivering them dynamically to test the system in real-time.
    """
    def __init__(self):
        self.attack_profiles = {
            "BENIGN": {
                "ports": [80, 443, 22, 53],
                "duration_range": (100000, 2000000),
                "fwd_pkts": (5, 30),
                "bwd_pkts": (5, 40),
                "pkt_len_fwd": (64, 1000),
                "pkt_len_bwd": (64, 1200),
                "syn_rate": 0.05,
                "ack_rate": 0.95
            },
            "DoS": {
                "ports": [80],
                "duration_range": (100, 3000),
                "fwd_pkts": (150, 600),
                "bwd_pkts": (0, 2),
                "pkt_len_fwd": (40, 64),
                "pkt_len_bwd": (0, 40),
                "syn_rate": 0.98,
                "ack_rate": 0.0
            },
            "PortScan": {
                "ports": range(1000, 5000),
                "duration_range": (1, 150),
                "fwd_pkts": (1, 2),
                "bwd_pkts": (0, 1),
                "pkt_len_fwd": (0, 40),
                "pkt_len_bwd": (0, 0),
                "syn_rate": 1.0,
                "ack_rate": 0.0
            },
            "BruteForce": {
                "ports": [22, 21],
                "duration_range": (2000000, 8000000),
                "fwd_pkts": (30, 80),
                "bwd_pkts": (30, 80),
                "pkt_len_fwd": (64, 256),
                "pkt_len_bwd": (64, 256),
                "syn_rate": 0.02,
                "ack_rate": 0.98
            },
            "Infiltration": {
                "ports": range(10000, 40000),
                "duration_range": (8000000, 30000000),
                "fwd_pkts": (60, 300),
                "bwd_pkts": (200, 1500),
                "pkt_len_fwd": (64, 500),
                "pkt_len_bwd": (1200, 1460),
                "syn_rate": 0.01,
                "ack_rate": 0.99
            }
        }

    def generate_flow(self, forced_label=None):
        label = forced_label if forced_label else random.choice(list(self.attack_profiles.keys()))
        profile = self.attack_profiles[label]

        # Port selection
        ports = profile["ports"]
        dest_port = random.choice(ports) if isinstance(ports, list) else random.choice(list(ports))

        # Generate base characteristics
        flow_duration = random.uniform(*profile["duration_range"])
        tot_fwd_pkts = random.randint(*profile["fwd_pkts"])
        tot_bwd_pkts = random.randint(*profile["bwd_pkts"])

        fwd_pkt_max = random.randint(*profile["pkt_len_fwd"])
        fwd_pkt_mean = random.uniform(fwd_pkt_max * 0.5, fwd_pkt_max)
        bwd_pkt_max = random.randint(*profile["pkt_len_bwd"]) if tot_bwd_pkts > 0 else 0
        bwd_pkt_mean = random.uniform(bwd_pkt_max * 0.5, bwd_pkt_max) if tot_bwd_pkts > 0 else 0.0

        tot_len_fwd = int(tot_fwd_pkts * fwd_pkt_mean)
        tot_len_bwd = int(tot_bwd_pkts * bwd_pkt_mean)

        syn_count = int(tot_fwd_pkts if profile["syn_rate"] > 0.8 else random.randint(0, 1))
        ack_count = int((tot_fwd_pkts + tot_bwd_pkts) * profile["ack_rate"])

        fwd_hdr_len = tot_fwd_pkts * 20

        # Calculate packet rates
        duration_sec = flow_duration / 1e6
        if duration_sec > 0:
            flow_bytes_s = (tot_len_fwd + tot_len_bwd) / duration_sec
            flow_packets_s = (tot_fwd_pkts + tot_bwd_pkts) / duration_sec
        else:
            flow_bytes_s = 0.0
            flow_packets_s = 0.0

        # Source / Dest IP mocks
        src_ip = f"192.168.1.{random.randint(10, 250)}"
        dst_ip = f"10.0.0.{random.randint(2, 20)}"

        flow_data = {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "protocol": "TCP" if dest_port in [80, 443, 22, 21] else "UDP",
            "Destination Port": dest_port,
            "Flow Duration": flow_duration,
            "Total Fwd Packets": tot_fwd_pkts,
            "Total Bwd Packets": tot_bwd_pkts,
            "Total Length of Fwd Packets": tot_len_fwd,
            "Total Length of Bwd Packets": tot_len_bwd,
            "Fwd Packet Length Max": fwd_pkt_max,
            "Fwd Packet Length Mean": fwd_pkt_mean,
            "Bwd Packet Length Max": bwd_pkt_max,
            "Bwd Packet Length Mean": bwd_pkt_mean,
            "Flow Bytes/s": flow_bytes_s,
            "Flow Packets/s": flow_packets_s,
            "SYN Flag Count": syn_count,
            "ACK Flag Count": ack_count,
            "Fwd Header Length": fwd_hdr_len,
            "timestamp": time.strftime("%H:%M:%S")
        }
        return flow_data, label

def run_simulation(detector, count=10, delay=1.5):
    """
    Runs the pipeline locally in a simulator loop and prints threat predictions to the CLI.
    """
    simulator = TrafficSimulator()
    print(f"\nStarting live flow simulation loop (count={count}, delay={delay}s)...")
    print("-" * 80)
    
    for i in range(count):
        flow_features, true_label = simulator.generate_flow()
        result = detector.predict_flow(flow_features)
        
        # Pretty print prediction
        print(f"[{flow_features['timestamp']}] Flow: {flow_features['src_ip']} -> {flow_features['dst_ip']}:{flow_features['Destination Port']} ({flow_features['protocol']})")
        print(f"    - True Label: {true_label} | Model Prediction: {result['prediction']} ({result['confidence']*100:.1f}%)")
        print(f"    - Stage 1 Anomaly Likelihood: {result['anomaly_likelihood']*100:.1f}% | Risk Level: {result['risk_level']}")
        
        if result['risk_level'] != "Low":
            print(f"    - [!] ALERT: Top factor: {result['explanations'][0]['feature']} ({result['explanations'][0]['direction']})")
        print("-" * 80)
        time.sleep(delay)

def start_live_sniffer(detector, interface=None):
    """
    Starts Scapy network interface sniffing. Runs feature compilation on incoming packets
    and performs model evaluations.
    """
    if not SCAPY_AVAILABLE:
        print("Scapy library is not installed or configured correctly. Live sniffing is unavailable.")
        return
        
    print(f"Starting Scapy Live Sniffer on interface '{interface if interface else 'Default'}'...")
    print("Listening to TCP/UDP packet flows. Press Ctrl+C to terminate.")
    
    # Store temporary active flow aggregations
    # Key: (src_ip, dst_ip, src_port, dst_port, protocol)
    flows = {}
    
    def packet_callback(pkt):
        if not (IP in pkt and (TCP in pkt or UDP in pkt)):
            return
            
        ip_layer = pkt[IP]
        proto = "TCP" if TCP in pkt else "UDP"
        
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        src_port = pkt.sport
        dst_port = pkt.dport
        
        flow_key = (src_ip, dst_ip, dst_port, proto)
        t_now = time.time()
        
        if flow_key not in flows:
            flows[flow_key] = {
                "start_time": t_now,
                "last_time": t_now,
                "fwd_packets": 0,
                "bwd_packets": 0,
                "fwd_lengths": [],
                "bwd_lengths": [],
                "syn_flags": 0,
                "ack_flags": 0,
                "fwd_hdr_len": 0
            }
            
        flow = flows[flow_key]
        flow["last_time"] = t_now
        
        # Check packet direction (outgoing/incoming)
        # Simplistic heuristic for demo
        is_fwd = True
        flow["fwd_packets"] += 1
        flow["fwd_lengths"].append(len(pkt))
        
        if TCP in pkt:
            flags = pkt[TCP].flags
            if 'S' in flags:
                flow["syn_flags"] += 1
            if 'A' in flags:
                flow["ack_flags"] += 1
            flow["fwd_hdr_len"] += pkt[TCP].dataofs * 4
        else:
            flow["fwd_hdr_len"] += 8 # UDP Header length
            
        # Process and predict flows if they have been active for > 2 seconds or gathered 20 packets
        duration = t_now - flow["start_time"]
        total_pkts = flow["fwd_packets"] + flow["bwd_packets"]
        
        if duration >= 2.0 or total_pkts >= 20:
            # compile flow features
            flow_duration_us = duration * 1e6
            tot_fwd_pkts = flow["fwd_packets"]
            tot_bwd_pkts = flow["bwd_packets"]
            tot_len_fwd = sum(flow["fwd_lengths"])
            tot_len_bwd = sum(flow["bwd_lengths"])
            
            fwd_pkt_max = max(flow["fwd_lengths"]) if flow["fwd_lengths"] else 0
            fwd_pkt_mean = np.mean(flow["fwd_lengths"]) if flow["fwd_lengths"] else 0.0
            bwd_pkt_max = max(flow["bwd_lengths"]) if flow["bwd_lengths"] else 0
            bwd_pkt_mean = np.mean(flow["bwd_lengths"]) if flow["bwd_lengths"] else 0.0
            
            flow_bytes_s = (tot_len_fwd + tot_len_bwd) / duration if duration > 0 else 0.0
            flow_pkts_s = total_pkts / duration if duration > 0 else 0.0
            
            flow_features = {
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "protocol": proto,
                "Destination Port": dst_port,
                "Flow Duration": flow_duration_us,
                "Total Fwd Packets": tot_fwd_pkts,
                "Total Bwd Packets": tot_bwd_pkts,
                "Total Length of Fwd Packets": tot_len_fwd,
                "Total Length of Bwd Packets": tot_len_bwd,
                "Fwd Packet Length Max": fwd_pkt_max,
                "Fwd Packet Length Mean": fwd_pkt_mean,
                "Bwd Packet Length Max": bwd_pkt_max,
                "Bwd Packet Length Mean": bwd_pkt_mean,
                "Flow Bytes/s": flow_bytes_s,
                "Flow Packets/s": flow_pkts_s,
                "SYN Flag Count": flow["syn_flags"],
                "ACK Flag Count": flow["ack_flags"],
                "Fwd Header Length": flow["fwd_hdr_len"],
                "timestamp": time.strftime("%H:%M:%S")
            }
            
            # Predict
            result = detector.predict_flow(flow_features)
            
            # Clean up processed flow
            del flows[flow_key]
            
            # Display Alert
            print(f"[{flow_features['timestamp']}] Flow: {src_ip} -> {dst_ip}:{dst_port} | Class: {result['prediction']} ({result['confidence']*100:.1f}%) | Risk: {result['risk_level']}")
            if result['risk_level'] != "Low":
                print(f"    - [!] ALERT: Top threat indicators: {result['explanations'][0]['feature']} ({result['explanations'][0]['direction']})")
                print(f"    - Anomaly Probability: {result['anomaly_likelihood']*100:.1f}%")
                print("-" * 50)
                
    try:
        # Capture TCP and UDP packets on the interface
        sniff(filter="tcp or udp", prn=packet_callback, store=0, iface=interface)
    except Exception as e:
        print(f"Error executing Scapy Sniff: {e}")
        print("Note: Windows users must run terminal shell as Administrator to execute scapy packet captures.")

def main():
    parser = argparse.ArgumentParser(description="Hybrid NIDS Detection & Sniffing Tool")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory where ML model objects are saved")
    parser.add_argument("--simulate", action="store_true", help="Run traffic flow simulation loop instead of live capture")
    parser.add_argument("--interface", type=str, default=None, help="Network interface name to sniff on")
    args = parser.parse_args()

    detector = HybridIDSDetector(models_dir=args.models_dir)
    
    if not detector.is_loaded:
        print("Error: Model files not found. Creating mock dataset and training models first...")
        from train import train_models
        train_models(models_dir=args.models_dir)
        detector.load_models()

    if args.simulate:
        run_simulation(detector, count=10)
    else:
        start_live_sniffer(detector, interface=args.interface)

if __name__ == "__main__":
    main()
