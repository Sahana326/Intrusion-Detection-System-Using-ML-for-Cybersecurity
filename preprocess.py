import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Target list of key features we will use for our NIDS models
FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Bwd Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "SYN Flag Count",
    "ACK Flag Count",
    "Fwd Header Length"
]

def generate_mock_dataset(num_samples=5000, output_path="data/raw/network_traffic.csv"):
    """
    Generates a highly realistic mock network flow dataset for NIDS training.
    Includes Benign traffic and several common attack profiles (DoS, PortScan, Brute Force, Infiltration).
    """
    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = []
    
    # We will generate 60% Benign, 15% DoS, 15% PortScan, 7% Brute Force, 3% Infiltration
    categories = ["BENIGN", "DoS", "PortScan", "BruteForce", "Infiltration"]
    probs = [0.60, 0.15, 0.15, 0.07, 0.03]
    
    labels = np.random.choice(categories, size=num_samples, p=probs)
    
    for i in range(num_samples):
        label = labels[i]
        
        # Base attributes
        dest_port = 80
        flow_duration = 0.0
        tot_fwd_pkts = 1
        tot_bwd_pkts = 0
        tot_len_fwd = 0
        tot_len_bwd = 0
        fwd_pkt_max = 0
        fwd_pkt_mean = 0.0
        bwd_pkt_max = 0
        bwd_pkt_mean = 0.0
        syn_count = 0
        ack_count = 0
        fwd_hdr_len = 20
        
        if label == "BENIGN":
            # Normal HTTP/HTTPS/SSH traffic
            dest_port = int(np.random.choice([80, 443, 22, 53, 123]))
            flow_duration = float(np.random.exponential(scale=500000)) # 0.5 sec
            tot_fwd_pkts = int(np.random.randint(2, 50))
            tot_bwd_pkts = int(np.random.randint(2, 50))
            
            # Normal sizes
            fwd_pkt_max = int(np.random.randint(64, 1500))
            fwd_pkt_mean = float(np.random.uniform(64, 800))
            bwd_pkt_max = int(np.random.randint(64, 1500))
            bwd_pkt_mean = float(np.random.uniform(64, 800))
            
            tot_len_fwd = int(tot_fwd_pkts * fwd_pkt_mean)
            tot_len_bwd = int(tot_bwd_pkts * bwd_pkt_mean)
            
            syn_count = int(np.random.choice([0, 1]))
            ack_count = int(np.random.randint(2, tot_fwd_pkts + tot_bwd_pkts))
            fwd_hdr_len = int(tot_fwd_pkts * 20)
            
        elif label == "DoS":
            # DoS: High volume of small packets to a specific target port (usually port 80)
            dest_port = 80
            flow_duration = float(np.random.uniform(10, 5000)) # very short
            tot_fwd_pkts = int(np.random.randint(100, 1000)) # massive amount of packets
            tot_bwd_pkts = int(np.random.randint(0, 5)) # very few replies
            
            # Small, uniform packet sizes to exhaust resources quickly
            fwd_pkt_max = int(np.random.randint(40, 64))
            fwd_pkt_mean = float(fwd_pkt_max)
            bwd_pkt_max = int(np.random.choice([0, 40]))
            bwd_pkt_mean = float(bwd_pkt_max)
            
            tot_len_fwd = int(tot_fwd_pkts * fwd_pkt_mean)
            tot_len_bwd = int(tot_bwd_pkts * bwd_pkt_mean)
            
            syn_count = tot_fwd_pkts # SYN Flood
            ack_count = 0
            fwd_hdr_len = int(tot_fwd_pkts * 20)
            
        elif label == "PortScan":
            # PortScan: Probing many different ports, small flows
            dest_port = int(np.random.randint(1, 65535))
            flow_duration = float(np.random.uniform(1, 100)) # instant
            tot_fwd_pkts = int(np.random.choice([1, 2])) # usually just 1 SYN packet
            tot_bwd_pkts = int(np.random.choice([0, 1]))
            
            fwd_pkt_max = int(np.random.choice([0, 40]))
            fwd_pkt_mean = float(fwd_pkt_max)
            bwd_pkt_max = 0
            bwd_pkt_mean = 0.0
            
            tot_len_fwd = int(tot_fwd_pkts * fwd_pkt_mean)
            tot_len_bwd = 0
            
            syn_count = tot_fwd_pkts
            ack_count = 0
            fwd_hdr_len = int(tot_fwd_pkts * 20)
            
        elif label == "BruteForce":
            # SSH/FTP brute forcing (port 22 or 21), high packet count, regular intervals
            dest_port = int(np.random.choice([22, 21]))
            flow_duration = float(np.random.uniform(1000000, 10000000)) # 1-10 seconds
            tot_fwd_pkts = int(np.random.randint(20, 100))
            tot_bwd_pkts = int(np.random.randint(20, 100))
            
            fwd_pkt_max = int(np.random.randint(64, 256))
            fwd_pkt_mean = float(np.random.uniform(40, 128))
            bwd_pkt_max = int(np.random.randint(64, 256))
            bwd_pkt_mean = float(np.random.uniform(40, 128))
            
            tot_len_fwd = int(tot_fwd_pkts * fwd_pkt_mean)
            tot_len_bwd = int(tot_bwd_pkts * bwd_pkt_mean)
            
            syn_count = 1
            ack_count = int(tot_fwd_pkts + tot_bwd_pkts - 2)
            fwd_hdr_len = int(tot_fwd_pkts * 20)
            
        elif label == "Infiltration":
            # Infiltration: slow, stealthy data exfiltration, large backward packet volume
            dest_port = int(np.random.randint(1024, 49151))
            flow_duration = float(np.random.uniform(5000000, 60000000)) # 5-60 seconds
            tot_fwd_pkts = int(np.random.randint(50, 500))
            tot_bwd_pkts = int(np.random.randint(100, 2000)) # heavy download
            
            fwd_pkt_max = int(np.random.randint(64, 500))
            fwd_pkt_mean = float(np.random.uniform(64, 200))
            bwd_pkt_max = 1460 # large packet size (exfiltration/download)
            bwd_pkt_mean = float(np.random.uniform(1000, 1460))
            
            tot_len_fwd = int(tot_fwd_pkts * fwd_pkt_mean)
            tot_len_bwd = int(tot_bwd_pkts * bwd_pkt_mean)
            
            syn_count = 1
            ack_count = int(tot_fwd_pkts + tot_bwd_pkts - 2)
            fwd_hdr_len = int(tot_fwd_pkts * 20)

        # Calculate bytes/s and packets/s safely
        duration_sec = flow_duration / 1e6
        if duration_sec > 0:
            flow_bytes_s = (tot_len_fwd + tot_len_bwd) / duration_sec
            flow_pkts_s = (tot_fwd_pkts + tot_bwd_pkts) / duration_sec
        else:
            flow_bytes_s = 0.0
            flow_pkts_s = 0.0
            
        data.append([
            dest_port, flow_duration, tot_fwd_pkts, tot_bwd_pkts,
            tot_len_fwd, tot_len_bwd, fwd_pkt_max, fwd_pkt_mean,
            bwd_pkt_max, bwd_pkt_mean, flow_bytes_s, flow_pkts_s,
            syn_count, ack_count, fwd_hdr_len, label
        ])
        
    columns = FEATURES + ["Label"]
    df = pd.DataFrame(data, columns=columns)
    
    # Save to disk
    df.to_csv(output_path, index=False)
    print(f"Mock dataset generated successfully at: {output_path} ({num_samples} samples)")
    return df

def load_and_preprocess_data(dataset_path="data/raw/network_traffic.csv", test_size=0.2, scaler_path="models/scaler.pkl"):
    """
    Loads raw CSV dataset, performs normalization scaling, and splits into train/test sets.
    Saves the scaler binary to disk.
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Generating mock dataset...")
        df = generate_mock_dataset(output_path=dataset_path)
    else:
        df = pd.read_csv(dataset_path)
        
    X = df[FEATURES].copy()
    y = df["Label"].copy()
    
    # Handle infinite/NaN values from division by zero in packet/byte rates
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Return as DataFrames / series to retain column context where needed
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=FEATURES)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=FEATURES)
    
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler

if __name__ == "__main__":
    # Test generation and preprocessing
    generate_mock_dataset(100)
    load_and_preprocess_data("data/raw/network_traffic.csv")
