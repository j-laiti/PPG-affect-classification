
"""
WESAD End-to-End CNN Deployment Training Time Measurement
Trains CNN on ALL subjects (deployment scenario) for fair comparison with hybrid/TD approaches
Table X in the paper
DOI: 10.1109/TAFFC.2025.3628467

Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""
#%% Imports

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from pathlib import Path
import time
import gc

class FixedDilatedCNN(nn.Module):
    """
    Improved Dilated CNN fixing issues from Motaman et al.
    - Reasonable dilation rates for 120s windows
    - Proper output layer for CrossEntropyLoss
    - More efficient architecture
    """
    def __init__(self, input_length=7680, num_classes=2):
        super(FixedDilatedCNN, self).__init__()

        # More reasonable dilation progression
        # For 7680 length signal (120s), max dilation of 16 is sufficient
        self.conv_blocks = nn.ModuleList([
            # Block 1: dilation=1, capture fine details
            nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=7, dilation=1, padding='same'),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2)
            ),
            # Block 2: dilation=2
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=7, dilation=2, padding='same'),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3)
            ),
            # Block 3: dilation=4
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=7, dilation=4, padding='same'),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3)
            ),
            # Block 4: dilation=8
            nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=7, dilation=8, padding='same'),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4)
            ),
            # Block 5: dilation=16 (final)
            nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=7, dilation=16, padding='same'),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.4)
            )
        ])

        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)  # NO sigmoid - raw logits for CrossEntropyLoss
        )

    def forward(self, x):
        # Apply dilated conv blocks
        for block in self.conv_blocks:
            x = block(x)

        # Global pooling and classification
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

class PPGDataset(Dataset):
    """Optimized Dataset class for GPU training"""
    def __init__(self, windows, labels):
        # Convert to tensors immediately
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx].unsqueeze(0), self.labels[idx]

def load_wesad_data_colab(data_dir="./"):
    """
    Load WESAD data for Colab/Kaggle environment
    Adjust data_dir based on where you uploaded files
    """
    print("Loading WESAD data...")
    data_dir = Path(data_dir)

    all_subjects = []
    all_windows = []
    all_labels = []

    # Look for CSV files
    csv_files = list(data_dir.glob("S*.csv"))
    if not csv_files:
        # Try different common locations
        possible_dirs = ["./data/", "./WESAD_BVP_extracted/", "/content/", "/kaggle/input/"]
        for pdir in possible_dirs:
            csv_files = list(Path(pdir).glob("S*.csv"))
            if csv_files:
                data_dir = Path(pdir)
                break

    print(f"Found {len(csv_files)} CSV files in {data_dir}")

    for subject_file in sorted(csv_files):
        subject_id = subject_file.stem
        print(f"Processing {subject_id}...")

        try:
            # Load CSV
            df = pd.read_csv(subject_file)

            if 'BVP' not in df.columns or 'label' not in df.columns:
                print(f"  Skipping {subject_id}: missing BVP or label columns")
                continue

            # Extract labeled segments
            windows, labels, subject_ids = extract_labeled_segments(
                df['BVP'].values, df['label'].values, subject_id
            )

            if len(windows) > 0:
                all_windows.extend(windows)
                all_labels.extend(labels)
                all_subjects.extend(subject_ids)
                print(f"  {subject_id}: Created {len(windows)} windows")

        except Exception as e:
            print(f"  Error with {subject_id}: {e}")
            continue

    print(f"\nTotal windows: {len(all_windows)}")
    if len(all_labels) > 0:
        print(f"Label distribution: {np.bincount(all_labels)}")

    return np.array(all_windows), np.array(all_labels), np.array(all_subjects)

def extract_labeled_segments(bvp_signal, labels, subject_id, fs=64):
    """Extract windows from labeled segments - SAME as your existing function"""
    window_size = int(120 * fs)  # 120 seconds = 7680 samples
    step_size = int(30 * fs)     # 30 seconds = 1920 samples

    windows = []
    window_labels = []
    subject_ids = []

    # Find labeled segments
    label_series = pd.Series(labels)
    valid_mask = label_series.notna()

    if not valid_mask.any():
        return windows, window_labels, subject_ids

    # Process stress (1.0) and non-stress (0.0) segments
    for label_val in [0.0, 1.0]:
        label_mask = (label_series == label_val) & valid_mask

        if not label_mask.any():
            continue

        # Find continuous segments
        label_indices = np.where(label_mask)[0]

        # Create windows within segments
        for start_idx in range(0, len(label_indices) - window_size + 1, step_size):
            if start_idx + window_size <= len(label_indices):
                # Get indices for this window
                window_indices = label_indices[start_idx:start_idx + window_size]

                # Check if indices are continuous
                if np.all(np.diff(window_indices) == 1):
                    window = bvp_signal[window_indices]

                    # Normalize window (same as your existing preprocessing)
                    window = (window - np.mean(window)) / (np.std(window) + 1e-8)

                    windows.append(window)
                    window_labels.append(int(label_val))
                    subject_ids.append(subject_id)

    return windows, window_labels, subject_ids

def train_cnn_deployment(X_train, y_train, device, epochs=30):
    """
    Train CNN for deployment (using ALL data)
    Simplified for timing measurement
    """
    print(f"Training CNN for deployment...")
    print(f"Training data: {len(X_train)} windows")
    print(f"Label distribution: {np.bincount(y_train)}")

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train).unsqueeze(1)
    y_tensor = torch.LongTensor(y_train)

    # Simple train/val split for monitoring (optional)
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor)

    # Create data loaders
    train_dataset = PPGDataset(X_tr.squeeze(1), y_tr)  # Remove extra dimension
    val_dataset = PPGDataset(X_val.squeeze(1), y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize model
    model = FixedDilatedCNN().to(device)

    # Calculate class weights
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_windows, batch_labels in train_loader:
            batch_windows = batch_windows.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(batch_windows)
            loss = criterion(outputs, batch_labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            epoch_total += batch_labels.size(0)
            epoch_correct += (predicted == batch_labels).sum().item()

        # Quick validation check
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_windows, batch_labels in val_loader:
                batch_windows = batch_windows.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_windows)
                _, predicted = torch.max(outputs, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        model.train()

        avg_loss = epoch_loss / len(train_loader)
        train_acc = 100 * epoch_correct / epoch_total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')

        # Memory cleanup
        torch.cuda.empty_cache()

    print(f"CNN deployment training completed!")
    return model

def run_cnn_deployment_timing(windows, labels, subjects):
    """
    Measure CNN training time for deployment scenario (ALL subjects)
    This matches the approach used for TD-only and Hybrid methods
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("="*70)
    print("CNN DEPLOYMENT TRAINING TIME MEASUREMENT")
    print("="*70)
    print("Training CNN on ALL subjects (deployment scenario)")
    print("This matches the training approach used for TD-only and Hybrid methods")

    # Use ALL subjects for training (deployment scenario)
    all_windows = windows
    all_labels = labels
    
    print(f"\nTraining data:")
    print(f"  Total subjects: {len(np.unique(subjects))}")
    print(f"  Total windows: {len(all_windows)}")
    print(f"  Label distribution: {np.bincount(all_labels)}")

    # ===== TIMING MEASUREMENT =====
    print(f"\n--- Starting CNN Training Timing ---")
    training_start_time = time.perf_counter()
    
    # Train CNN on all data (1 epoch for timing, can adjust)
    trained_model = train_cnn_deployment(all_windows, all_labels, device, epochs=30)
    
    training_end_time = time.perf_counter()
    cnn_training_time = training_end_time - training_start_time
    # ===== END TIMING =====

    print(f"--- CNN Training Timing Complete ---")

    # Calculate additional metrics
    training_time_per_sample_ms = (cnn_training_time * 1000) / len(all_windows)

    # Save results
    results = {
        'approach': 'end_to_end_cnn_deployment',
        'training_time_epoch': cnn_training_time,
        'training_samples': len(all_windows),
        'training_subjects': len(np.unique(subjects)),
        'device': str(device)
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('cnn_deployment_timing_results.csv', index=False)
    print(f"\nResults saved to 'cnn_deployment_timing_results.csv'")
    
    # Cleanup - but keep trained_model for inference testing
    torch.cuda.empty_cache()
    gc.collect()
    
    return results, trained_model  # Return model for inference testing

def measure_cnn_inference_time(trained_model, test_signal_path, device, n_tests=100):
    """
    Measure CNN inference time using a test signal
    
    Args:
        trained_model: Pre-trained CNN model
        test_signal_path: Path to test CSV file with raw PPG signal
        device: Training device
        n_tests: Number of inference runs for averaging
    """
    print(f"\n=== CNN Inference Time Measurement ===")
    print(f"Test signal: {test_signal_path}")
    print(f"Number of tests: {n_tests}")
    
    # Load test signal
    try:
        test_df = pd.read_csv(test_signal_path)
        raw_ppg = test_df.iloc[:, 0].values  # First column should be PPG
        
        # Remove NaN values
        clean_ppg = raw_ppg[~np.isnan(raw_ppg)]
        
        # Extract window for inference (120s = 7680 samples at 64Hz)
        window_samples = 120 * 64
        if len(clean_ppg) >= window_samples:
            test_window = clean_ppg[:window_samples]
        else:
            # Pad if necessary
            test_window = np.pad(clean_ppg, (0, window_samples - len(clean_ppg)), mode='edge')
        
        # Normalize window (same preprocessing as training)
        window_norm = (test_window - np.mean(test_window)) / (np.std(test_window) + 1e-8)
        print(f"Test window prepared: {len(window_norm)} samples")
        
    except Exception as e:
        print(f"Error loading test signal: {e}")
        return np.nan, np.nan
    
    # Prepare model for inference
    trained_model.eval()
    
    inference_times = []
    successful_predictions = 0
    
    # Run inference timing tests
    for i in range(n_tests):
        try:
            # Prepare input tensor
            input_tensor = torch.FloatTensor(window_norm).unsqueeze(0).unsqueeze(0).to(device)
            
            # Time the inference
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = trained_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1)
            
            end_time = time.perf_counter()
            
            inference_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            inference_times.append(inference_time_ms)
            successful_predictions += 1
            
        except Exception as e:
            print(f"Error in inference {i}: {e}")
            continue
    
    if len(inference_times) == 0:
        print("No successful inferences!")
        return np.nan, np.nan
    
    # Calculate statistics
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    print(f"Inference timing completed!")
    print(f"Successful predictions: {successful_predictions}/{n_tests}")
    print(f"Average inference time: {avg_inference_time:.3f} ± {std_inference_time:.3f} ms")
    
    return avg_inference_time, std_inference_time

def main_deployment_timing():
    """Main function for CNN deployment timing measurement"""
    print("=== End-to-End CNN Deployment Training Time ===")
    print("Training CNN on ALL subjects for fair comparison with TD/Hybrid approaches")

    # Load data
    data_dir = "../../data/WESAD_BVP_extracted/"  # Adjust path as needed
    windows, labels, subjects = load_wesad_data_colab(data_dir)

    if len(windows) == 0:
        print("No data found! Check your data directory.")
        return

    print(f"Loaded {len(windows)} windows from {len(np.unique(subjects))} subjects")

    # Run deployment timing measurement
    timing_results, trained_model = run_cnn_deployment_timing(windows, labels, subjects)

    # if timing_results:
    #     print(f"\n=== CNN Deployment Timing Complete ===")
        
    #     # Measure inference time
    #     test_signal_path = "../../data/WESAD_inference_test.csv"  # Adjust path as needed
    #     if os.path.exists(test_signal_path):
    #         avg_inf_time, std_inf_time = measure_cnn_inference_time(
    #             trained_model, test_signal_path, timing_results['device'], n_tests=100
    #         )
            
    #         # Add inference timing to results
    #         timing_results['avg_inference_time_ms'] = avg_inf_time
    #         timing_results['std_inference_time_ms'] = std_inf_time
            
    #         print(f"\nInference Time: {avg_inf_time:.3f} ± {std_inf_time:.3f} ms")
            
    #         # Save updated results
    results_df = pd.DataFrame([timing_results])
    results_df.to_csv('cnn_deployment_timing_results.csv', index=False)
    #         print(f"Complete results saved to 'cnn_deployment_timing_results.csv'")
            
    #     else:
    #         print(f"Test signal file not found: {test_signal_path}")
    #         print(f"Skipping inference time measurement")
    
    return timing_results

if __name__ == "__main__":
    results = main_deployment_timing()