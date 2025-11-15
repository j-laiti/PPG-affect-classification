"""
Wellby Dataset CNN Efficiency Evaluation
Table X in the paper
DOI: 10.1109/TAFFC.2025.3628467

Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""

# imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import time
import gc
import os
from pathlib import Path

# Configuration
output_filename = 'wellby_cnn_efficiency.csv'

class WellbyDilatedCNN(nn.Module):
    """
    Dilated CNN adapted for Wellby variable-length signals (typically shorter than WESAD/AKTIVES)
    Uses smaller dilations suitable for ~30-60 second recordings at 50Hz
    """
    def __init__(self, input_length=3000, num_classes=2):  # ~60s at 50Hz
        super(WellbyDilatedCNN, self).__init__()

        # Smaller dilation rates for shorter Wellby signals
        self.conv_blocks = nn.ModuleList([
            # Block 1: dilation=1, capture fine details
            nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=7, dilation=1, padding='same'),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.4)
            ),
            # Block 2: dilation=2
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=7, dilation=2, padding='same'),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5)
            ),
            # Block 3: dilation=4 
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=7, dilation=4, padding='same'),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.6)
            ),
            # Block 4: dilation=8
            nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=7, dilation=8, padding='same'),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.7)  # Increased from 0.4
            ),
            # Block 5: dilation=16
            nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=7, dilation=16, padding='same'),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.7)  # Increased from 0.4
            )
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)

        x = self.global_pool(x)
        x = self.classifier(x)
        return x

class PPGDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx].unsqueeze(0), self.labels[idx]

def load_wellby_data():
    """Load Wellby data using existing data loading approach"""
    print("Loading Wellby data...")
    
    ppg_data = pd.read_csv("../../data/Wellby/selected_PPG_data.csv")
    session_info = pd.read_csv("../../data/Wellby/Wellby_all_subjects_features.csv")
    
    ppg_session_ids = ppg_data.columns.tolist()
    info_session_ids = session_info['Session_ID'].tolist()
    
    common_sessions = list(set(ppg_session_ids) & set(info_session_ids))
    
    if len(common_sessions) == 0:
        return None, None, None
    
    aligned_data = []
    
    for session_id in common_sessions:
        ppg_signal = ppg_data[session_id].dropna().values
        session_row = session_info[session_info['Session_ID'] == session_id].iloc[0]
        
        aligned_data.append({
            'session_id': session_id,
            'participant': session_row['Participant'],
            'ppg_signal': ppg_signal,
            'stress_label': session_row['stress_binary'],
            'school': session_row['School'],
            'age': session_row['Age'],
            'gender': session_row['Gender'],
            'PSS': session_row['PSS'],
            'PSQI': session_row['PSQI'],
            'EPOCH': session_row['EPOCH'],
            'SQI': session_row['SQI']
        })
    
    return aligned_data

def prepare_wellby_cnn_data(aligned_data, target_length=None, skip_start_samples=0):
    """Prepare Wellby PPG signals for CNN training with variable length handling"""
    
    if target_length is None:
        signal_lengths = [len(item['ppg_signal']) - skip_start_samples for item in aligned_data 
                         if len(item['ppg_signal']) > skip_start_samples]
        if len(signal_lengths) == 0:
            return None, None, None
        target_length = min(signal_lengths)
        print(f"Using target length: {target_length} samples")
    
    all_signals = []
    all_labels = []
    all_participants = []
    
    for item in aligned_data:
        ppg_signal = item['ppg_signal']
        
        if len(ppg_signal) <= skip_start_samples:
            continue
            
        trimmed_signal = ppg_signal[skip_start_samples:]
        
        if len(trimmed_signal) >= target_length:
            processed_signal = trimmed_signal[:target_length]
        else:
            padding = target_length - len(trimmed_signal)
            processed_signal = np.pad(trimmed_signal, (0, padding), mode='constant', constant_values=0)
        
        clean_signal = processed_signal[~np.isnan(processed_signal)]
        if len(clean_signal) == 0:
            continue
        
        # Standardization for CNN (same as hybrid approach)
        ppg_mean = np.mean(clean_signal)
        ppg_std = np.std(clean_signal)
        if ppg_std == 0:
            ppg_std = 1e-8
        standardized_signal = (clean_signal - ppg_mean) / ppg_std
        
        # Additional normalization
        normalized_signal = (standardized_signal - np.mean(standardized_signal)) / (np.std(standardized_signal) + 1e-8)
        
        all_signals.append(normalized_signal)
        all_labels.append(int(item['stress_label']))
        all_participants.append(item['participant'])
    
    return np.array(all_signals), np.array(all_labels), np.array(all_participants)

def train_cnn_deployment_wellby(X_train, y_train, target_length, device, epochs=30):
    """Train CNN for deployment using ALL Wellby data"""
    print(f"Training CNN for deployment...")
    print(f"Training data: {len(X_train)} windows")
    print(f"Label distribution: {np.bincount(y_train)}")
    print(f"Signal length: {target_length} samples")

    # Train/val split for monitoring
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Create data loaders
    train_dataset = PPGDataset(X_tr, y_tr)
    val_dataset = PPGDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)  # Small batch for small dataset
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize model
    model = WellbyDilatedCNN(input_length=target_length).to(device)

    # Calculate class weights
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)  # Lower LR for small dataset

    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")

    best_val_acc = 0.0
    patience_counter = 0
    patience = 29

    for epoch in range(epochs):
        # Training
        model.train()
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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            epoch_total += batch_labels.size(0)
            epoch_correct += (predicted == batch_labels).sum().item()

        # Validation
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

        avg_loss = epoch_loss / len(train_loader)
        train_acc = 100 * epoch_correct / epoch_total
        val_acc = 100 * val_correct / val_total
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_wellby_cnn.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        torch.cuda.empty_cache()

    # Load best model
    model.load_state_dict(torch.load('best_wellby_cnn.pth'))
    print(f"CNN deployment training completed. Best val acc: {best_val_acc:.2f}%")
    
    return model

def run_wellby_cnn_deployment_timing(signals, labels, participants, target_length):
    """Measure CNN training time for deployment scenario"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("="*70)
    print("WELLBY CNN DEPLOYMENT TRAINING TIME MEASUREMENT")
    print("="*70)

    print(f"Training data:")
    print(f"  Total subjects: {len(np.unique(participants))}")
    print(f"  Total windows: {len(signals)}")
    print(f"  Signal length: {target_length} samples (~{target_length/50:.1f}s at 50Hz)")
    print(f"  Label distribution: {np.bincount(labels)}")

    # ===== TIMING MEASUREMENT =====
    print(f"Starting CNN Training...")
    training_start_time = time.perf_counter()
    
    trained_model = train_cnn_deployment_wellby(signals, labels, target_length, device, epochs=30)
    
    training_end_time = time.perf_counter()
    cnn_training_time = training_end_time - training_start_time
    # ===== END TIMING =====

    print(f"CNN Training completed in {cnn_training_time:.3f} seconds")

    # Calculate metrics
    training_time_per_sample_ms = (cnn_training_time * 1000) / len(signals)

    print(f"\n" + "="*70)
    print("WELLBY CNN DEPLOYMENT TRAINING TIME RESULTS")
    print("="*70)
    
    print(f"Training Configuration:")
    print(f"  Training samples: {len(signals)}")
    print(f"  Training subjects: {len(np.unique(participants))}")
    print(f"  Signal length: {target_length} samples")
    print(f"  Epochs: 30")
    print(f"  Device: {device}")
    
    print(f"\nTiming Results:")
    print(f"  Total training time: {cnn_training_time:.3f} seconds ({cnn_training_time/60:.1f} minutes)")
    print(f"  Training time per sample: {training_time_per_sample_ms:.3f} ms")

    # Save results
    results = {
        'approach': 'end_to_end_cnn_deployment_wellby',
        'training_time_seconds': cnn_training_time,
        'training_time_per_sample_ms': training_time_per_sample_ms,
        'training_samples': len(signals),
        'training_subjects': len(np.unique(participants)),
        'signal_length': target_length,
        'device': str(device)
    }
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return results, trained_model

def measure_wellby_cnn_inference_time(trained_model, test_signal, device, n_tests=100):
    """Measure CNN inference time using Wellby test signal"""
    print(f"\n=== WELLBY CNN Inference Time Measurement ===")
    print(f"Number of tests: {n_tests}")
    
    trained_model.eval()
    
    inference_times = []
    successful_predictions = 0
    
    for i in range(n_tests):
        try:
            # Prepare input tensor
            input_tensor = torch.FloatTensor(test_signal).unsqueeze(0).unsqueeze(0).to(device)
            
            # Time the inference
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = trained_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1)
            
            end_time = time.perf_counter()
            
            inference_time_ms = (end_time - start_time) * 1000
            inference_times.append(inference_time_ms)
            successful_predictions += 1
            
        except Exception as e:
            print(f"Error in inference {i}: {e}")
            continue
    
    if len(inference_times) == 0:
        return np.nan, np.nan
    
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    print(f"Inference timing completed!")
    print(f"Successful predictions: {successful_predictions}/{n_tests}")
    print(f"Average inference time: {avg_inference_time:.3f} ± {std_inference_time:.3f} ms")
    
    return avg_inference_time, std_inference_time

def main_wellby_cnn_deployment_timing():
    """Main function for Wellby CNN deployment timing measurement"""
    print("=== Wellby End-to-End CNN Deployment Training Time ===")

    try:
        # Load data
        aligned_data = load_wellby_data()
        if aligned_data is None:
            print("Error: Could not load Wellby data!")
            return None

        # Prepare CNN data
        signals, labels, participants = prepare_wellby_cnn_data(aligned_data)
        if signals is None:
            print("Error: Could not prepare CNN data!")
            return None

        print(f"Loaded {len(signals)} windows from {len(np.unique(participants))} subjects")

        # Run deployment timing measurement
        target_length = len(signals[0])  # Get actual signal length
        timing_results, trained_model = run_wellby_cnn_deployment_timing(signals, labels, participants, target_length)

        if timing_results:
            print(f"\n=== Wellby CNN Deployment Timing Complete ===")
            print(f"CNN deployment training time: {timing_results['training_time_seconds']:.3f}s")
            
            # Measure inference time using first signal as test
            avg_inf_time, std_inf_time = measure_wellby_cnn_inference_time(
                trained_model, signals[0], timing_results['device'], n_tests=100
            )
            
            # Add inference timing to results
            timing_results['avg_inference_time_ms'] = avg_inf_time
            timing_results['std_inference_time_ms'] = std_inf_time
            
            print(f"Inference Time: {avg_inf_time:.3f} ± {std_inf_time:.3f} ms")
            
            # Save results
            results_df = pd.DataFrame([timing_results])
            results_df.to_csv(output_filename, index=False)
            
        # Cleanup
        if os.path.exists('best_wellby_cnn.pth'):
            os.remove('best_wellby_cnn.pth')
            
        return timing_results
        
    except Exception as e:
        print(f"Error in Wellby CNN deployment timing: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    print("=== Wellby End-to-End CNN Deployment Timing ===")
    
    results = main_wellby_cnn_deployment_timing()
    
    if results:
        print(f"\n=== Wellby CNN Deployment Complete ===")
        print(f"Training time: {results['training_time_seconds']:.3f}s")
        print(f"Inference time: {results.get('avg_inference_time_ms', 'N/A'):.3f}ms")
        print(f"Compare with Wellby hybrid approach training + inference times")
    else:
        print("Deployment timing failed!")
    
    return results

if __name__ == "__main__":
    results = main()