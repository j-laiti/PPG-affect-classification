"""
AKTIVES Dataset CNN Efficiency Evaluation
Table X in the paper
DOI: 10.1109/TAFFC.2025.3628467

Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""
# %%===== MAIN CODE =====

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import gc
from tqdm import tqdm
import os

class FixedDilatedCNN(nn.Module):
    """
    Same CNN as WESAD - only input_length parameter changed
    """
    def __init__(self, input_length=1920, num_classes=2):
        super(FixedDilatedCNN, self).__init__()

        # Same dilation progression as WESAD
        self.conv_blocks = nn.ModuleList([
            # Block 1: dilation=1, capture fine details
            nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=7, dilation=1, padding='same'),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3)
            ),
            # Block 2: dilation=2
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=7, dilation=2, padding='same'),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.4)
            ),
            # Block 3: dilation=4
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=7, dilation=4, padding='same'),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4)
            ),
            # Block 4: dilation=8
            nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=7, dilation=8, padding='same'),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5)
            ),
            # Block 5: dilation=16 (EXACT same as WESAD)
            nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=7, dilation=16, padding='same'),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)  # Raw logits for CrossEntropyLoss
        )

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)

        # Global pooling and classification
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

class PPGDataset(Dataset):
    """Optimized Dataset class for GPU training"""
    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx].unsqueeze(0), self.labels[idx]

def load_all_aktives_windows_colab():
    """
    Load all AKTIVES analysis windows from all cohorts (Colab version)
    """
    print("Loading all AKTIVES analysis windows...")

    # Look for analysis windows in common locations
    possible_paths = [
        "../../data/Aktives/analysis_windows"
    ]

    windows_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            windows_dir = Path(path)
            break

    if windows_dir is None:
        # Try to find it recursively
        for root, dirs, files in os.walk("."):
            if "analysis_windows" in dirs:
                windows_dir = Path(root) / "analysis_windows"
                break

    if windows_dir is None:
        raise ValueError("Could not find analysis_windows directory!")

    print(f"Found analysis windows directory: {windows_dir}")

    cohorts = ['dyslexia', 'ID', 'OBPI', 'TD']
    all_windows = []

    for cohort in cohorts:
        window_file = windows_dir / f"analysis_windows_{cohort}.csv"

        if window_file.exists():
            windows_df = pd.read_csv(window_file)
            windows_df['cohort'] = cohort
            all_windows.append(windows_df)
            print(f"  Loaded {cohort}: {len(windows_df)} windows")
        else:
            print(f"  Warning: {window_file} not found")

    if not all_windows:
        raise ValueError("No analysis windows found!")

    combined_windows = pd.concat(all_windows, ignore_index=True)
    print(f"Total windows loaded: {len(combined_windows)}")

    return combined_windows

def load_ppg_for_window_colab(row, target_length=1920, fs=64):
    """
    Load raw PPG data for a specific window (Colab version)
    Matches your hybrid approach but returns raw signal for CNN
    """
    try:
        # Extract window information
        participant_cell = row["Participant"]
        participant = participant_cell.split("_")[0]
        game = participant_cell.split("_")[1]

        interval_start = row["Interval_Start"]
        interval_end = row["Interval_End"]
        cohort = row["cohort"]

        # Map cohort names to folder names
        cohort_folder_map = {
            'dyslexia': 'Dyslexia',
            'ID': 'Intellectual Disabilities',
            'OBPI': 'Obstetric Brachial Plexus Injuries',
            'TD': 'Typically Developed'
        }

        cohort_folder = cohort_folder_map[cohort]

        # Look for PPG data in common locations
        possible_ppg_paths = [
            f"../../data/Aktives/PPG/{cohort_folder}/{participant}/{game}/BVP.csv",
        ]

        ppg_file_path = None
        for path in possible_ppg_paths:
            if os.path.exists(path):
                ppg_file_path = path
                break

        if ppg_file_path is None:
            return None, None

        # Load PPG data
        ppg_data = pd.read_csv(ppg_file_path)

        # Add time column
        ppg_data["Time"] = ppg_data.index / fs

        # Clean and convert values
        ppg_data['values'] = ppg_data['values'].astype(str).str.replace(',', '.', regex=False).astype(float)

        # Select the interval
        ppg_interval = ppg_data[(ppg_data['Time'] >= interval_start) &
                               (ppg_data['Time'] <= interval_end)]

        if len(ppg_interval) == 0:
            return None, None

        raw_ppg_values = ppg_interval['values'].values

        # Check for sufficient data
        if len(raw_ppg_values) < fs * 10:  # Less than 10 seconds
            return None, None

        # Minimal preprocessing for CNN (same as hybrid approach)
        clean_ppg_values = raw_ppg_values[~np.isnan(raw_ppg_values)]
        if len(clean_ppg_values) == 0:
            return None, None

        # Basic standardization only (no filtering for CNN)
        ppg_mean = np.mean(clean_ppg_values)
        ppg_std = np.std(clean_ppg_values)
        if ppg_std == 0:
            ppg_std = 1e-8
        ppg_standardized = (clean_ppg_values - ppg_mean) / ppg_std

        # Handle variable length windows - pad or truncate to target_length
        if len(ppg_standardized) > target_length:
            processed_signal = ppg_standardized[:target_length]
        elif len(ppg_standardized) < target_length:
            padding = target_length - len(ppg_standardized)
            processed_signal = np.pad(ppg_standardized, (0, padding), mode='constant', constant_values=0)
        else:
            processed_signal = ppg_standardized

        # Additional normalization for CNN (matching hybrid approach)
        final_signal = (processed_signal - np.mean(processed_signal)) / (np.std(processed_signal) + 1e-8)

        return final_signal, row['Label']

    except Exception as e:
        print(f"Error loading PPG for window: {str(e)}")
        return None, None

def load_aktives_data_colab():
    """
    Load all AKTIVES data for CNN training
    """
    print("Loading AKTIVES data for CNN training...")

    # Load analysis windows
    all_windows = load_all_aktives_windows_colab()

    all_signals = []
    all_labels = []
    all_participants = []
    all_cohorts = []

    print(f"Processing {len(all_windows)} windows...")

    for idx, row in tqdm(all_windows.iterrows(), total=len(all_windows), desc="Loading PPG data"):
        signal, label = load_ppg_for_window_colab(row)

        if signal is not None:
            participant = row["Participant"].split("_")[0]
            all_signals.append(signal)
            all_labels.append(int(label))
            all_participants.append(participant)
            all_cohorts.append(row['cohort'])

    print(f"Successfully loaded {len(all_signals)} windows")
    print(f"Label distribution: {np.bincount(all_labels)}")
    print(f"Participants: {len(np.unique(all_participants))}")
    print(f"Cohorts: {np.unique(all_cohorts)}")

    return np.array(all_signals), np.array(all_labels), np.array(all_participants), np.array(all_cohorts)

def train_dilated_cnn_aktives(model, train_loader, device, epochs=30, patience=29):
    """
    Training adapted for AKTIVES (smaller dataset, more patience)
    """
    # Calculate class weights
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())

    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    print(f"  Class distribution: {class_counts}")
    print(f"  Class weights: {class_weights}")

    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_loss = float('inf')
    patience_counter = 0

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

        model.train()

        avg_loss = epoch_loss / len(train_loader)
        epoch_acc = 100 * epoch_correct / epoch_total

        scheduler.step(avg_loss)

        # Print progress every 10 epochs
        # if (epoch + 1) % 10 == 0:
        #     print(f'  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={epoch_acc:.2f}%')

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_aktives_cnn.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  Early stopping at epoch {epoch+1}')
                break

        torch.cuda.empty_cache()

    # Load best model
    model.load_state_dict(torch.load('best_aktives_cnn.pth'))
    return model

# update this to train on all the data
def run_aktives_cnn_70_30(signals, labels, participants):
    """
    Run CNN with 70/30 split (matching your hybrid evaluation)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Total data: {len(signals)} windows from {len(np.unique(participants))} participants")

    # 70/30 split by participants (same as hybrid)
    unique_participants = np.unique(participants)
    train_participants, test_participants = train_test_split(
        unique_participants, test_size=0.3, random_state=42
    )

    print(f"Training participants ({len(train_participants)}): {sorted(train_participants)}")
    print(f"Test participants ({len(test_participants)}): {sorted(test_participants)}")

    # Create train/test splits
    train_mask = np.isin(participants, train_participants)
    test_mask = np.isin(participants, test_participants)

    X_train, X_test = signals[train_mask], signals[test_mask]
    y_train, y_test = labels[train_mask], labels[test_mask]

    print(f"\nData split:")
    print(f"Train: {len(X_train)} samples, distribution: {np.bincount(y_train)}")
    print(f"Test: {len(X_test)} samples, distribution: {np.bincount(y_test)}")

    # Create datasets and loaders
    train_dataset = PPGDataset(X_train, y_train)
    test_dataset = PPGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Initialize model (EXACT same as WESAD, just different input length)
    model = FixedDilatedCNN(input_length=1920).to(device)  # Only change: 1920 vs 7680

    print(f"\nModel architecture:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train
    print(f"\nTraining CNN...")
    start_time = time.time()
    model = train_dilated_cnn_aktives(model, train_loader, device, epochs=30, patience=29)
    train_time = time.time() - start_time

    # Test
    print(f"\nEvaluating on test set...")
    model.eval()
    test_preds = []
    test_probs = []
    test_true = []

    with torch.no_grad():
        for batch_windows, batch_labels in test_loader:
            batch_windows = batch_windows.to(device)
            outputs = model(batch_windows)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_true.extend(batch_labels.numpy())

    # Calculate metrics (matching hybrid format)
    test_preds = np.array(test_preds)
    test_probs = np.array(test_probs)
    test_true = np.array(test_true)

    accuracy = accuracy_score(test_true, test_preds) * 100
    f1 = f1_score(test_true, test_preds, average='weighted') * 100

    if len(np.unique(test_true)) > 1:
        auc = roc_auc_score(test_true, test_probs[:, 1]) * 100  # Use probabilities for AUC
    else:
        auc = 0.0

    results = {
        'model': 'Dilated CNN',
        'accuracy': accuracy,
        'auc_roc': auc,
        'f1_score': f1,
        'train_time': train_time,
        'total_params': total_params
    }

    return results

def save_aktives_cnn_results(results):
    """Save results in format compatible with hybrid comparison"""

    print(f"\n=== AKTIVES CNN Results (70/30 Split) ===")
    print(f"Model: {results['model']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"AUC-ROC:  {results['auc_roc']:.2f}%")
    print(f"F1-Score: {results['f1_score']:.2f}%")
    print(f"Training Time: {results['train_time']:.1f}s")
    print(f"Parameters: {results['total_params']:,}")

    # Save to CSV for comparison
    results_df = pd.DataFrame([results])
    results_df.to_csv('aktives_cnn_results.csv', index=False)
    print(f"\nResults saved to 'aktives_cnn_results.csv'")

    return results_df

def train_cnn_deployment_aktives(X_train, y_train, device, epochs=30):
    """
    Train CNN for deployment (using ALL data) - AKTIVES version
    Simplified for timing measurement
    """
    print(f"Training CNN for deployment...")
    print(f"Training data: {len(X_train)} windows")
    print(f"Label distribution: {np.bincount(y_train)}")

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train).unsqueeze(1)
    y_tensor = torch.LongTensor(y_train)

    # Simple train/val split for monitoring (optional)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor)

    # Create data loaders
    train_dataset = PPGDataset(X_tr.squeeze(1), y_tr)  # Remove extra dimension
    val_dataset = PPGDataset(X_val.squeeze(1), y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize model - AKTIVES uses 1920 length (30s at 64Hz)
    model = FixedDilatedCNN(input_length=1920).to(device)

    # Calculate class weights
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=5e-4)

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

def run_aktives_cnn_deployment_timing(signals, labels, participants):
    """
    Measure CNN training time for deployment scenario (ALL subjects)
    AKTIVES version matching WESAD approach
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("="*70)
    print("AKTIVES CNN DEPLOYMENT TRAINING TIME MEASUREMENT")
    print("="*70)

    # Use ALL subjects for training (deployment scenario)
    all_windows = signals
    all_labels = labels
    
    print(f"\nTraining data:")
    print(f"  Total subjects: {len(np.unique(participants))}")
    print(f"  Total windows: {len(all_windows)}")
    print(f"  Label distribution: {np.bincount(all_labels)}")

    # ===== TIMING MEASUREMENT =====
    print(f"\n--- Starting CNN Training Timing ---")
    training_start_time = time.perf_counter()
    
    # Train CNN on all data
    trained_model = train_cnn_deployment_aktives(all_windows, all_labels, device, epochs=30)
    
    training_end_time = time.perf_counter()
    cnn_training_time = training_end_time - training_start_time
    # ===== END TIMING =====

    print(f"--- CNN Training Timing Complete ---")

    # Calculate additional metrics
    training_time_per_sample_ms = (cnn_training_time * 1000) / len(all_windows)

    print(f"\n" + "="*70)
    print("AKTIVES CNN DEPLOYMENT TRAINING TIME RESULTS")
    print("="*70)
    
    print(f"Training Configuration:")
    print(f"  Training samples: {len(all_windows)}")
    print(f"  Training subjects: {len(np.unique(participants))}")
    print(f"  Epochs: 1")
    print(f"  Device: {device}")
    print(f"  Signal length: 1920 samples (30s at 64Hz)")
    
    print(f"\nTiming Results:")
    print(f"  Total training time: {cnn_training_time:.3f} seconds")
    print(f"  Training time per sample: {training_time_per_sample_ms:.3f} ms")


    # Save results
    results = {
        'approach': 'end_to_end_cnn_deployment_aktives',
        'training_time': cnn_training_time,
        'training_samples': len(all_windows),
        'training_subjects': len(np.unique(participants)),
        'signal_length': 1920,
        'device': str(device)
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('aktives_cnn_deployment_timing_results.csv', index=False)
    print(f"\nResults saved to 'aktives_cnn_deployment_timing_results.csv'")
    
    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return results, trained_model

def measure_aktives_cnn_inference_time(trained_model, signals, device, n_tests=10):
    """
    Measure CNN inference time using a sample from the dataset
    AKTIVES version
    """
    print(f"\n=== AKTIVES CNN Inference Time Measurement ===")
    print(f"Number of tests: {n_tests}")
    
    # Use first signal as test sample
    if len(signals) == 0:
        print("No signals available for inference testing!")
        return np.nan, np.nan
    
    # Prepare model for inference
    trained_model.eval()
    
    inference_times = []
    successful_predictions = 0
    
    # Run inference timing tests
    for i in range(n_tests):
        try:
            
            test_window = signals[i-1]  # Use first window
            print(f"Test window: {len(test_window)} samples")
            print(f"i value: {i-1}")

            # Prepare input tensor (same preprocessing as training)
            input_tensor = torch.FloatTensor(test_window).unsqueeze(0).unsqueeze(0).to(device)
            
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

            print(f"Input tensor shape: {input_tensor.shape}")
            print(f"Output tensor shape: {outputs.shape}")
            print(f"Predicted class: {prediction.item()}")
            print(f"Prediction confidence: {torch.max(probabilities).item():.3f}")
            
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

def main_aktives_deployment_timing():
    """Main function for AKTIVES CNN deployment timing measurement"""
    print("=== End-to-End AKTIVES CNN Deployment Training Time ===")
    print("Training CNN on ALL subjects for fair comparison with TD/Hybrid approaches")

    try:
        # Load data using existing function
        signals, labels, participants, cohorts = load_aktives_data_colab()

        if len(signals) == 0:
            print("No data found! Check your data directory.")
            return None

        print(f"Loaded {len(signals)} windows from {len(np.unique(participants))} subjects")

        # Run deployment timing measurement
        timing_results, trained_model = run_aktives_cnn_deployment_timing(signals, labels, participants)

        if timing_results:
            print(f"\n=== AKTIVES CNN Deployment Timing Complete ===")
            
            # Measure inference time using dataset sample
            avg_inf_time, std_inf_time = measure_aktives_cnn_inference_time(
                trained_model, signals, timing_results['device'], n_tests=10
            )
            
            # Add inference timing to results
            timing_results['avg_inference_time_ms'] = avg_inf_time
            timing_results['std_inference_time_ms'] = std_inf_time
            
            print(f"\nInference Time: {avg_inf_time:.3f} ± {std_inf_time:.3f} ms")
            
            # Save updated results
            results_df = pd.DataFrame([timing_results])
            results_df.to_csv('aktives_cnn_deployment_timing_results.csv', index=False)
            print(f"Complete results saved to 'aktives_cnn_deployment_timing_results.csv'")
        
        return timing_results
        
    except Exception as e:
        print(f"Error in AKTIVES deployment timing: {e}")
        import traceback
        traceback.print_exc()
        return None

# Replace the existing main() function with this:
def main():
    """Updated main function for deployment timing"""
    print("=== AKTIVES End-to-End CNN Deployment Timing ===")
    print("Measuring training and inference time for deployment scenario")
    
    # Run deployment timing instead of 70/30 evaluation
    results = main_aktives_deployment_timing()
    
    if results:
        print(f"\n=== AKTIVES CNN Deployment Complete ===")
        print(f"Inference time: {results.get('avg_inference_time_ms', 'N/A'):.3f}ms")
    else:
        print("Deployment timing failed!")
    
    return results

# Run the main function
if __name__ == "__main__":
    results = main()
# %%
