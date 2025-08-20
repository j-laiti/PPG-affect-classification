# CNN classification of the WESAD dataset based on PPG signals labels stress and non-stress

# imports

# select the raw data from ../data/WESAD_BVP_extracted/ based on data labeled stress (1.0) and non-stress (0.0)

# break into window sizes of 120 for the raw data

# assemble model architecture to match the Motaman paper

# Dilated CNN Architecture (Motaman et al., 2025)

# run the training process

# evaluate the model similar to the previous evaluation of WESAD with they hybrid and TD features only

# output the results

"""
CNN classification of the WESAD dataset based on PPG signals labels stress and non-stress
Based on Motaman et al. (2025) Dilated CNN architecture
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
import pickle
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

# Add path for preprocessing functions if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class DilatedCNN(nn.Module):
    """
    Dilated CNN Architecture based on Motaman et al. (2025)
    Adapted for 120-second windows (7680 samples at 64 Hz)
    """
    def __init__(self, input_length=7680):  # 120s at 64Hz
        super(DilatedCNN, self).__init__()
        
        # 8 Dilated Convolutional Layers with progressive dilation rates
        # Adjusted filter progression for longer sequences
        
        # Layer 1: Dilation rate = 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=8, dilation=1, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Layer 2: Dilation rate = 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=8, dilation=2, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Layer 3: Dilation rate = 4
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, dilation=4, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Layer 4: Dilation rate = 8
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=8, dilation=8, padding='same'),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Layer 5: Dilation rate = 16
        self.conv5 = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=8, dilation=16, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Layer 6: Dilation rate = 32
        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, dilation=32, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Layer 7: Dilation rate = 64
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 320, kernel_size=8, dilation=64, padding='same'),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Layer 8: Dilation rate = 128
        self.conv8 = nn.Sequential(
            nn.Conv1d(320, 512, kernel_size=8, dilation=128, padding='same'),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        
        x = self.global_pool(x)  # Global max pooling
        x = self.classifier(x)
        
        return x

class PPGDataset(Dataset):
    """Dataset class for PPG windows"""
    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx].unsqueeze(0), self.labels[idx]  # Add channel dimension

def load_wesad_raw_data(data_dir="../data/WESAD_BVP_extracted/"):
    """
    Load raw WESAD data labeled as stress (1.0) and non-stress (0.0) from CSV files
    """
    print("Loading WESAD raw data from CSV files...")
    data_dir = Path(data_dir)
    
    all_subjects = []
    all_windows = []
    all_labels = []
    
    # Load data for each subject
    for subject_file in sorted(data_dir.glob("S*.csv")):
        subject_id = subject_file.stem  # e.g., 'S02'
        print(f"Processing {subject_id}...")
        
        try:
            # Load CSV file
            subject_data = pd.read_csv(subject_file)
            
            # Extract BVP signal and labels
            if 'BVP' in subject_data.columns and 'label' in subject_data.columns:
                bvp_signal = subject_data['BVP'].values
                labels = subject_data['label'].values
                
                print(f"  {subject_id}: Total {len(bvp_signal)} samples")
                
                # Extract labeled segments and create windows
                subject_windows, subject_labels, subject_ids = extract_labeled_segments_and_windows(
                    bvp_signal, labels, subject_id
                )
                
                if len(subject_windows) > 0:
                    all_windows.extend(subject_windows)
                    all_labels.extend(subject_labels)
                    all_subjects.extend(subject_ids)
                    
                    print(f"  {subject_id}: Created {len(subject_windows)} windows")
                else:
                    print(f"  {subject_id}: No valid windows created")
                    
        except Exception as e:
            print(f"  Error loading {subject_id}: {str(e)}")
            continue
    
    print(f"\nTotal windows created: {len(all_windows)}")
    if len(all_labels) > 0:
        print(f"Stress windows: {np.sum(np.array(all_labels) == 1)}")
        print(f"Non-stress windows: {np.sum(np.array(all_labels) == 0)}")
    
    return np.array(all_windows), np.array(all_labels), np.array(all_subjects)

def extract_labeled_segments_and_windows(bvp_signal, labels, subject_id, fs=64):
    """
    Extract continuous labeled segments and create overlapping windows
    
    Parameters:
    - bvp_signal: PPG signal array
    - labels: Label array (same length as bvp_signal)
    - subject_id: Subject identifier
    - fs: Sampling frequency (64 Hz)
    
    Returns:
    - windows: List of signal windows
    - window_labels: List of corresponding labels
    - subject_ids: List of subject IDs for each window
    """
    window_size = int(120 * fs)  # 120 seconds = 7680 samples at 64 Hz
    step_size = int(30 * fs)     # 30 seconds = 1920 samples (overlap = 90 seconds)
    
    windows = []
    window_labels = []
    subject_ids = []
    
    # Find continuous labeled segments
    labeled_segments = find_labeled_segments(labels)
    
    print(f"    Found {len(labeled_segments)} labeled segments")
    
    for segment_start, segment_end, segment_label in labeled_segments:
        # Only process stress (1.0) and non-stress (0.0) segments
        if segment_label not in [0.0, 1.0]:
            continue
            
        segment_length = segment_end - segment_start
        segment_duration = segment_length / fs
        
        print(f"    Processing {segment_label} segment: {segment_length} samples ({segment_duration:.1f}s)")
        
        # Extract signal for this segment
        segment_signal = bvp_signal[segment_start:segment_end]
        
        # Create overlapping windows within this segment
        segment_windows = 0
        for start_idx in range(0, len(segment_signal) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window = segment_signal[start_idx:end_idx]
            
            # Verify window has correct length
            if len(window) == window_size:
                windows.append(window)
                window_labels.append(int(segment_label))
                subject_ids.append(subject_id)
                segment_windows += 1
        
        print(f"      Created {segment_windows} windows from this segment")
    
    return windows, window_labels, subject_ids

def find_labeled_segments(labels):
    """
    Find continuous segments with the same label (ignoring NaN/empty labels)
    
    Returns list of (start_idx, end_idx, label) tuples
    """
    segments = []
    
    # Convert to pandas Series for easier handling of NaN
    label_series = pd.Series(labels)
    
    # Find where labels are not null
    valid_mask = label_series.notna()
    
    if not valid_mask.any():
        return segments
    
    # Get valid indices and labels
    valid_indices = np.where(valid_mask)[0]
    valid_labels = label_series[valid_mask].values
    
    if len(valid_indices) == 0:
        return segments
    
    # Find segments of continuous labels
    current_label = valid_labels[0]
    segment_start = valid_indices[0]
    
    for i in range(1, len(valid_indices)):
        idx = valid_indices[i]
        label = valid_labels[i]
        
        # Check if label changed or if there's a gap in indices
        if label != current_label or (idx - valid_indices[i-1]) > 1:
            # End current segment
            segment_end = valid_indices[i-1] + 1
            segments.append((segment_start, segment_end, current_label))
            
            # Start new segment
            current_label = label
            segment_start = idx
    
    # Add final segment
    segment_end = valid_indices[-1] + 1
    segments.append((segment_start, segment_end, current_label))
    
    return segments

# This function is no longer needed as it's replaced by extract_labeled_segments_and_windows

def preprocess_signal(signal, normalize=True):
    """
    Minimal preprocessing for raw PPG signal
    """
    if normalize:
        # Z-score normalization per signal
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    
    return signal

def train_model(model, train_loader, val_loader, device, epochs=50, patience=10):
    """
    Train the CNN model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("Starting training...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_windows, batch_labels in train_loader:
            batch_windows = batch_windows.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_windows)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_windows, batch_labels in val_loader:
                batch_windows = batch_windows.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_windows)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}] - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model, train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model and return predictions
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch_windows, batch_labels in test_loader:
            batch_windows = batch_windows.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_windows)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate classification metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # AUC-ROC (using probability of positive class)
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc
    }

def run_logo_evaluation(windows, labels, subjects):
    """
    Run Leave-One-Group-Out cross-validation
    """
    print("\nStarting LOGO cross-validation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Preprocess all windows
    print("Preprocessing signals...")
    processed_windows = []
    for i, window in enumerate(windows):
        processed_window = preprocess_signal(window, normalize=True)
        processed_windows.append(processed_window)
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(windows)} windows")
    
    processed_windows = np.array(processed_windows)
    
    # Get unique subjects
    unique_subjects = np.unique(subjects)
    print(f"Found {len(unique_subjects)} subjects: {unique_subjects}")
    
    # Store results for each fold
    fold_results = []
    all_predictions = []
    all_true_labels = []
    all_probabilities = []
    
    logo = LeaveOneGroupOut()
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(processed_windows, labels, subjects)):
        test_subject = unique_subjects[fold]
        print(f"\nFold {fold + 1}/{len(unique_subjects)} - Testing on {test_subject}")
        
        # Split data
        X_train, X_test = processed_windows[train_idx], processed_windows[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        print(f"  Train distribution: {np.bincount(y_train)}")
        print(f"  Test distribution: {np.bincount(y_test)}")
        
        # Further split training data for validation (80/20)
        train_size = int(0.8 * len(X_train))
        indices = np.random.permutation(len(X_train))
        
        X_train_split = X_train[indices[:train_size]]
        X_val_split = X_train[indices[train_size:]]
        y_train_split = y_train[indices[:train_size]]
        y_val_split = y_train[indices[train_size:]]
        
        # Create datasets and loaders
        train_dataset = PPGDataset(X_train_split, y_train_split)
        val_dataset = PPGDataset(X_val_split, y_val_split)
        test_dataset = PPGDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        model = DilatedCNN(input_length=7680).to(device)
        
        # Train model
        start_time = time.time()
        model, train_losses, val_losses, val_accuracies = train_model(
            model, train_loader, val_loader, device, epochs=50, patience=10
        )
        training_time = time.time() - start_time
        
        # Evaluate model
        start_time = time.time()
        fold_predictions, fold_probabilities, fold_true = evaluate_model(model, test_loader, device)
        inference_time = time.time() - start_time
        
        # Calculate metrics for this fold
        fold_metrics = calculate_metrics(fold_true, fold_predictions, fold_probabilities)
        fold_metrics['training_time'] = training_time
        fold_metrics['inference_time'] = inference_time
        fold_metrics['test_subject'] = test_subject
        
        fold_results.append(fold_metrics)
        
        # Store predictions for overall evaluation
        all_predictions.extend(fold_predictions)
        all_true_labels.extend(fold_true)
        all_probabilities.extend(fold_probabilities)
        
        print(f"  Results: Acc={fold_metrics['accuracy']:.4f}, "
              f"AUC={fold_metrics['auc_roc']:.4f}, "
              f"F1={fold_metrics['f1_score']:.4f}")
        print(f"  Training time: {training_time:.2f}s, Inference time: {inference_time:.4f}s")
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(
        np.array(all_true_labels), 
        np.array(all_predictions), 
        np.array(all_probabilities)
    )
    
    return fold_results, overall_metrics

def save_results(fold_results, overall_metrics, output_dir="results/"):
    """
    Save results to files
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Convert fold results to DataFrame
    fold_df = pd.DataFrame(fold_results)
    
    # Save detailed results
    fold_df.to_csv(f"{output_dir}/cnn_logo_fold_results.csv", index=False)
    
    # Save summary statistics
    summary_stats = {
        'metric': ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'],
        'mean': [fold_df[metric].mean() for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']],
        'std': [fold_df[metric].std() for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']],
        'overall': [overall_metrics[metric] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(f"{output_dir}/cnn_logo_summary.csv", index=False)
    
    print(f"\nResults saved to {output_dir}")
    print("\nSummary Statistics:")
    print(summary_df.round(4))

def main():
    """
    Main execution function
    """
    print("=== WESAD CNN Classification Pipeline ===")
    print("Based on Motaman et al. (2025) Dilated CNN architecture\n")
    
    # Load raw data
    windows, labels, subjects = load_wesad_raw_data()
    
    if len(windows) == 0:
        print("No data loaded. Please check the data directory.")
        return
    
    print(f"\nDataset Summary:")
    print(f"Total windows: {len(windows)}")
    print(f"Window shape: {windows[0].shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    print(f"Subjects: {len(np.unique(subjects))}")
    
    # Run LOGO evaluation
    fold_results, overall_metrics = run_logo_evaluation(windows, labels, subjects)
    
    # Save results
    save_results(fold_results, overall_metrics)
    
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()