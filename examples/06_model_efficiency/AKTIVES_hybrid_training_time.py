# Code to extract DL and time-domain features from the AKTIVES dataset
# Adapted from WESAD hybrid approach for AKTIVES structure WITH COMPREHENSIVE TIMING

#%% imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add path for preprocessing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from preprocessing.feature_extraction import get_ppg_features
from preprocessing.filters import *
from preprocessing.peak_detection import threshold_peakdetection

class TrainableCNNFeatureExtractor(nn.Module):
    """
    CNN + classification head for training (same as WESAD)
    """
    def __init__(self, input_length=1920, num_classes=2):  # 30s at 64Hz = 1920 samples
        super().__init__()
        
        # 1D CNN architecture
        self.feature_extractor = Simple1DCNNFeatures(input_length)
        
        # Add classification head for training
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        outputs = self.classifier(features)
        
        if return_features:
            return outputs, features
        return outputs
    
    def extract_features_only(self, x):
        """Extract only features (for hybrid approach)"""
        return self.feature_extractor(x)

class Simple1DCNNFeatures(nn.Module):
    """
    Simple CNN for feature extraction from PPG signals (same as WESAD)
    """
    def __init__(self, input_length=1920, output_features=32):  # 30s at 64Hz
        super(Simple1DCNNFeatures, self).__init__()
        
        # Layer 1: Initial feature detection
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        # Layer 2: Pattern refinement
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=11, stride=2, padding=5),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        
        # Layer 3: High-level features
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, output_features//4, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)  # 32 features output
        )
        
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return x

def load_all_aktives_windows():
    """
    Load all analysis windows from all cohorts
    
    Returns:
        combined_windows: DataFrame with all windows and cohort info
    """
    print("Loading all AKTIVES analysis windows...")
    
    cohorts = ['dyslexia', 'ID', 'OBPI', 'TD']
    all_windows = []
    
    for cohort in cohorts:
        window_file = f"../../data/Aktives/analysis_windows/analysis_windows_{cohort}.csv"
        
        if os.path.exists(window_file):
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

def load_ppg_for_window(row, target_length=1920, fs=64):
    """
    Load raw PPG data for a specific window (for CNN input)
    
    Args:
        row: Window row from analysis_windows CSV
        target_length: Target length in samples (30s * 64Hz = 1920)
        fs: Sampling frequency
    
    Returns:
        raw_ppg_signal: Raw PPG signal (only NaN removal + standardization) or None if failed
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
        
        # Load PPG data
        ppg_file_path = f"../../data/Aktives/PPG/{cohort_folder}/{participant}/{game}/BVP.csv"
        
        if not os.path.exists(ppg_file_path):
            return None
        
        ppg_data = pd.read_csv(ppg_file_path)
        
        # Add time column
        ppg_data["Time"] = ppg_data.index / fs
        
        # Clean and convert values
        ppg_data['values'] = ppg_data['values'].astype(str).str.replace(',', '.', regex=False).astype(float)
        
        # Select the interval
        ppg_interval = ppg_data[(ppg_data['Time'] >= interval_start) & 
                               (ppg_data['Time'] <= interval_end)]
        
        if len(ppg_interval) == 0:
            return None
        
        raw_ppg_values = ppg_interval['values'].values
        
        # Check for sufficient data
        if len(raw_ppg_values) < fs * 10:  # Less than 10 seconds
            return None
        
        # Minimal preprocessing for CNN (same as WESAD approach)
        clean_ppg_values = raw_ppg_values[~np.isnan(raw_ppg_values)]
        if len(clean_ppg_values) == 0:
            return None
            
        # Basic standardization only (no filtering for CNN)
        ppg_standardized = standardize(clean_ppg_values)
        
        # Handle variable length windows - pad or truncate to target_length
        if len(ppg_standardized) > target_length:
            # Truncate to target length
            processed_signal = ppg_standardized[:target_length]
        elif len(ppg_standardized) < target_length:
            # Pad with zeros
            padding = target_length - len(ppg_standardized)
            processed_signal = np.pad(ppg_standardized, (0, padding), mode='constant', constant_values=0)
        else:
            processed_signal = ppg_standardized
        
        return processed_signal
        
    except Exception as e:
        print(f"Error loading PPG for window: {str(e)}")
        return None

def prepare_aktives_training_data(windows_df, test_participants, target_length=1920):
    """
    Prepare training data for CNN from AKTIVES windows WITH TIMING
    
    Args:
        windows_df: All analysis windows
        test_participants: List of participants to exclude from training
        target_length: Target signal length in samples
    
    Returns:
        X_train: Training signals
        y_train: Training labels
    """
    print("Preparing training data for CNN...")
    
    # ===== TIMING: Data preparation start =====
    prep_start_time = time.perf_counter()
    
    # Filter out test participants
    train_windows = windows_df[~windows_df['Participant'].str.split('_').str[0].isin(test_participants)]
    
    print(f"Training windows: {len(train_windows)}")
    print(f"Test participants excluded: {test_participants}")
    
    X_train = []
    y_train = []
    
    for idx, row in tqdm(train_windows.iterrows(), total=len(train_windows), desc="Loading training data"):
        ppg_signal = load_ppg_for_window(row, target_length)
        
        if ppg_signal is not None:
            X_train.append(ppg_signal)
            y_train.append(int(row['Label']))  # Assuming Label is 0 or 1
    
    prep_time = time.perf_counter() - prep_start_time
    print(f"Data preparation completed in {prep_time:.2f} seconds")
    print(f"Successfully loaded {len(X_train)} training windows")
    print(f"Label distribution: {np.bincount(y_train)}")
    
    return np.array(X_train), np.array(y_train)

def train_aktives_cnn(X_train, y_train, device, epochs=30, validation_split=0.2):
    """
    Train CNN for AKTIVES stress detection WITH TIMING
    """
    print("Training CNN for AKTIVES stress detection...")
    print(f"Training data: {len(X_train)} windows")
    print(f"Label distribution: {np.bincount(y_train)}")
    
    # ===== TIMING: CNN training start =====
    training_start_time = time.perf_counter()
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
    y_tensor = torch.LongTensor(y_train)
    
    # Train/validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tensor, y_tensor, test_size=validation_split, random_state=42, stratify=y_tensor
    )
    
    # Create data loaders
    from torch.utils.data import DataLoader, TensorDataset
    train_dataset = TensorDataset(X_tr, y_tr)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Smaller batch for AKTIVES
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = TrainableCNNFeatureExtractor(input_length=1920).to(device)  # 30s at 64Hz
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 29  # Run all 30 epochs for training test
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate accuracies
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'   Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'temp_best_aktives_cnn.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'   Early stopping at epoch {epoch+1}')
                break
        
        torch.cuda.empty_cache()
    
    # Load best model
    model.load_state_dict(torch.load('temp_best_aktives_cnn.pth'))
    
    training_time = time.perf_counter() - training_start_time
    print(f'CNN training completed in {training_time:.2f} seconds. Best validation accuracy: {best_val_acc:.1f}%')
    
    return model, training_time

def extract_aktives_features_for_window(row, trained_model, device, target_length=1920, fs=64):
    """
    Extract both CNN and TD features for a single window
    
    Args:
        row: Window row from analysis_windows
        trained_model: Trained CNN model
        device: PyTorch device
        target_length: Target signal length
        fs: Sampling frequency
    
    Returns:
        features_dict: Dictionary with CNN features, TD features, and metadata
    """
    try:
        # Load raw PPG signal for CNN
        raw_ppg_signal = load_ppg_for_window(row, target_length, fs)
        
        if raw_ppg_signal is None:
            return None
        
        participant_cell = row["Participant"]
        participant = participant_cell.split("_")[0]
        game = participant_cell.split("_")[1]
        interval_start = row["Interval_Start"]
        interval_end = row["Interval_End"]
        cohort = row["cohort"]
        label = int(row['Label'])

        # measure feature extraction starting here
        start_time_feat_extraction = time.perf_counter()
        
        # ===== CNN FEATURE EXTRACTION =====
        # Normalize for CNN (following WESAD approach)
        cnn_signal = (raw_ppg_signal - np.mean(raw_ppg_signal)) / (np.std(raw_ppg_signal) + 1e-8)
        
        with torch.no_grad():
            signal_tensor = torch.FloatTensor(cnn_signal).unsqueeze(0).unsqueeze(0)  # (1, 1, length)
            signal_tensor = signal_tensor.to(device)
            cnn_features = trained_model.extract_features_only(signal_tensor)
            cnn_features_np = cnn_features.cpu().numpy().flatten()
        
        # ===== TIME-DOMAIN FEATURE EXTRACTION =====
        
        # Process for TD features (following AKTIVES processing + WESAD filtering)
        clean_ppg_values = raw_ppg_signal[~np.isnan(raw_ppg_signal)]
        ppg_standardized = standardize(clean_ppg_values)
        
        # Apply filtering for TD features (following WESAD approach)
        bp_bvp = bandpass_filter(ppg_standardized, 0.5, 10, fs, order=2)  # Using WESAD frequencies
        smoothed_signal = moving_average_filter(bp_bvp, window_size=5)
        
        segment_stds, std_ths = simple_dynamic_threshold(smoothed_signal, 64, 95, window_size= 3)
        sim_clean_signal, clean_signal_indices = simple_noise_elimination(smoothed_signal, 64, std_ths)
        sim_final_clean_signal = moving_average_filter(sim_clean_signal, window_size=3)
        
        # Extract TD features
        td_stats = get_ppg_features(ppg_seg=sim_final_clean_signal.tolist(), 
                                  fs=fs, 
                                  label=label, 
                                  calc_sq=True)
        
        if td_stats is None or len(cnn_features_np) != 32:
            return None
        
        # Combine features
        features_dict = {
            'participant': participant,
            'game': game,
            'cohort': cohort,
            'label': label,
            'interval_start': interval_start,
            'interval_end': interval_end
        }
        
        # Add CNN features
        for i, feat in enumerate(cnn_features_np):
            features_dict[f'cnn_feature_{i:04d}'] = feat
        
        # Add TD features
        if isinstance(td_stats, dict):
            features_dict.update(td_stats)
        else:
            # If td_stats is a list, use standard names
            td_names = ['mean_hr', 'std_hr', 'rmssd', 'sdnn', 'sdsd', 
                       'mean_nn', 'mean_sd', 'median_nn', 'pnn20', 'pnn50']
            for i, feat in enumerate(td_stats[:len(td_names)]):
                features_dict[td_names[i]] = feat

        end_time_feat_extraction = time.perf_counter()
        feat_extraction_duration = end_time_feat_extraction - start_time_feat_extraction

        return features_dict, feat_extraction_duration

    except Exception as e:
        print(f"Error extracting features for window: {str(e)}")
        return None

def extract_all_aktives_hybrid_features_with_timing():
    """
    Main function to extract hybrid features from all AKTIVES data WITH COMPREHENSIVE TIMING
    """
    print("="*70)
    print("AKTIVES HYBRID FEATURE EXTRACTION WITH COMPREHENSIVE TIMING")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ===== TIMING STARTS HERE =====
    total_start_time = time.perf_counter()
    
    # Step 1: Load all analysis windows
    print("\n1. Loading all analysis windows...")
    data_loading_start = time.perf_counter()
    
    try:
        all_windows = load_all_aktives_windows()
        data_loading_time = time.perf_counter() - data_loading_start
        print(f"Data loading completed in {data_loading_time:.2f} seconds")
    except Exception as e:
        print(f"Error loading analysis windows: {e}")
        return None
    
    # Step 2: Create 70/30 participant split
    print(f"\n2. Creating participant split...")
    split_start = time.perf_counter()
    
    all_participants = all_windows['Participant'].str.split('_').str[0].unique()
    print(f"Total unique participants: {len(all_participants)}")
    
    # Random split ensuring no participant overlap
    train_participants, test_participants = train_test_split(
        all_participants, test_size=0.3, random_state=42
    )
    
    split_time = time.perf_counter() - split_start
    print(f"Participant split completed in {split_time:.3f} seconds")
    print(f"Training participants ({len(train_participants)}): {sorted(train_participants)}")
    print(f"Test participants ({len(test_participants)}): {sorted(test_participants)}")
    
    # Step 3: Prepare training data
    print(f"\n3. Preparing training data...")
    X_train, y_train = prepare_aktives_training_data(all_windows, test_participants)
    
    if len(X_train) < 50:
        print(f"Error: Insufficient training data ({len(X_train)} windows)")
        return None
    
    # Step 4: Train CNN
    print(f"\n4. Training CNN...")
    trained_model, cnn_training_time = train_aktives_cnn(X_train, y_train, device)
    
    # Step 5: Extract features for all windows
    print(f"\n5. Extracting hybrid features for all windows...")
    
    trained_model.eval()
    
    all_features = []
    failed_windows = 0
    feature_extraction_durations = []

    for idx, row in tqdm(all_windows.iterrows(), total=len(all_windows), desc="Extracting features"):
        features, feature_extraction_duration = extract_aktives_features_for_window(row, trained_model, device)
        feature_extraction_durations.append(feature_extraction_duration)
        if features is not None:
            all_features.append(features)
        else:
            failed_windows += 1

    feature_extraction_time = sum(feature_extraction_durations)
    print(f"Feature extraction completed in {feature_extraction_time:.2f} seconds")
    print(f"Successfully extracted features from {len(all_features)}/{len(all_windows)} windows")
    print(f"Failed windows: {failed_windows}")
    
    if len(all_features) == 0:
        print("Error: No features extracted!")
        return None
    
    # Step 6: Create DataFrame and standardize features
    print(f"\n6. Processing and standardizing features...")
    processing_start = time.perf_counter()
    
    features_df = pd.DataFrame(all_features)
    
    # Identify feature columns
    cnn_columns = [col for col in features_df.columns if col.startswith('cnn_feature_')]
    td_columns = [col for col in features_df.columns if col not in cnn_columns and 
                  col not in ['participant', 'game', 'cohort', 'label', 'interval_start', 'interval_end']]
    
    print(f"CNN features: {len(cnn_columns)}")
    print(f"TD features: {len(td_columns)}")
    
    # Standardize features separately (following WESAD approach)
    scaler_cnn = StandardScaler()
    scaler_td = StandardScaler()
    
    features_df[cnn_columns] = scaler_cnn.fit_transform(features_df[cnn_columns])
    features_df[td_columns] = scaler_td.fit_transform(features_df[td_columns])
    
    processing_time = time.perf_counter() - processing_start
    print(f"Feature processing completed in {processing_time:.2f} seconds")
    
    # Step 7: Save results
    print(f"\n7. Saving results...")
    saving_start = time.perf_counter()
    
    output_dir = "../../data/AKTIVES_hybrid_features/"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "all_subjects_AKTIVES_hybrid_features_timing.csv")
    features_df.to_csv(output_file, index=False)
    
    saving_time = time.perf_counter() - saving_start
    print(f"Results saved in {saving_time:.2f} seconds")
    print(f"Output file: {output_file}")
    
    # ===== TIMING ENDS HERE =====
    total_time = time.perf_counter() - total_start_time
    
    # Final Summary (following WESAD format)
    print("\n" + "="*70)
    print("TIMING SUMMARY")
    print("="*70)
    print(f"Data Loading Time:        {data_loading_time:.3f} seconds")
    print(f"Participant Split Time:   {split_time:.3f} seconds")
    print(f"CNN Training Time:        {cnn_training_time:.3f} seconds")
    print(f"Feature Extraction Time:  {feature_extraction_time:.3f} seconds")
    print(f"Feature Processing Time:  {processing_time:.3f} seconds")
    print(f"Data Saving Time:         {saving_time:.3f} seconds")
    print(f"TOTAL TIME:              {total_time:.3f} seconds")
    
    print("\nExtraction Summary:")
    success_rate = len(all_features)/len(all_windows)*100
    print(f"Total windows processed: {len(all_windows)}")
    print(f"Successfully extracted: {len(all_features)}")
    print(f"Failed extractions: {failed_windows}")
    print(f"Success rate: {success_rate:.1f}%")
    
    print(f"\nDataset characteristics:")
    print(f"  Total features: {len(cnn_columns) + len(td_columns)}")
    print(f"  CNN features: {len(cnn_columns)}")
    print(f"  TD features: {len(td_columns)}")
    
    print(f"\nParticipant distribution:")
    participant_counts = features_df['participant'].value_counts()
    print(f"  Unique participants: {len(participant_counts)}")
    print(f"  Windows per participant - Mean: {participant_counts.mean():.1f}, Std: {participant_counts.std():.1f}")
    
    print(f"\nCohort distribution:")
    cohort_counts = features_df['cohort'].value_counts()
    for cohort, count in cohort_counts.items():
        print(f"  {cohort}: {count} windows")
    
    print(f"\nLabel distribution:")
    label_counts = features_df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = count / len(features_df) * 100
        print(f"  Label {label}: {count} windows ({percentage:.1f}%)")
    
    # Clean up temporary files
    
    # Return timing information for comparison (following WESAD format)
    return {
        'total_time': total_time,
        'data_loading_time': data_loading_time,
        'participant_split_time': split_time,
        'cnn_training_time': cnn_training_time,
        'feature_extraction_time': feature_extraction_time,
        'processing_time': processing_time,
        'saving_time': saving_time,
        'successful_windows': len(all_features),
        'total_windows': len(all_windows),
        'success_rate': success_rate,
        'final_dataset_shape': features_df.shape
    }

# Run the extraction with comprehensive timing
if __name__ == "__main__":
    print("Starting AKTIVES hybrid feature extraction with comprehensive timing...")
    results = extract_all_aktives_hybrid_features_with_timing()
    
    if results is not None:
        print("✓ Extraction completed successfully!")
        print(f"Total processing time: {results['total_time']:.2f} seconds")
        print(f"Success rate: {results['success_rate']:.1f}%")
    else:
        print("✗ Extraction failed!")
# %% save results to csv

results_df = pd.DataFrame([results])
results_df.to_csv("AKTIVES_hybrid_efficiency.csv", index=False)
# %%
