"""
Wellby Dataset Hybrid Processing Pipeline

Processes the Wellby dataset for stress classification using PPG signals.
Implements a hybrid feature extraction approach combining CNN-based features
and traditional time-domain HRV features.
    
Author: Justin Laiti
Note: Code organization and documentation assisted by Claude Sonnet 4.5
Last updated: Nov 15, 2025
"""
#%% imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add path for preprocessing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.feature_extraction import get_ppg_features
from preprocessing.filters import bandpass_filter, moving_average_filter, standardize
from preprocessing.peak_detection import threshold_peakdetection

class TrainableCNNFeatureExtractor(nn.Module):
    """
    CNN + classification head for training
    """
    def __init__(self, input_length=3000, num_classes=2):  # 1 min at 50Hz = 3000 samples (variable)
        super().__init__()
        
        # 1D CNN architecture
        self.feature_extractor = Simple1DCNNFeatures(input_length)
        
        # Add classification head for training
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),  # Higher dropout for tiny dataset
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
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
    Simple CNN for feature extraction from PPG signals
    """
    def __init__(self, input_length=3000, output_features=32):  # Adapted for ~1 min at 50Hz
        super(Simple1DCNNFeatures, self).__init__()
        
        # Layer 1: Initial feature detection (adapted for 50Hz vs 64Hz)
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
            nn.AdaptiveAvgPool1d(4)  # 32 features output (8*4=32)
        )
        
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return x

def load_wellby_data():
    """
    Load and align Wellby PPG data with session information
    
    Returns:
        aligned_data: DataFrame with PPG signals and session info
    """
    print("Loading Wellby data...")
    
    # Load PPG signals (transposed format - sessions as columns)
    ppg_data = pd.read_csv("../../data/Wellby/selected_PPG_data.csv")
    print(f"PPG data shape: {ppg_data.shape}")
    
    # Load session information
    session_info = pd.read_csv("../../data/Wellby/Wellby_all_subjects_features.csv")
    print(f"Session info shape: {session_info.shape}")
    print(f"Available session IDs: {len(session_info['Session_ID'].unique())}")
    
    # Get session IDs from both datasets
    ppg_session_ids = ppg_data.columns.tolist()
    info_session_ids = session_info['Session_ID'].tolist()
    
    print(f"PPG sessions: {len(ppg_session_ids)}")
    print(f"Info sessions: {len(info_session_ids)}")
    
    # Find common session IDs
    common_sessions = list(set(ppg_session_ids) & set(info_session_ids))
    print(f"Common sessions: {len(common_sessions)}")
    
    if len(common_sessions) == 0:
        print("Warning: No matching session IDs found!")
        print(f"Sample PPG session IDs: {ppg_session_ids[:5]}")
        print(f"Sample info session IDs: {info_session_ids[:5]}")
        return None
    
    aligned_data = []
    
    for session_id in common_sessions:
        # Get PPG signal for this session
        ppg_signal = ppg_data[session_id].dropna().values
        
        # Get session info
        session_row = session_info[session_info['Session_ID'] == session_id].iloc[0]
        
        # Store aligned data
        aligned_data.append({
            'session_id': session_id,
            'participant': session_row['Participant'],
            'ppg_signal': ppg_signal,
            'stress_label': session_row['stress_binary'],
            'sleep_label': session_row['sleep_binary'],
            'school': session_row['School'],
            'age': session_row['Age'],
            'gender': session_row['Gender'],
            'PSS': session_row['PSS'],
            'PSQI': session_row['PSQI'],
            'EPOCH': session_row['EPOCH'],
            'SQI': session_row['SQI']
        })
    
    print(f"Successfully aligned {len(aligned_data)} sessions")
    
    # Check signal lengths
    signal_lengths = [len(item['ppg_signal']) for item in aligned_data]
    print(f"Signal length stats: min={min(signal_lengths)}, max={max(signal_lengths)}, mean={np.mean(signal_lengths):.1f}")
    
    return aligned_data

def prepare_wellby_signals(aligned_data, target_length=None, skip_start_samples=100):
    """
    Prepare PPG signals for CNN training (handle variable lengths)
    
    Args:
        aligned_data: List of aligned session data
        target_length: Target signal length (if None, use shortest after skipping start)
        skip_start_samples: Number of samples to skip at the beginning (default 100)
    
    Returns:
        processed_data: List with processed signals
    """
    print(f"Preparing PPG signals for CNN (skipping first {skip_start_samples} samples)...")
    
    # Determine target length after skipping start samples
    if target_length is None:
        signal_lengths = [len(item['ppg_signal']) - skip_start_samples for item in aligned_data 
                         if len(item['ppg_signal']) > skip_start_samples]
        if len(signal_lengths) == 0:
            print("ERROR: All signals are shorter than skip_start_samples!")
            return []
        target_length = min(signal_lengths)
        print(f"Using shortest signal length after skipping: {target_length} samples")
    
    processed_data = []
    skipped_sessions = 0
    
    for item in aligned_data:
        ppg_signal = item['ppg_signal']
        
        # Skip initial noisy samples
        if len(ppg_signal) <= skip_start_samples:
            print(f"  Skipping session {item['session_id']}: too short ({len(ppg_signal)} <= {skip_start_samples})")
            skipped_sessions += 1
            continue
            
        trimmed_signal = ppg_signal[skip_start_samples:]
        
        # Handle variable length - truncate to target_length
        if len(trimmed_signal) >= target_length:
            processed_signal = trimmed_signal[:target_length]
        else:
            # Zero pad if shorter
            padding = target_length - len(trimmed_signal)
            processed_signal = np.pad(trimmed_signal, (0, padding), mode='constant', constant_values=0)
        
        # Minimal preprocessing for CNN
        clean_signal = processed_signal[~np.isnan(processed_signal)]
        if len(clean_signal) == 0:
            skipped_sessions += 1
            continue
            
        # Basic standardization only
        standardized_signal = standardize(clean_signal)
        
        # Normalize for CNN
        normalized_signal = (standardized_signal - np.mean(standardized_signal)) / (np.std(standardized_signal) + 1e-8)
        
        # Update item with processed signal
        item_copy = item.copy()
        item_copy['cnn_signal'] = normalized_signal
        item_copy['raw_signal'] = trimmed_signal  # Keep trimmed signal for TD features
        item_copy['signal_length'] = target_length
        item_copy['skip_start_samples'] = skip_start_samples
        
        processed_data.append(item_copy)
    
    print(f"Successfully processed {len(processed_data)} signals")
    print(f"Skipped {skipped_sessions} sessions (too short or processing failed)")
    return processed_data

def train_wellby_cnn_with_cv(processed_data, task='stress', device=None, epochs=30):
    """
    Train CNN using 3-fold CV for Wellby stress/sleep detection
    
    Args:
        processed_data: List of processed session data
        task: 'stress' or 'sleep'
        device: PyTorch device
        epochs: Number of training epochs
    
    Returns:
        trained_models: List of trained models from each fold
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    print(f"Training CNN for Wellby {task} detection using 3-fold CV...")
    print(f"Using device: {device}")
    
    # Prepare data for CV
    participants = [item['participant'] for item in processed_data]
    labels = [item[f'{task}_label'] for item in processed_data]
    signals = [item['cnn_signal'] for item in processed_data]
    
    print(f"Total sessions: {len(processed_data)}")
    print(f"Label distribution: {np.bincount(labels)}")
    print(f"Unique participants: {len(set(participants))}")
    
    # Convert to arrays
    X = np.array(signals)
    y = np.array(labels)
    groups = np.array(participants)
    
    # 3-fold stratified group CV
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    
    trained_models = []
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        print(f"\n--- Fold {fold_idx + 1}/3 ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
        print(f"Train labels: {np.bincount(y_train)}, Val labels: {np.bincount(y_val)}")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
        X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create data loaders (small batches for tiny dataset)
        from torch.utils.data import DataLoader, TensorDataset
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Very small batch
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        # Initialize model for this fold
        model = TrainableCNNFeatureExtractor(input_length=len(X_train[0])).to(device)
        
        # Training setup with strong regularization
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)  # Lower LR, higher weight decay
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10
        
        # Save initial model state
        torch.save(model.state_dict(), f'temp_best_wellby_cnn_fold_{fold_idx}.pth')
        
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
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'   Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%')
            
            # Early stopping with guaranteed model saving
            if val_acc >= best_val_acc:  # Changed > to >= to handle ties
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), f'temp_best_wellby_cnn_fold_{fold_idx}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'   Early stopping at epoch {epoch+1}')
                    break
        
        # Load best model for this fold (guaranteed to exist now)
        model.load_state_dict(torch.load(f'temp_best_wellby_cnn_fold_{fold_idx}.pth'))
        trained_models.append(model)
        fold_results.append(best_val_acc)
        
        print(f'Fold {fold_idx + 1} completed. Best val accuracy: {best_val_acc:.1f}%')
        
        torch.cuda.empty_cache()
    
    print(f'\nCV Training completed. Fold accuracies: {[f"{acc:.1f}%" for acc in fold_results]}')
    print(f'Mean CV accuracy: {np.mean(fold_results):.1f}% ± {np.std(fold_results):.1f}%')
    
    return trained_models

def extract_wellby_features(processed_data, trained_models, task='stress', fs=50):
    """
    Extract both CNN and TD features for all Wellby sessions
    
    Args:
        processed_data: List of processed session data
        trained_models: List of trained CNN models from CV
        task: 'stress' or 'sleep'
        fs: Sampling frequency
    
    Returns:
        features_list: List of feature dictionaries
    """
    print(f"Extracting hybrid features for Wellby {task} detection...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features_list = []
    failed_extractions = 0
    
    for item in tqdm(processed_data, desc="Extracting features"):
        try:
            # ===== CNN FEATURE EXTRACTION =====
            # Use ensemble of models from CV (average features)
            cnn_features_ensemble = []
            
            for model in trained_models:
                model.eval()
                with torch.no_grad():
                    signal_tensor = torch.FloatTensor(item['cnn_signal']).unsqueeze(0).unsqueeze(0)  # (1, 1, length)
                    signal_tensor = signal_tensor.to(device)
                    cnn_features = model.extract_features_only(signal_tensor)
                    cnn_features_np = cnn_features.cpu().numpy().flatten()
                    cnn_features_ensemble.append(cnn_features_np)
            
            # Average features across folds
            cnn_features_avg = np.mean(cnn_features_ensemble, axis=0)
            
            # ===== TIME-DOMAIN FEATURE EXTRACTION =====
            # Process signal for TD features (following WESAD/AKTIVES approach)
            # Use the trimmed signal (already has start samples removed)
            raw_signal = item['raw_signal']  # This is already trimmed
            
            # Remove NaN values
            clean_signal = raw_signal[~np.isnan(raw_signal)]
            if len(clean_signal) == 0:
                failed_extractions += 1
                continue
                
            # Standardize
            standardized_signal = standardize(clean_signal)
            
            # Apply filtering for TD features (following WESAD approach)
            bp_signal = bandpass_filter(standardized_signal, 0.5, 10, fs, order=2)
            smoothed_signal = moving_average_filter(bp_signal, window_size=5)
            
            # Extract TD features
            td_stats = get_ppg_features(ppg_seg=smoothed_signal.tolist(), 
                                      fs=fs, 
                                      label=item[f'{task}_label'], 
                                      calc_sq=False)
            
            if td_stats is None or len(cnn_features_avg) != 32:
                failed_extractions += 1
                continue
            
            # ===== COMBINE FEATURES =====
            features_dict = {
                'session_id': item['session_id'],
                'participant': item['participant'],
                'label': item[f'{task}_label'],
                'school': item['school'],
                'age': item['age'],
                'gender': item['gender'],
                'PSS': item['PSS'],
                'PSQI': item['PSQI'],
                'EPOCH': item['EPOCH'],
                'SQI': item['SQI']
            }
            
            # Add CNN features
            for i, feat in enumerate(cnn_features_avg):
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
            
            features_list.append(features_dict)
            
        except Exception as e:
            print(f"Error extracting features for session {item['session_id']}: {str(e)}")
            failed_extractions += 1
    
    print(f"Successfully extracted features from {len(features_list)}/{len(processed_data)} sessions")
    print(f"Failed extractions: {failed_extractions}")
    
    return features_list

def extract_all_wellby_hybrid_features():
    """
    Main function to extract hybrid features from Wellby data
    """
    print("="*70)
    print("WELLBY HYBRID FEATURE EXTRACTION")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Load and align data
    aligned_data = load_wellby_data()
    if aligned_data is None:
        print("Error: Could not load Wellby data!")
        return None
    
    # Step 2: Prepare signals for CNN
    processed_data = prepare_wellby_signals(aligned_data)
    if len(processed_data) == 0:
        print("Error: No signals could be processed!")
        return None
    
    # Step 3: Train CNN for both tasks
    print("\n" + "="*50)
    print("TRAINING CNNs")
    print("="*50)
    
    # Train for stress detection
    print("\n--- Training for STRESS detection ---")
    stress_models = train_wellby_cnn_with_cv(processed_data, task='stress', device=device)
    
    # Train for sleep/fatigue detection  
    print("\n--- Training for SLEEP/FATIGUE detection ---")
    sleep_models = train_wellby_cnn_with_cv(processed_data, task='sleep', device=device)
    
    # Step 4: Extract features for both tasks
    print("\n" + "="*50)
    print("EXTRACTING FEATURES")
    print("="*50)
    
    # Extract stress features
    print("\n--- Extracting STRESS features ---")
    stress_features = extract_wellby_features(processed_data, stress_models, task='stress')
    
    # Extract sleep features
    print("\n--- Extracting SLEEP features ---")
    sleep_features = extract_wellby_features(processed_data, sleep_models, task='sleep')
    
    # Step 5: Create and standardize feature DataFrames
    print("\n" + "="*50)
    print("FINALIZING FEATURES")
    print("="*50)
    
    # Process stress features
    if len(stress_features) > 0:
        stress_df = pd.DataFrame(stress_features)
        
        # Identify feature columns
        cnn_columns = [col for col in stress_df.columns if col.startswith('cnn_feature_')]
        td_columns = [col for col in stress_df.columns if col not in cnn_columns and 
                      col not in ['session_id', 'participant', 'label', 'school', 'age', 'gender', 'PSS', 'PSQI', 'EPOCH', 'SQI']]
        
        print(f"Stress features - CNN: {len(cnn_columns)}, TD: {len(td_columns)}")
        
        # Standardize features separately
        scaler_cnn = StandardScaler()
        scaler_td = StandardScaler()
        
        stress_df[cnn_columns] = scaler_cnn.fit_transform(stress_df[cnn_columns])
        if len(td_columns) > 0:
            stress_df[td_columns] = scaler_td.fit_transform(stress_df[td_columns])
    
    # Process sleep features
    if len(sleep_features) > 0:
        sleep_df = pd.DataFrame(sleep_features)
        
        # Identify feature columns (same structure as stress)
        cnn_columns = [col for col in sleep_df.columns if col.startswith('cnn_feature_')]
        td_columns = [col for col in sleep_df.columns if col not in cnn_columns and 
                      col not in ['session_id', 'participant', 'label', 'school', 'age', 'gender', 'PSS', 'PSQI', 'EPOCH', 'SQI']]
        
        print(f"Sleep features - CNN: {len(cnn_columns)}, TD: {len(td_columns)}")
        
        # Standardize features separately
        scaler_cnn = StandardScaler()
        scaler_td = StandardScaler()
        
        sleep_df[cnn_columns] = scaler_cnn.fit_transform(sleep_df[cnn_columns])
        if len(td_columns) > 0:
            sleep_df[td_columns] = scaler_td.fit_transform(sleep_df[td_columns])
    
    # Step 6: Save results
    output_dir = "../../data/Wellby_hybrid_features/"
    os.makedirs(output_dir, exist_ok=True)
    
    if len(stress_features) > 0:
        stress_output = os.path.join(output_dir, "wellby_stress_hybrid_features.csv")
        stress_df.to_csv(stress_output, index=False)
        print(f"Stress features saved to: {stress_output}")
    
    if len(sleep_features) > 0:
        sleep_output = os.path.join(output_dir, "wellby_sleep_hybrid_features.csv")
        sleep_df.to_csv(sleep_output, index=False)
        print(f"Sleep features saved to: {sleep_output}")
    
    # Step 7: Summary statistics
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    
    print(f"Total sessions processed: {len(processed_data)}")
    print(f"Stress features extracted: {len(stress_features)}")
    print(f"Sleep features extracted: {len(sleep_features)}")
    
    if len(stress_features) > 0:
        print(f"\nStress dataset characteristics:")
        print(f"  Total features: {len(cnn_columns) + len(td_columns)}")
        print(f"  CNN features: {len(cnn_columns)}")
        print(f"  TD features: {len(td_columns)}")
        print(f"  Label distribution: {stress_df['label'].value_counts().to_dict()}")
    
    if len(sleep_features) > 0:
        print(f"\nSleep dataset characteristics:")
        print(f"  Label distribution: {sleep_df['label'].value_counts().to_dict()}")
    
    # Clean up temporary files
    for i in range(3):  # 3 folds
        for temp_file in [f'temp_best_wellby_cnn_fold_{i}.pth']:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    return stress_df if len(stress_features) > 0 else None, sleep_df if len(sleep_features) > 0 else None

# Run the extraction
if __name__ == "__main__":
    print("Starting Wellby hybrid feature extraction...")
    stress_results, sleep_results = extract_all_wellby_hybrid_features()
    
    if stress_results is not None or sleep_results is not None:
        print("✓ Extraction completed successfully!")
    else:
        print("✗ Extraction failed!")
# %%
