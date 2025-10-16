"""
1D CNN Model for Biosonar Detection
 
This module detects the presence of odontocetes (toothed cetaceans such as dolphins,
orcas, sperm whales) in underwater audio recordings. It uses an approach combining
CNN audio signal analysis and extraction of specific features related to the
echolocation clicks produced by these animals.
"""
 
import os
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import librosa
from scipy import signal
from tqdm import tqdm
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
 
def explore_audio_files(audio_dir, sample_size=5):
    """
    Explores the characteristics of audio files in the directory.
    """
    print(f"Exploring audio files in {audio_dir}...")
 
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    if not wav_files:
        print("No WAV files found in the directory.")
        return
 
    print(f"Total number of WAV files: {len(wav_files)}")
    np.random.seed(42)
    sample_files = np.random.choice(wav_files, min(sample_size, len(wav_files)), replace=False)
 
    sample_rates = []
    durations = []
 
    for file_name in sample_files:
        file_path = os.path.join(audio_dir, file_name)
        try:
            audio, sr = librosa.load(file_path, sr=None)
            duration = len(audio) / sr
            print(f"File: {file_name}")
            print(f"  - Sample rate: {sr} Hz")
            print(f"  - Duration: {duration:.4f} seconds")
            print(f"  - Number of samples: {len(audio)}")
            sample_rates.append(sr)
            durations.append(duration)
        except Exception as e:
            print(f"Error analyzing {file_name}: {e}")
 
    if sample_rates:
        print("\nStatistics on the sample:")
        print(f"  - Average sample rate: {np.mean(sample_rates):.2f} Hz")
        print(f"  - Average duration: {np.mean(durations):.4f} seconds")
        print(f"  - Minimum duration: {np.min(durations):.4f} seconds")
        print(f"  - Maximum duration: {np.max(durations):.4f} seconds")
 
    return sample_rates, durations
 
def bandpass_filter(audio, sr, low_freq=5000, high_freq=120000):
    """Applies a bandpass filter to the audio signal."""
    sos = signal.butter(6, [low_freq, high_freq], 'bandpass', fs=sr, output='sos')
    filtered_audio = signal.sosfiltfilt(sos, audio)
    return filtered_audio
 
def normalize_audio(audio):
    """Normalizes the audio signal."""
    return audio / (np.max(np.abs(audio)) + 1e-10)
 
def convert_to_mono(audio):
    """Converts a stereo signal to mono if needed."""
    if audio.ndim > 1 and audio.shape[1] > 1:
        return np.mean(audio, axis=1)
    return audio
 
def load_and_preprocess_file(file_path, target_length, sr=None):
    """
    Loads and preprocesses an audio file.
    """
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None, None
 
        audio, sample_rate = librosa.load(file_path, sr=sr)
 
        audio = convert_to_mono(audio)
 
        audio = bandpass_filter(audio, sample_rate)
 
        audio = normalize_audio(audio)
 
        # Adjust the length
        if len(audio) > target_length:
            start = np.random.randint(0, len(audio) - target_length)
            audio = audio[start:start + target_length]
        elif len(audio) < target_length:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
 
        return audio, sample_rate
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None
 
def hilbert_envelope(audio):
    """Extracts the signal envelope via Hilbert transform (ideal for detecting clicks)."""
    analytic_signal = signal.hilbert(audio)
    envelope = np.abs(analytic_signal)
    return envelope
 
def extract_click_features(audio, sr, threshold=0.5):
    """
    Extracts features specific to odontocete clicks.
    """
    if audio is None or len(audio) == 0:
        return np.zeros(4) 
 
    # Get the envelope via Hilbert
    envelope = hilbert_envelope(audio)
    normalized_env = envelope / (np.max(envelope) + 1e-10)
 
    # Detect peaks (potential clicks)
    peaks, _ = signal.find_peaks(normalized_env, height=threshold, distance=int(0.001*sr))
 
    if len(peaks) == 0:
        return np.zeros(4) 
 
    # Measurements on the clicks
    peak_values = normalized_env[peaks]
    intervals = np.diff(peaks) / sr  # Intervals in seconds
 
    features = np.array([
        len(peaks),                    # Number of clicks
        np.mean(peak_values),          # Average amplitude
        np.std(intervals) if len(intervals) > 0 else 0,  # Regularity of intervals
        np.mean(intervals) if len(intervals) > 0 else 0  # Average interval
    ])
 
    return features
 
class CNN1DDataset(Dataset):
    """PyTorch Dataset for cetacean detection."""
 
    def __init__(self, file_paths, labels=None, target_length=51200, sample_rate=None, transform=None):
        """
        Initializes the dataset.
 
        Args:
            file_paths: List of audio file paths
            labels: Labels (None for test set)
            target_length: Target length for audio files
            sample_rate: Sample rate (None to keep original)
            transform: Transformations to apply
        """
        self.file_paths = file_paths
        self.labels = labels
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.transform = transform
        self.is_test = (labels is None)
 
    def __len__(self):
        return len(self.file_paths)
 
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
 
        audio, sr = load_and_preprocess_file(file_path, self.target_length, self.sample_rate)
 
        if audio is None:
            audio = np.zeros(self.target_length, dtype=np.float32)
            sr = self.sample_rate if self.sample_rate else 256000
 
        click_features = extract_click_features(audio, sr)
 
        if self.transform:
            audio = self.transform(audio)
 
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        click_features_tensor = torch.tensor(click_features, dtype=torch.float32)
 
        if self.is_test:
            return audio_tensor, click_features_tensor, file_path
        else:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return audio_tensor, click_features_tensor, label_tensor
 
def prepare_data_loaders(df, target_length, sample_rate=None, batch_size=32, valid_size=0.2, seed=42):
    """
    Prepares DataLoaders for training and validation.
    """
    if 'file_path' in df.columns:
        sample_paths = df['file_path'].iloc[:5].tolist()
        for path in sample_paths:
            if not os.path.exists(path):
                print(f"WARNING: Path {path} does not exist")
                if path.endswith('.wav.wav'):
                    correct_path = path[:-4]
                    if os.path.exists(correct_path):
                        print(f"Possible correction: {correct_path}")
 
    train_df, valid_df = train_test_split(
        df, test_size=valid_size, random_state=seed, stratify=df['pos_label']
    )
 
    train_dataset = CNN1DDataset(
        train_df['file_path'].values,
        train_df['pos_label'].values,
        target_length,
        sample_rate
    )
 
    valid_dataset = CNN1DDataset(
        valid_df['file_path'].values,
        valid_df['pos_label'].values,
        target_length,
        sample_rate
    )
 
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
 
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
 
    return train_loader, valid_loader
 
def prepare_test_loader(test_dir, target_length, sample_rate=None, batch_size=32):
    """
    Prepares the DataLoader for the test set.
    """
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.wav')]
    test_files.sort()
 
    test_dataset = CNN1DDataset(
        test_files,
        None,
        target_length,
        sample_rate
    )
 
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
 
    return test_loader
 
class CNN1DModel(nn.Module):
    def __init__(self, input_size=51200, click_features_size=4, dropout_rate=0.3):
        """
        Optimized CNN model for cetacean detection.
        """
        super(CNN1DModel, self).__init__()
 
        self.audio_features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
 
            nn.Conv1d(32, 48, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
 
            nn.Conv1d(48, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )
 
        conv_output_size = 64 * (input_size // 16)
 
        self.audio_fc = nn.Sequential(
            nn.Linear(conv_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
 
        self.click_fc = nn.Sequential(
            nn.Linear(click_features_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
 
        self.classifier = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
 
    def forward(self, audio, click_features):
        audio_features = self.audio_features(audio)
        audio_features = torch.flatten(audio_features, 1)
        audio_features = self.audio_fc(audio_features)
 
        click_processed = self.click_fc(click_features)
        combined = torch.cat((audio_features, click_processed), dim=1)
        output = self.classifier(combined)
 
        return output
 
def train_model(model, train_loader, valid_loader, num_epochs=20, lr=0.0001,
                patience=5, save_path='best_model.pt'):
    """
    Trains the CNN model.
 
    Args:
        model: Model to train
        train_loader, valid_loader: Training and validation DataLoaders
        num_epochs: Maximum number of epochs
        lr: Learning rate
        patience: Number of epochs to wait before early stopping
        save_path: Path to save the best model
 
    Returns:
        Training history
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
 
    # Define scheduler that adjusts learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=4, factor=0.5, verbose=True
    )
 
    # For tracking metrics
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    valid_aucs = []
 
    # For early stopping
    best_valid_loss = float('inf')
    best_auc = 0.0
    no_improve_epochs = 0
 
    # Training
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
 
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
 
        for batch in tqdm(train_loader, desc="Training"):
            audio, click_features, labels = batch
            audio = audio.to(device)
            click_features = click_features.to(device)
            labels = labels.to(device).unsqueeze(1)
 
            # Forward pass
            optimizer.zero_grad()
            outputs = model(audio, click_features)
            loss = criterion(outputs, labels)
 
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
 
            # Metrics
            train_loss += loss.item() * audio.size(0)
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
 
            del audio, click_features, labels, outputs, predicted
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
 
        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
 
        all_preds = []
        all_labels = []
 
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation"):
                audio, click_features, labels = batch
                audio = audio.to(device)
                click_features = click_features.to(device)
                labels = labels.to(device).unsqueeze(1)
 
                outputs = model(audio, click_features)
                loss = criterion(outputs, labels)
 
                valid_loss += loss.item() * audio.size(0)
                predicted = (outputs > 0.5).float()
                valid_correct += (predicted == labels).sum().item()
                valid_total += labels.size(0)
 
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
 
                del audio, click_features, labels, outputs, predicted
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
 
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        epoch_valid_loss = valid_loss / valid_total
        epoch_valid_acc = valid_correct / valid_total
 
        all_preds_np = np.array(all_preds).flatten()
        all_labels_np = np.array(all_labels).flatten()
        epoch_valid_auc = roc_auc_score(all_labels_np, all_preds_np)
 
        scheduler.step(epoch_valid_auc)
 
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        valid_losses.append(epoch_valid_loss)
        valid_accs.append(epoch_valid_acc)
        valid_aucs.append(epoch_valid_auc)
 
        print(f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}')
        print(f'Valid Loss: {epoch_valid_loss:.4f} | Valid Acc: {epoch_valid_acc:.4f}')
        print(f'Valid AUC: {epoch_valid_auc:.4f}')
 
        # Classification report
        all_preds_binary = (all_preds_np > 0.5).astype(int)
        print("\nClassification Report:")
        print(classification_report(all_labels_np, all_preds_binary,
                                   target_names=['Non-Cetacean', 'Cetacean']))
 
        if epoch_valid_auc > best_auc:
            best_auc = epoch_valid_auc
            torch.save(model.state_dict(), save_path)
            print(f'Best model saved! (AUC: {best_auc:.4f})')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
 
        # Early stopping
        if no_improve_epochs >= patience:
            print(f'Early stopping after {epoch+1} epochs without improvement')
            break
 
        print(f'Epoch time: {time.time() - start_time:.2f}s')
        print()
 
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
 
    # Load the best model
    model.load_state_dict(torch.load(save_path))
    print(f'Best model loaded (AUC: {best_auc:.4f})')
 
    # Return training history
    history = {
        'train_loss': train_losses,
        'valid_loss': valid_losses,
        'train_acc': train_accs,
        'valid_acc': valid_accs,
        'valid_auc': valid_aucs
    }
 
    return model, history
 
def plot_learning_curves(train_losses, valid_losses, train_accs, valid_accs):
    """Plots learning curves."""
    plt.figure(figsize=(12, 5))
 
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(valid_losses, label='Validation')
    plt.title('Loss Function')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
 
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(valid_accs, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
 
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()
 
def plot_confusion_matrix(y_true, y_pred):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.savefig('confusion_matrix.png')
    plt.show()
 
def evaluate_model_detailed(model, data_loader):
    """
    Evaluates the model with detailed metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
 
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation"):
            audio, click_features, labels = batch
            audio = audio.to(device)
            click_features = click_features.to(device)
 
            # Forward pass
            outputs = model(audio, click_features)
            predicted = (outputs > 0.5).float()
 
            # Collect for evaluation
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
 
            del audio, click_features, labels, outputs, predicted
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
 
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
 
    roc_auc = roc_auc_score(all_labels, all_preds)
 
    all_preds_binary = (all_preds > 0.5).astype(int)
    accuracy = (all_preds_binary == all_labels).mean()
 
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds_binary,
                               target_names=['Non-Cetacean', 'Cetacean']))
 
    plot_confusion_matrix(all_labels, all_preds_binary)
 
    return roc_auc
 
def predict_test(model, test_loader):
    """
    Makes predictions on the test set and returns probability values instead of binary labels.
    """
    model.eval()
    all_probs = []  
    all_file_paths = []
 
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predictions"):
            audio, click_features, file_paths = batch
            audio = audio.to(device)
            click_features = click_features.to(device)
 
            # Forward pass
            outputs = model(audio, click_features)
            
            # Récupérer directement les probabilités (sans conversion en binaire)
            all_probs.extend(outputs.cpu().numpy())
            all_file_paths.extend(file_paths)
 
            del audio, click_features, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
 
    # Extract file names from paths
    file_names = [os.path.basename(fp) for fp in all_file_paths]
    file_ids = [os.path.splitext(fn)[0] for fn in file_names]
 
    # Create predictions DataFrame with probability values
    df_preds = pd.DataFrame({
        'id': file_ids,
        'pos_label': np.array(all_probs).flatten()  # Probabilités brutes au lieu de valeurs binaires
    })
 
    return df_preds
 
def main():
    """
    Main function that executes the complete training and prediction pipeline.
    """
    # Parameters and paths
    AUDIO_DIR = "/content/drive/MyDrive/VO_TARVERDIAN/data/data/X_train"
    LABELS_FILE = "/content/drive/MyDrive/VO_TARVERDIAN/data/data/Y_train_ofTdMHi.csv"
    TEST_DIR = "/content/drive/MyDrive/VO_TARVERDIAN/data/data/X_test"
    MODEL_SAVE_PATH = "/content/drive/MyDrive/VO_TARVERDIAN/final_CNN/best_cetacean_model.pt"
    PREDICTION_FILE = "/content/drive/MyDrive/VO_TARVERDIAN/final_CNN/predictions.csv"
 
    # Audio parameters
    SAMPLE_RATE = 256000
    AUDIO_DURATION = 0.20
    TARGET_LENGTH = int(SAMPLE_RATE * AUDIO_DURATION)
 
    # Training parameters
    BATCH_SIZE = 16
    EPOCHS = 25
    PATIENCE = 5
    LEARNING_RATE = 0.001
 
    # Explore audio files
    explore_audio_files(AUDIO_DIR)
 
    # Load labels
    labels_df = pd.read_csv(LABELS_FILE)
    print(f"Loaded {len(labels_df)} entries from {LABELS_FILE}")
    print(labels_df.head())
 
    # Add the complete file path
    labels_df['file_path'] = labels_df['id'].apply(lambda x: os.path.join(AUDIO_DIR, x))
 
    # Display class distribution
    print(f"\nClass distribution:")
    class_counts = labels_df['pos_label'].value_counts()
    print(f"Class 0 (Non-Cetacean): {class_counts[0]}")
    print(f"Class 1 (Cetacean): {class_counts[1]}")
 
    # Determine extracted features size
    test_audio = np.zeros(TARGET_LENGTH)
    test_sr = SAMPLE_RATE
    test_features = extract_click_features(test_audio, test_sr)
    num_features = len(test_features)
    print(f"Number of extracted features: {num_features}")
 
    # Prepare DataLoaders
    train_loader, valid_loader = prepare_data_loaders(
        labels_df, TARGET_LENGTH, SAMPLE_RATE, BATCH_SIZE
    )
    print(f"\nNumber of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(valid_loader)}")
 
    # Create the model with the right number of features
    model = CNN1DModel(TARGET_LENGTH, click_features_size=num_features).to(device)
 
    # Display model summary
    print("\nModel Architecture:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params:,}")
    print(f"Number of trainable parameters: {trainable_params:,}")
 
    # Train the model
    print("\nStarting training...")
    trained_model, history = train_model(
        model, train_loader, valid_loader,
        num_epochs=EPOCHS,
        lr=LEARNING_RATE,
        patience=PATIENCE,
        save_path=MODEL_SAVE_PATH
    )
 
    # Plot learning curves
    plot_learning_curves(
        history['train_loss'], history['valid_loss'],
        history['train_acc'], history['valid_acc']
    )
 
    print("\nFinal evaluation on validation set:")
    evaluate_model_detailed(trained_model, valid_loader)
   
    print("\nPreparing test data...")
    test_loader = prepare_test_loader(TEST_DIR, TARGET_LENGTH, SAMPLE_RATE, BATCH_SIZE)
 
    print("\nMaking predictions on test set...")
    predictions_df = predict_test(trained_model, test_loader)
    predictions_df.to_csv(PREDICTION_FILE, index=False)
    print(f"\nPredictions saved to {PREDICTION_FILE}")
    print(f"Preview of predictions:")
    print(predictions_df.head(10))
 
  
    print("\nTraining and prediction completed successfully!")
 
if __name__ == "__main__":
    main()
