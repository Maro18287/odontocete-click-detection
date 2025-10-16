"""
2D CNN Model for Biosonar Detection

This module provides a 2D CNN model for detecting biosonar signals in audio recordings
along with utilities for data preparation, training, and evaluation.
"""

import os
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, precision_recall_fscore_support
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class CNN2DModelTrainer(Dataset):
    """
    Dataset class for loading and preprocessing biosonar audio files.
    """
    def __init__(self, file_ids, labels=None, is_train=True, augment=False, preload=True, 
                 folder_path=None, sr=256000, duration=0.2, n_fft=1024, hop_length=128, n_mels=128):
        """
        Initialize the dataset.
        
        Args:
            file_ids (list): List of file IDs or filenames
            labels (list, optional): List of labels corresponding to file_ids
            is_train (bool): Whether this is a training dataset
            augment (bool): Whether to apply data augmentation
            preload (bool): Whether to preload all data into memory
            folder_path (str): Path to audio files
            sr (int): Sampling rate
            duration (float): Duration in seconds
            n_fft (int): FFT window size
            hop_length (int): Hop length for STFT
            n_mels (int): Number of mel bands
        """
        self.file_ids = file_ids
        self.labels = labels
        self.is_train = is_train
        self.augment = augment and is_train
        self.folder_path = folder_path
        self.preload = preload
        
        # Audio parameters
        self.sr = sr
        self.duration = duration
        self.samples = int(sr * duration)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.noise_level = 0.005

        self.min_val = -80
        self.max_val = 0

        self.specs = []
        valid_indices = []

        if preload:
            print(f"Loading {len(self.file_ids)} audio files...")
            for i, file_id in enumerate(tqdm(self.file_ids, desc="Loading spectrograms")):
                if file_id.endswith('.wav'):
                    file_path = os.path.join(self.folder_path, file_id)
                else:
                    file_path = os.path.join(self.folder_path, f"{file_id}.wav")

                if os.path.exists(file_path):
                    try:
                        mel_spec_db = self._process_audio_file(file_path)
                        if mel_spec_db is not None:
                            self.specs.append(mel_spec_db)
                            valid_indices.append(i)
                    except Exception as e:
                        continue
                else:
                    pass

            if len(self.specs) == 0:
                raise ValueError("No audio files could be loaded correctly. Check paths and file formats.")

            self.file_ids = [self.file_ids[i] for i in valid_indices]
            if self.labels is not None:
                self.labels = [self.labels[i] for i in valid_indices]

            all_values = []
            for spec in self.specs:
                sample_indices = np.random.choice(spec.size, min(1000, spec.size), replace=False)
                all_values.extend(spec.flatten()[sample_indices])

            if all_values:
                self.min_val = np.min(all_values)
                self.max_val = np.max(all_values)

            for i in range(len(self.specs)):
                self.specs[i] = (self.specs[i] - self.min_val) / (self.max_val - self.min_val)

            print(f"Loading completed. {len(self.specs)} files processed successfully.")
            print(f"Spectrogram value range: [{self.min_val}, {self.max_val}]")

            if len(self.specs) > 0:
                print(f"Spectrogram size: {self.specs[0].shape}")

                if is_train and labels is not None and len(self.specs) > 1:
                    for label in np.unique(labels).astype(int):
                        indices = [i for i, l in enumerate(self.labels) if int(l) == label]
                        if indices:
                            idx = random.choice(indices)
                            class_name = 'Biosonar' if label == 1 else 'Noise'
                            self._plot_spectrogram(self.specs[idx], f"Example of class {label} - {class_name}")

            gc.collect()
        else:
            valid_files = []
            for file_id in file_ids:
                if file_id.endswith('.wav'):
                    file_path = os.path.join(self.folder_path, file_id)
                else:
                    file_path = os.path.join(self.folder_path, f"{file_id}.wav")
                if os.path.exists(file_path):
                    valid_files.append(file_id)

            self.file_ids = valid_files
            if self.labels is not None:
                self.labels = [labels[i] for i, file_id in enumerate(file_ids) if file_id in valid_files]

    def _plot_spectrogram(self, spec, title=None):
        """
        Visualize a spectrogram.
        
        Args:
            spec (numpy.ndarray): Spectrogram data
            title (str, optional): Title for the plot
        """
        plt.figure(figsize=(10, 4))
        plt.imshow(spec, aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        if title:
            plt.title(title)
        plt.show()

    def _add_noise(self, spec):
        """
        Add Gaussian noise to the spectrogram.
        
        Args:
            spec (numpy.ndarray): Input spectrogram
            
        Returns:
            numpy.ndarray: Spectrogram with added noise
        """
        noise = np.random.normal(0, self.noise_level, spec.shape)
        return spec + noise

    def _time_shift(self, spec, max_shift_percent=0.1):
        """
        Temporal shift of the spectrogram.
        
        Args:
            spec (numpy.ndarray): Input spectrogram
            max_shift_percent (float): Maximum shift as percentage of spectrogram width
            
        Returns:
            numpy.ndarray: Shifted spectrogram
        """
        max_shift = int(spec.shape[1] * max_shift_percent)
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift > 0:
            return np.pad(spec[:, :-shift], ((0, 0), (shift, 0)), mode='constant')
        elif shift < 0:
            return np.pad(spec[:, -shift:], ((0, 0), (0, -shift)), mode='constant')
        return spec

    def _process_audio_file(self, file_path):
        """
        Process a single audio file and return its mel spectrogram.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            numpy.ndarray: Mel spectrogram in dB scale or None if processing fails
        """
        try:
            try:
                audio, sr = sf.read(file_path)
                if sr != self.sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
            except Exception:
                audio, sr = librosa.load(file_path, sr=self.sr)

            if len(audio) > self.samples:
                center = len(audio) // 2
                half_samples = self.samples // 2
                audio = audio[center - half_samples:center + half_samples]
            elif len(audio) < self.samples:
                audio = np.pad(audio, (0, self.samples - len(audio)), 'constant')

            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=1000,  # Biosonars are mainly in high frequencies
                fmax=self.sr/2    # Nyquist frequency
            )

            # Convert to decibels
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            del audio, mel_spec

            return mel_spec_db

        except Exception as e:
            return None

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.file_ids) if not self.preload else len(self.specs)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            tuple: (spectrogram tensor, label) or (spectrogram tensor, file_id)
        """
        if self.preload:
            spec = self.specs[idx].copy()
        else:
            if idx >= len(self.file_ids):
                raise IndexError(f"Index {idx} out of range for dataset with {len(self.file_ids)} files")

            file_id = self.file_ids[idx]
            if file_id.endswith('.wav'):
                file_path = os.path.join(self.folder_path, file_id)
            else:
                file_path = os.path.join(self.folder_path, f"{file_id}.wav")

            mel_spec_db = self._process_audio_file(file_path)
            if mel_spec_db is None:
                mel_spec_db = np.zeros((self.n_mels, int(self.samples / self.hop_length) + 1))

            spec = (mel_spec_db - self.min_val) / (self.max_val - self.min_val)

        if self.augment and random.random() < 0.5:
            if random.random() < 0.5:
                spec = self._add_noise(spec)
            if random.random() < 0.5:
                spec = self._time_shift(spec)

        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

        if self.is_train and self.labels is not None:
            label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            return spec_tensor, label
        else:
            return spec_tensor, self.file_ids[idx]


class DCNN(nn.Module):
    """
    2D CNN model for biosonar detection.
    """
    def __init__(self, dropout_rate=0.5):
        """
        Initialize the model.
        
        Args:
            dropout_rate (float): Dropout rate for the fully connected layers
        """
        super(DCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output logits
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class DCNN_Trainer:
    """
    Trainer class for the 2D CNN model.
    """
    def __init__(self, model, device=None, learning_rate=0.001, batch_size=16, epochs=20, patience=3):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): The model to train
            device (torch.device): Device to use for training
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            epochs (int): Maximum number of epochs
            patience (int): Early stopping patience
        """
        self.model = model
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
    def train(self, train_loader, val_loader):
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): DataLoader for training data
            val_loader (DataLoader): DataLoader for validation data
            
        Returns:
            dict: Training history
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.patience//3, verbose=True
        )

        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        val_aucs = []

        best_val_loss = float('inf')
        best_model_state = self.model.state_dict().copy()
        patience_counter = 0

        best_val_preds = []
        best_val_targets = []
        best_val_probs = []

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} - Training"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                del inputs, labels, outputs, loss, predicted
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total
            train_losses.append(epoch_train_loss)
            train_accs.append(epoch_train_acc)

            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            all_preds = []
            all_targets = []
            all_probs = []

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.epochs} - Validation"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    probs = torch.nn.functional.softmax(outputs, dim=1)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())

                    del inputs, labels, outputs, loss, predicted, probs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)

            epoch_val_auc = roc_auc_score(all_targets, all_probs)
            val_aucs.append(epoch_val_auc)

            scheduler.step(epoch_val_loss)

            precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')

            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
            print(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val AUC: {epoch_val_auc:.4f}")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = self.model.state_dict().copy()
                best_val_preds = all_preds
                best_val_targets = all_targets
                best_val_probs = all_probs
                patience_counter = 0
                print("  => Best model saved!")
            else:
                patience_counter += 1
                print(f"  => No improvement for {patience_counter} epochs")
                if patience_counter >= self.patience:
                    print("Early stopping!")
                    break

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.model.load_state_dict(best_model_state)

        print("\nFinal metrics for the best model:")
        print(classification_report(best_val_targets, best_val_preds, target_names=["Noise", "Biosonar"]))

        best_auc = roc_auc_score(best_val_targets, best_val_probs)
        print(f"ROC AUC Score: {best_auc:.4f}")

        self._plot_roc_curve(best_val_targets, best_val_probs, best_auc)
        self._plot_confusion_matrix(best_val_targets, best_val_preds)
        
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'val_aucs': val_aucs
        }
        
        return history

    def _plot_roc_curve(self, targets, probs, auc):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(targets, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        
    def _plot_confusion_matrix(self, targets, preds):
        """Plot confusion matrix."""
        cm = confusion_matrix(targets, preds)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["Noise", "Biosonar"], rotation=45)
        plt.yticks(tick_marks, ["Noise", "Biosonar"])
        plt.tight_layout()
        plt.ylabel('True class')
        plt.xlabel('Predicted class')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.show()
        
    def predict(self, test_loader):
        """
        Generate predictions on test data.
        
        Args:
            test_loader (DataLoader): DataLoader for test data
            
        Returns:
            tuple: (predictions, file_ids)
        """
        self.model.eval()
        predictions = []
        file_ids = []

        print("Predicting on test data...")
        with torch.no_grad():
            for inputs, ids in tqdm(test_loader, desc="Prediction"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                pos_probs = probs[:, 1].cpu().numpy()
                predictions.extend(pos_probs)
                file_ids.extend(ids)

                del inputs, outputs, probs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        return predictions, file_ids
    
    def save_model(self, path, additional_info=None):
        """
        Save the model and additional information.
        
        Args:
            path (str): Path to save the model
            additional_info (dict, optional): Additional information to save
        """
        if additional_info is None:
            additional_info = {}
            
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            **additional_info
        }
        
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
        
    def load_model(self, path):
        """
        Load a saved model.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            dict: Additional information saved with the model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Remove model_state_dict from the checkpoint to get the additional info
        del checkpoint['model_state_dict']
        
        return checkpoint


def plot_training_history(history):
    """
    Plot training history metrics.
    
    Args:
        history (dict): Training history dictionary
    """
    plt.figure(figsize=(15, 5))

    # Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history['train_accs'], label='Train')
    plt.plot(history['val_accs'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Loss
    plt.subplot(1, 3, 2)
    plt.plot(history['train_losses'], label='Train')
    plt.plot(history['val_losses'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # ROC AUC
    plt.subplot(1, 3, 3)
    plt.plot(history['val_aucs'], label='Validation', color='green')
    plt.title('ROC AUC Score')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def check_audio_files(file_path, n_samples=5):
    """
    Check if audio files exist and display information about some of them.
    
    Args:
        file_path (str): Directory path containing audio files
        n_samples (int): Number of sample files to check
        
    Returns:
        bool: True if audio files exist and can be read, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"Directory {file_path} does not exist!")
        return False

    files = [f for f in os.listdir(file_path) if f.endswith('.wav')]
    if not files:
        print(f"No .wav files found in {file_path}")
        return False

    print(f"Found {len(files)} .wav files in {file_path}")

    sample_files = random.sample(files, min(n_samples, len(files)))
    for file in sample_files:
        full_path = os.path.join(file_path, file)
        try:
            audio, sr = sf.read(full_path)
            duration = len(audio) / sr
            print(f"  {file}: {len(audio)} samples, {sr} Hz, {duration:.3f} seconds")
        except Exception as e:
            print(f"  {file}: ERROR - {str(e)}")

    return True


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")