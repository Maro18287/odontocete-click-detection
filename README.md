# 🐬 Odontocete Click Detection

Detection of odontocete (toothed whale) echolocation clicks using underwater acoustic recordings from the **CARI’MAM** project (2017–2021).  
The goal is to identify **extreme wave clicks** likely to impact coastlines and contribute to marine risk monitoring.

## Overview
- Labeled dataset (~23k audio files, 200 ms each) from **Copernicus** and **CARI’MAM**.  
- Binary classification: *click* vs *no-click*.  
- Several modeling strategies were tested: **MLP**, **1D CNN**, and **2D CNN (Mel spectrograms)**.  
- Best performance achieved with a **hybrid 1D CNN (AUC = 0.94)** combining waveform and acoustic features.

## Workflow
1. **Preprocessing:** filtering (5–100 kHz), normalization, wavelet denoising  
2. **Feature engineering:** RMS power, spectral centroid, SNR, lags, seasonality  
3. **Modeling:** MLP, CNN1D, CNN2D  
4. **Evaluation:** AUC-ROC metric

## Tech Stack
Python, NumPy, SciPy, Librosa, TensorFlow, scikit-learn, Matplotlib


