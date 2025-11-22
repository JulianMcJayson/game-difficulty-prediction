# Adaptive Difficulty System - Deep Learning Classification API

A PyTorch-based neural network that dynamically adjusts game difficulty based on player behavior and performance metrics.

## Overview

This system uses a binary classification model to determine whether game difficulty should be **increased** (class 1) or **decreased** (class 0) based on:
- Player failure count
- Player activity score (combination of movement, rotation, and action rates)

The model learns patterns from synthetic training data that simulates realistic player behavior scenarios.

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── api_model.py          # Pydantic request/response models
│   ├── dataset.py             # Synthetic data generation
│   ├── main.py                # FastAPI endpoints
│   ├── model.py               # PyTorch model architecture & training
│   └── service_model.py       # Prediction service & model loading
├── data/
│   └── classification_dataset.json   # Generated training data
├── model/
│   ├── adaptive_model.pth     # Trained model weights
│   └── x_scalar.gz            # StandardScaler for input normalization
└── README.md
```

## Core Logic

### Classification Rules
- **Class 0 (Decrease Difficulty)**: 
  - High fail count (>10 failures), OR
  - Low fail count with high activity (>0.35)
- **Class 1 (Increase Difficulty)**: 
  - Low fail count AND low activity

### Model Architecture
- Input: 2 features (log-transformed fail count, activity score)
- Deep neural network with PReLU/LeakyReLU activations
- Dropout layers (0.3) for regularization
- Output: Binary classification via BCEWithLogitsLoss
- Device: MPS (Apple Silicon) or CPU fallback

## Installation

```bash
pip install fastapi uvicorn torch scikit-learn pydantic joblib matplotlib seaborn numpy
```

## Usage

### 1. Generate Training Data
```bash
python -c "from app.dataset import *"
```
Generates 100,000 synthetic samples in `data/classification_dataset.json`

### 2. Train Model
```python
from app.service_model import train_and_test_caller
train_and_test_caller()
```
Trains for 10 epochs with early stopping, saves model to `model/adaptive_model.pth`

### 3. Run API Server
```bash
uvicorn app.main:app --reload
```

### 4. Make Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "fail": 7.0,
    "movement_rate": 0.6,
    "rotation_rate": 0.8,
    "action_rate": 0.7
  }'
```

Response:
```json
{
  "adaptive_difficulty": 0
}
```

## API Reference

### POST `/predict`
**Request Body:**
```typescript
{
  fail: float,           // Number of player failures
  movement_rate: float,  // Movement activity [0-1]
  rotation_rate: float,  // Rotation activity [0-1]
  action_rate: float     // Action activity [0-1]
}
```

**Response:**
```typescript
{
  adaptive_difficulty: int  // 0 = Decrease, 1 = Increase
}
```

## Training Details

- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=50)
- **Early Stopping**: 100 epochs patience
- **Batch Size**: 2048
- **Train/Test Split**: 80/20 with stratification
- **Preprocessing**: StandardScaler on log-transformed fail count and raw activity score

## Model Performance

The trained model outputs:
- Classification report with precision/recall/F1 for both classes
- Confusion matrix heatmap
- Training/validation loss curves
- Validation accuracy over epochs

Typical results: **~95%+ test accuracy** on balanced synthetic data.

## Key Features

- **Logarithmic fail transformation**: Handles wide range of failure counts
- **Gradient clipping**: Prevents exploding gradients (max_norm=1.0)
- **Device-agnostic**: Auto-detects MPS/CUDA/CPU
- **Production-ready**: FastAPI with proper validation and error handling

---

**Note**: This is a classification problem, not regression. The model predicts discrete difficulty adjustment classes rather than continuous values.
