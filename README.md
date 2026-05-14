# Real-Time Sign Language Recognition using MediaPipe and Machine Learning

A real-time hand gesture recognition system that detects sign language gestures using a webcam, extracts hand landmarks using :contentReference[oaicite:0]{index=0}, and classifies gestures using a :contentReference[oaicite:1]{index=1} model built with :contentReference[oaicite:2]{index=2}.

The system currently recognizes multiple one-handed static signs in real time with confidence scoring.

---

## Features

- Real-time webcam-based gesture recognition
- 21 hand landmark detection
- Custom dataset creation pipeline
- Machine learning based classification
- Confidence-based prediction filtering
- Live hand skeleton visualization
- Multiple sign vocabulary support
- Easy scalability for adding new signs

---

## Current Supported Signs

- HELLO
- YES
- NO
- OK
- HELP
- LOVE

---

## Demo Pipeline

```text
Webcam
   ↓
Hand Detection
   ↓
21 Landmark Extraction
   ↓
Feature Vector Creation
   ↓
Random Forest Classifier
   ↓
Real-Time Sign Prediction
```

---

## Tech Stack

- Python
- :contentReference[oaicite:3]{index=3}
- :contentReference[oaicite:4]{index=4}
- :contentReference[oaicite:5]{index=5}
- :contentReference[oaicite:6]{index=6}
- :contentReference[oaicite:7]{index=7}
- Joblib

---

## Project Structure

```bash
sign-language-recognition/
│
├── data/
│   └── data.csv
│
├── objects/
│   ├── model.pkl
│   └── label_encoder.pkl
│
├── collect.py
├── model.py
├── sign_lang.py
├── vocabular.txt
├── requirements.txt
└── README.md
```

---

## Dataset Creation

Custom dataset created using live webcam capture.

Each sample contains:

```text
x1,y1,x2,y2,x3,y3...x21,y21,sign
```

- 21 hand landmarks
- 42 features per sample
- Manually labeled

---

## Model Training

The classifier is trained using:

- Random Forest Classifier
- 5-Fold Cross Validation
- Hyperparameter tuning using RandomizedSearchCV

Achieved:

```text
98%+ accuracy
```

on validation data.

---

## Installation

Clone the repository:

```bash
git clone <your_repo_url>
cd sign-language-recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Collect Data

```bash
python collect.py
```

### 2. Train Model

```bash
python model.py
```

### 3. Run Real-Time Recognition

```bash
python sign_lang.py
```

---

## Future Improvements

- Dynamic sign recognition
- Sentence generation
- Text-to-speech conversion
- Two-hand gesture support
- GUI deployment
- Cloud deployment

---

## Applications

- Assistive communication
- Human-computer interaction
- Smart interfaces
- Accessibility technology
- Educational tools

---

## Author

Uvan Rishee

---