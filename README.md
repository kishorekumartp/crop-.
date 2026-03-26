# 🌾 CropAI — AI-Enhanced Predictive Model for Crop Yields & Recommendation

> **Software-only AI system** using Machine Learning to predict crop yields and recommend the most suitable crop based on soil & climate conditions.

---

## 📁 Project Structure

```
crop-ai/
├── ml_pipeline.py      # Full ML pipeline (data gen, preprocessing, training, evaluation)
│── app.py              # Flask REST API
├── index.html          # Interactive web dashboard (open in browser — no server needed)
├── models/                 # Saved .pkl model files (generated after training)
├── data/                   # Place Kaggle datasets here
├── notebooks/              # Jupyter notebooks for EDA
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models
```bash
python ml_pipeline.py
```
This generates `models/preprocessor.pkl`, `models/crop_classifier.pkl`, `models/yield_regressor.pkl`

### 3. Start the Flask API
```bash
python app.py
# Running on http://localhost:5000
```

### 4. Open the dashboard
Simply open `frontend/index.html` in your browser.
- In **demo mode** (no API): Uses built-in JS inference engine
- With API running: Replace the JS inference with `fetch('http://localhost:5000/api/predict', ...)`

---

## 🧠 ML Models Used

| Task | Models | Best |
|------|--------|------|
| Crop Recommendation (Classification) | Random Forest, Decision Tree, SVM, Logistic Regression | **Random Forest (97.3% accuracy)** |
| Yield Prediction (Regression) | Random Forest, Decision Tree, SVR, Gradient Boosting | **Gradient Boosting (R²=0.961)** |

---

## 📊 Input Features

| Feature | Unit | Description |
|---------|------|-------------|
| N | kg/ha | Nitrogen content of soil |
| P | kg/ha | Phosphorus content of soil |
| K | kg/ha | Potassium content of soil |
| temperature | °C | Average temperature |
| humidity | % | Relative humidity |
| ph | 0–14 | Soil pH level |
| rainfall | mm/month | Monthly rainfall |

---

## 🎯 Outputs

- **Top-5 crop recommendations** with confidence percentages
- **Predicted yield** in tonnes/hectare
- **Agronomic hints** (soil deficiency warnings, pH alerts)

---

## 🌐 API Endpoints

```
GET  /api/health      → System status
GET  /api/crops       → List of 22 supported crops
POST /api/predict     → Crop recommendation + yield prediction
GET  /api/model-info  → Model performance metrics
```

### POST /api/predict — Example
```json
{
  "N": 90, "P": 42, "K": 43,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9
}
```

### Response
```json
{
  "recommendations": [
    {"rank": 1, "crop": "Rice", "confidence": 98.2},
    ...
  ],
  "predicted_yield": 3.84,
  "yield_unit": "tonnes / hectare",
  "hints": [{"type": "success", "message": "Soil conditions look healthy!"}]
}
```

---

## 📈 Model Performance

### Classification (Crop Recommendation)
| Model | Accuracy | F1 Score | CV Score |
|-------|----------|----------|----------|
| **Random Forest** | **97.3%** | **97.4%** | **97.1%** |
| SVM (RBF) | 93.4% | 93.5% | 93.0% |
| Decision Tree | 86.0% | 86.1% | 84.9% |
| Logistic Regression | 78.1% | 78.2% | 77.6% |

### Regression (Yield Prediction)
| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| **Gradient Boosting** | **0.9612** | **0.842** | **0.611** |
| Random Forest | 0.9478 | 0.961 | 0.703 |
| SVR | 0.8813 | 1.452 | 1.108 |
| Decision Tree | 0.8344 | 1.722 | 1.291 |

---

## 🔮 Future Enhancements

- Real-time weather API integration (OpenWeatherMap)
- Mobile application (React Native)
- Satellite/NDVI remote sensing data
- IoT soil sensor integration
- Regional language support

---

## 🛠️ Tech Stack

- **Python** 3.11
- **scikit-learn** — ML algorithms
- **Pandas / NumPy** — Data processing
- **Flask** — REST API
- **Matplotlib / Seaborn** — Visualization
- **HTML5 / CSS3 / JavaScript** — Frontend

---

*Project: AI-Enhanced Predictive Model for Optimizing Crop Yields and Data-Driven Crop Recommendation*
