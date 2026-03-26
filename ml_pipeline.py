"""
AI-Enhanced Predictive Model for Optimizing Crop Yields
and Data-Driven Crop Recommendation
========================================================
ML Pipeline: Data preprocessing, model training, evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, mean_squared_error, r2_score,
                              classification_report, confusion_matrix)
import joblib
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
#    (mirrors real Kaggle crop recommendation dataset structure)
# ─────────────────────────────────────────────

CROPS = [
    'Rice', 'Maize', 'Chickpea', 'KidneyBeans', 'PigeonPeas',
    'MothBeans', 'MungBean', 'Blackgram', 'Lentil', 'Pomegranate',
    'Banana', 'Mango', 'Grapes', 'Watermelon', 'Muskmelon',
    'Apple', 'Orange', 'Papaya', 'Coconut', 'Cotton',
    'Jute', 'Coffee'
]

CROP_PROFILES = {
    'Rice':         dict(N=(60,100),  P=(40,60),  K=(30,50),  temp=(20,27), hum=(80,95), ph=(5.5,7.0), rain=(180,250), yield_base=3.5),
    'Maize':        dict(N=(70,100),  P=(50,70),  K=(40,60),  temp=(18,27), hum=(60,80), ph=(5.8,7.5), rain=(50,100),  yield_base=4.2),
    'Chickpea':     dict(N=(30,50),   P=(60,80),  K=(50,70),  temp=(15,25), hum=(14,22), ph=(6.0,8.0), rain=(60,100),  yield_base=1.8),
    'KidneyBeans':  dict(N=(15,25),   P=(60,80),  K=(15,25),  temp=(15,27), hum=(18,25), ph=(5.5,7.5), rain=(80,120),  yield_base=1.5),
    'PigeonPeas':   dict(N=(15,25),   P=(60,80),  K=(15,25),  temp=(18,28), hum=(30,40), ph=(5.5,7.0), rain=(60,100),  yield_base=1.2),
    'MothBeans':    dict(N=(15,25),   P=(35,55),  K=(35,55),  temp=(25,38), hum=(25,35), ph=(3.5,6.5), rain=(30,60),   yield_base=0.8),
    'MungBean':     dict(N=(15,25),   P=(35,60),  K=(15,25),  temp=(25,35), hum=(80,92), ph=(6.2,7.2), rain=(60,120),  yield_base=1.0),
    'Blackgram':    dict(N=(35,55),   P=(55,75),  K=(15,25),  temp=(25,35), hum=(60,80), ph=(6.0,7.5), rain=(65,100),  yield_base=1.1),
    'Lentil':       dict(N=(15,25),   P=(60,80),  K=(15,25),  temp=(15,25), hum=(60,75), ph=(6.0,7.5), rain=(35,65),   yield_base=1.3),
    'Pomegranate':  dict(N=(15,25),   P=(15,25),  K=(15,25),  temp=(18,30), hum=(90,95), ph=(6.5,7.5), rain=(105,125), yield_base=8.0),
    'Banana':       dict(N=(90,120),  P=(75,100), K=(45,60),  temp=(25,35), hum=(75,85), ph=(5.5,6.5), rain=(100,150), yield_base=25.0),
    'Mango':        dict(N=(15,25),   P=(15,25),  K=(15,25),  temp=(24,30), hum=(90,95), ph=(5.5,7.5), rain=(90,110),  yield_base=7.5),
    'Grapes':       dict(N=(15,25),   P=(15,25),  K=(15,25),  temp=(8,17),  hum=(80,82), ph=(5.5,6.5), rain=(60,80),   yield_base=12.0),
    'Watermelon':   dict(N=(95,115),  P=(8,12),   K=(48,52),  temp=(24,30), hum=(85,95), ph=(6.0,7.0), rain=(50,60),   yield_base=20.0),
    'Muskmelon':    dict(N=(95,115),  P=(8,12),   K=(48,52),  temp=(28,32), hum=(92,95), ph=(6.0,6.7), rain=(22,28),   yield_base=15.0),
    'Apple':        dict(N=(0,20),    P=(120,145),K=(195,210),temp=(21,24), hum=(90,95), ph=(5.5,6.5), rain=(110,125), yield_base=10.0),
    'Orange':       dict(N=(0,20),    P=(5,15),   K=(5,15),   temp=(10,15), hum=(90,95), ph=(6.0,7.5), rain=(100,120), yield_base=12.0),
    'Papaya':       dict(N=(45,60),   P=(45,60),  K=(43,55),  temp=(25,35), hum=(90,95), ph=(6.5,7.5), rain=(140,160), yield_base=30.0),
    'Coconut':      dict(N=(3,10),    P=(3,10),   K=(3,10),   temp=(27,30), hum=(90,95), ph=(5.0,8.0), rain=(140,160), yield_base=50.0),
    'Cotton':       dict(N=(100,130), P=(17,25),  K=(17,25),  temp=(21,26), hum=(75,85), ph=(5.8,8.0), rain=(60,100),  yield_base=2.0),
    'Jute':         dict(N=(70,85),   P=(45,55),  K=(40,50),  temp=(24,37), hum=(75,90), ph=(6.0,7.0), rain=(150,200), yield_base=2.5),
    'Coffee':       dict(N=(95,110),  P=(27,38),  K=(28,40),  temp=(25,30), hum=(55,65), ph=(6.0,6.8), rain=(145,165), yield_base=1.5),
}

def generate_dataset(n_samples=2200, seed=42):
    """Generate synthetic crop dataset based on realistic agronomic profiles."""
    np.random.seed(seed)
    records = []
    per_crop = n_samples // len(CROPS)

    for crop, p in CROP_PROFILES.items():
        for _ in range(per_crop):
            N    = np.random.uniform(*p['N'])
            P    = np.random.uniform(*p['P'])
            K    = np.random.uniform(*p['K'])
            temp = np.random.uniform(*p['temp'])
            hum  = np.random.uniform(*p['hum'])
            ph   = np.random.uniform(*p['ph'])
            rain = np.random.uniform(*p['rain'])

            # Yield = base × soil_factor × climate_factor + noise
            soil_score    = (N/100 + P/80 + K/60) / 3
            climate_score = (rain/150 + hum/90 + (1 - abs(temp-25)/25)) / 3
            yld = p['yield_base'] * (0.7 + 0.6*soil_score) * (0.7 + 0.6*climate_score)
            yld += np.random.normal(0, p['yield_base']*0.08)
            yld = max(0.1, round(yld, 2))

            records.append([N, P, K, temp, hum, ph, rain, crop, yld])

    df = pd.DataFrame(records, columns=['N','P','K','temperature','humidity','ph','rainfall','label','yield'])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────────

class DataPreprocessor:
    def __init__(self):
        self.scaler    = StandardScaler()
        self.le        = LabelEncoder()
        self.features  = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.fitted    = False

    def fit_transform(self, df):
        df = df.copy()
        # Handle missing values
        df[self.features] = df[self.features].fillna(df[self.features].median())
        # Remove outliers via IQR
        for col in self.features:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[df[col].between(Q1 - 3*IQR, Q3 + 3*IQR)]
        # Encode labels
        df['label_enc'] = self.le.fit_transform(df['label'])
        # Scale features
        df[self.features] = self.scaler.fit_transform(df[self.features])
        self.fitted = True
        return df

    def transform(self, X):
        return self.scaler.transform(X)

    def inverse_label(self, y):
        return self.le.inverse_transform(y)


# ─────────────────────────────────────────────
# 3. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────

class CropRecommendationModel:
    """Classification: predict most suitable crop."""

    MODELS = {
        'Random Forest':     RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
        'Decision Tree':     DecisionTreeClassifier(max_depth=10, random_state=42),
        'SVM':               SVC(kernel='rbf', C=10, probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    }

    def __init__(self):
        self.best_model = None
        self.best_name  = None
        self.results    = {}

    def train_evaluate(self, X_train, X_test, y_train, y_test):
        best_acc = 0
        for name, model in self.MODELS.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc  = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='weighted', zero_division=0)
            rec  = recall_score(y_test, preds, average='weighted', zero_division=0)
            f1   = f1_score(y_test, preds, average='weighted', zero_division=0)
            cv   = cross_val_score(model, X_train, y_train, cv=5).mean()
            self.results[name] = {'accuracy': acc, 'precision': prec,
                                  'recall': rec, 'f1': f1, 'cv_score': cv}
            if acc > best_acc:
                best_acc = acc
                self.best_model = model
                self.best_name  = name
        return self.results

    def predict(self, X):
        return self.best_model.predict(X)

    def predict_proba(self, X):
        return self.best_model.predict_proba(X)


class CropYieldModel:
    """Regression: predict crop yield (tonnes/hectare)."""

    MODELS = {
        'Random Forest':  RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
        'Decision Tree':  DecisionTreeRegressor(max_depth=10, random_state=42),
        'SVR':            SVR(kernel='rbf', C=10),
        'Gradient Boost': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    }

    def __init__(self):
        self.best_model = None
        self.best_name  = None
        self.results    = {}

    def train_evaluate(self, X_train, X_test, y_train, y_test):
        best_r2 = -999
        for name, model in self.MODELS.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse  = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            r2   = r2_score(y_test, preds)
            mae  = np.mean(np.abs(y_test - preds))
            self.results[name] = {'mse': mse, 'rmse': rmse, 'r2': r2, 'mae': mae}
            if r2 > best_r2:
                best_r2 = r2
                self.best_model = model
                self.best_name  = name
        return self.results

    def predict(self, X):
        return max(0.0, float(self.best_model.predict(X)[0]))


# ─────────────────────────────────────────────
# 4. MAIN TRAINING WORKFLOW
# ─────────────────────────────────────────────

def train_and_save():
    print("=" * 60)
    print("  CropAI — Training Pipeline")
    print("=" * 60)

    # Generate data
    print("\n[1/5] Generating dataset...")
    df = generate_dataset(2200)
    print(f"      Rows: {len(df)} | Crops: {df['label'].nunique()}")

    # Preprocess
    print("\n[2/5] Preprocessing...")
    prep = DataPreprocessor()
    df_proc = prep.fit_transform(df)

    features = prep.features
    X = df_proc[features].values
    y_cls = df_proc['label_enc'].values
    y_reg = df_proc['yield'].values

    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls)

    # Train classification
    print("\n[3/5] Training crop recommendation models...")
    clf = CropRecommendationModel()
    clf_results = clf.train_evaluate(X_tr, X_te, yc_tr, yc_te)
    for name, m in clf_results.items():
        print(f"      {name:20s}  Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}  CV={m['cv_score']:.3f}")
    print(f"  ✓  Best: {clf.best_name}")

    # Train regression
    print("\n[4/5] Training yield prediction models...")
    reg = CropYieldModel()
    reg_results = reg.train_evaluate(X_tr, X_te, yr_tr, yr_te)
    for name, m in reg_results.items():
        print(f"      {name:20s}  R²={m['r2']:.3f}  RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}")
    print(f"  ✓  Best: {reg.best_name}")

    # Save
    print("\n[5/5] Saving models...")
    import os
    save_dir = r"D:\New folder (2)\models"
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(prep, os.path.join(save_dir, 'preprocessor.pkl'))
    joblib.dump(clf,  os.path.join(save_dir, 'crop_classifier.pkl'))
    joblib.dump(reg,  os.path.join(save_dir, 'yield_regressor.pkl'))
    print(f"      Saved to {save_dir}")

    # Feature importance
    if hasattr(clf.best_model, 'feature_importances_'):
        imp = clf.best_model.feature_importances_
        print("\n  Feature Importances (Classification):")
        for f, i in sorted(zip(features, imp), key=lambda x: -x[1]):
            bar = '█' * int(i * 50)
            print(f"    {f:12s} {bar} {i:.3f}")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)
    return prep, clf, reg


def predict_single(prep, clf, reg, N, P, K, temperature, humidity, ph, rainfall, top_n=3):
    """Make prediction for a single input."""
    X_raw = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    X_scaled = prep.transform(X_raw)

    # Recommendation
    proba = clf.predict_proba(X_scaled)[0]
    top_idx = np.argsort(proba)[::-1][:top_n]
    recommendations = [
        {'crop': prep.le.inverse_transform([i])[0], 'confidence': round(float(proba[i])*100, 1)}
        for i in top_idx
    ]

    # Yield prediction
    predicted_yield = reg.predict(X_scaled)

    return {
        'top_recommendation': recommendations[0]['crop'],
        'confidence': recommendations[0]['confidence'],
        'all_recommendations': recommendations,
        'predicted_yield': round(predicted_yield, 2),
        'unit': 'tonnes/hectare'
    }


if __name__ == '__main__':
    prep, clf, reg = train_and_save()

    # Demo prediction
    print("\n--- Demo Prediction ---")
    result = predict_single(prep, clf, reg,
        N=90, P=42, K=43, temperature=20.8, humidity=82.0, ph=6.5, rainfall=202.9)
    print(f"  Top Crop:       {result['top_recommendation']} ({result['confidence']}%)")
    print(f"  Predicted Yield:{result['predicted_yield']} {result['unit']}")
    print(f"  All Top Picks:  {[r['crop'] for r in result['all_recommendations']]}")
