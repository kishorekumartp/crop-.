"""
Flask REST API — CropAI Backend
================================
Endpoints:
  POST /api/predict   — crop recommendation + yield prediction
  GET  /api/crops     — list of all supported crops
  GET  /api/health    — health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import sys
import requests as req
sys.path.insert(0, r"D:\Crop AI")
from ml_pipeline import DataPreprocessor, CropRecommendationModel, CropYieldModel
app = Flask(__name__)
CORS(app)

# Load models on startup
MODEL_DIR = os.path.join(os.path.dirname(__file__),  'models')
try:
    prep = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
    clf  = joblib.load(os.path.join(MODEL_DIR, 'crop_classifier.pkl'))
    reg  = joblib.load(os.path.join(MODEL_DIR, 'yield_regressor.pkl'))
    MODELS_LOADED = True
    print("✓ Models loaded successfully")
except Exception as e:
    MODELS_LOADED = False
    print(f"✗ Could not load models: {e}")
from flask import send_file

@app.route('/')
def index():
    return send_file(r"D:\Crop AI\index.html")

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models_loaded': MODELS_LOADED})


@app.route('/api/crops', methods=['GET'])
def get_crops():
    crops = list(prep.le.classes_) if MODELS_LOADED else []
    return jsonify({'crops': crops})


@app.route('/api/predict', methods=['POST'])
def predict():
    if not MODELS_LOADED:
        return jsonify({'error': 'Models not loaded'}), 503

    data = request.get_json()
    required = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

    # Validate inputs
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        vals = [float(data[k]) for k in required]
    except ValueError as e:
        return jsonify({'error': f'Invalid value: {e}'}), 400

    # Validate ranges
    N, P, K, temp, hum, ph, rain = vals
    validations = [
        (0 <= N <= 200,    'N (Nitrogen) must be 0–200 kg/ha'),
        (0 <= P <= 200,    'P (Phosphorus) must be 0–200 kg/ha'),
        (0 <= K <= 250,    'K (Potassium) must be 0–250 kg/ha'),
        (-10 <= temp <= 55,'Temperature must be -10–55 °C'),
        (0 <= hum <= 100,  'Humidity must be 0–100 %'),
        (0 <= ph <= 14,    'pH must be 0–14'),
        (0 <= rain <= 600, 'Rainfall must be 0–600 mm'),
    ]
    for ok, msg in validations:
        if not ok:
            return jsonify({'error': msg}), 400

    # Predict
    X_raw = np.array([[N, P, K, temp, hum, ph, rain]])
    X_scaled = prep.transform(X_raw)

    # Top-5 crop recommendations
    proba = clf.predict_proba(X_scaled)[0]
    top_idx = np.argsort(proba)[::-1][:5]
    recommendations = [
        {
            'rank': i + 1,
            'crop': prep.le.inverse_transform([idx])[0],
            'confidence': round(float(proba[idx]) * 100, 1)
        }
        for i, idx in enumerate(top_idx)
    ]

    # Yield prediction
    predicted_yield = max(0.0, float(reg.predict(X_scaled)))

    # Soil health hints
    hints = []
    if N < 40:  hints.append({'type': 'warning', 'message': 'Low Nitrogen — consider urea or compost application'})
    if P < 30:  hints.append({'type': 'warning', 'message': 'Low Phosphorus — consider DAP fertilizer'})
    if K < 25:  hints.append({'type': 'warning', 'message': 'Low Potassium — consider MOP (Muriate of Potash)'})
    if ph < 5.5: hints.append({'type': 'info', 'message': 'Acidic soil — liming recommended'})
    if ph > 7.5: hints.append({'type': 'info', 'message': 'Alkaline soil — sulfur amendment may help'})
    if rain < 40: hints.append({'type': 'warning', 'message': 'Low rainfall — consider irrigation planning'})
    if not hints:
        hints.append({'type': 'success', 'message': 'Soil and climate conditions look healthy!'})

    return jsonify({
        'status': 'success',
        'input': {k: v for k, v in zip(required, vals)},
        'recommendations': recommendations,
        'predicted_yield': round(predicted_yield, 2),
        'yield_unit': 'tonnes / hectare',
        'hints': hints
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    if not MODELS_LOADED:
        return jsonify({'error': 'Models not loaded'}), 503
    return jsonify({
        'classifier': {
            'best_model': clf.best_name,
            'results': clf.results
        },
        'regressor': {
            'best_model': reg.best_name,
            'results': reg.results
        }
    })


# ─────────────────────────────────────────────
# WEATHER API — OpenWeatherMap
# ─────────────────────────────────────────────

WEATHER_API_KEY = "put your weathers token"

# ── Hybrid Rainfall Lookup Table ──
# Annual average rainfall (mm/year) for Indian cities — IMD historical normals
# Used instead of unreliable 1h API reading for crop prediction accuracy
CITY_ANNUAL_RAINFALL_MM = {
    # Tamil Nadu
    'chennai':1400,'coimbatore':700,'madurai':850,'trichy':843,
    'salem':925,'tirunelveli':750,'vellore':1000,'thanjavur':950,
    'tiruppur':700,'erode':750,'dindigul':850,'kanchipuram':1200,
    'cuddalore':1200,'nagapattinam':1300,'kumbakonam':950,
    # Karnataka
    'bangalore':970,'bengaluru':970,'mysore':780,'mysuru':780,
    'hubli':750,'mangalore':3800,'mangaluru':3800,'belgaum':950,
    'davangere':700,'bellary':550,'shimoga':1800,'tumkur':800,
    # Kerala
    'thiruvananthapuram':1750,'kochi':3050,'kozhikode':2950,
    'thrissur':3050,'kollam':1700,'palakkad':2100,'kannur':3200,
    'malappuram':2800,'alappuzha':2800,
    # Andhra Pradesh & Telangana
    'hyderabad':812,'visakhapatnam':1050,'vijayawada':1000,
    'tirupati':950,'warangal':950,'guntur':950,'nellore':1000,
    'rajahmundry':1100,'karimnagar':950,'nizamabad':900,
    # Maharashtra
    'mumbai':2400,'pune':720,'nagpur':1200,'nashik':700,
    'aurangabad':700,'solapur':550,'kolhapur':1000,'thane':2500,
    'navi mumbai':2400,'amravati':850,
    # Delhi & NCR
    'delhi':790,'new delhi':790,'gurgaon':750,'noida':800,'faridabad':750,
    # Uttar Pradesh
    'lucknow':900,'kanpur':850,'agra':700,'varanasi':1000,
    'allahabad':1000,'prayagraj':1000,'meerut':800,'ghaziabad':800,
    # West Bengal
    'kolkata':1600,'calcutta':1600,'howrah':1600,'asansol':1300,
    'siliguri':2800,'durgapur':1300,
    # Rajasthan
    'jaipur':650,'jodhpur':370,'udaipur':650,'kota':750,'ajmer':500,
    # Gujarat
    'ahmedabad':800,'surat':1200,'vadodara':900,'rajkot':650,
    'bhavnagar':600,'jamnagar':550,
    # Madhya Pradesh
    'bhopal':1150,'indore':950,'gwalior':750,'jabalpur':1400,
    # Punjab & Haryana
    'chandigarh':1100,'ludhiana':750,'amritsar':700,'jalandhar':700,
    'ambala':900,'rohtak':700,
    # Odisha
    'bhubaneswar':1500,'cuttack':1500,'rourkela':1600,'sambalpur':1500,
    # Assam & Northeast
    'guwahati':1650,'shillong':2400,'imphal':1500,'agartala':2100,
    # Himachal & Uttarakhand
    'shimla':1575,'dehradun':2100,'haridwar':1800,'mussoorie':2000,
    # Bihar & Jharkhand
    'patna':1100,'ranchi':1400,'jamshedpur':1400,'gaya':1000,
    # Goa
    'panaji':2900,'goa':2900,'margao':3000,
    # Default fallback for unknown cities
    'default':1100
}

def get_annual_rainfall(city_name):
    """Lookup annual rainfall — tries full name, then first word, then default."""
    key = city_name.lower().strip()
    if key in CITY_ANNUAL_RAINFALL_MM:
        return CITY_ANNUAL_RAINFALL_MM[key], True
    # Try first word (e.g. "New Delhi" -> "new")
    first_word = key.split()[0]
    if first_word in CITY_ANNUAL_RAINFALL_MM:
        return CITY_ANNUAL_RAINFALL_MM[first_word], True
    return CITY_ANNUAL_RAINFALL_MM['default'], False

@app.route('/api/weather', methods=['GET'])
def get_weather():
    city = request.args.get('city', 'Chennai')
    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
        )
        response = req.get(url, timeout=5)
        data     = response.json()

        if response.status_code != 200:
            return jsonify({'error': f"City not found: {data.get('message','unknown')}"}), 404

        # REAL TIME from API
        temperature   = round(data['main']['temp'], 1)
        humidity      = round(data['main']['humidity'], 1)
        country       = data['sys']['country']

        # HYBRID — use lookup table for annual rainfall (far more accurate for crops)
        annual_mm, found = get_annual_rainfall(city)

        return jsonify({
            'city'              : city,
            'country'           : country,
            'temperature'       : temperature,       # real time ✅
            'humidity'          : humidity,           # real time ✅
            'rainfall'          : annual_mm,          # historical annual average ✅
            'rainfall_type'     : 'annual',
            'rainfall_source'   : 'IMD historical average' if found else 'India average (city not in database)',
            'city_in_database'  : found,
            'description'       : data['weather'][0]['description'].title(),
            'icon'              : data['weather'][0]['icon'],
            'auto_filled'       : ['temperature', 'humidity', 'rainfall'],
            'note'              : 'Temperature & Humidity are live. Rainfall is IMD annual historical average — best for crop prediction.'
        })

    except req.exceptions.Timeout:
        return jsonify({'error': 'Weather API timed out'}), 504
    except Exception as e:
        return jsonify({'error': f'Weather fetch failed: {str(e)}'}), 500


# ─────────────────────────────────────────────
# PROFIT ESTIMATOR — All values in INR (Rs.)
# ─────────────────────────────────────────────

CROP_PRICES_INR = {
    'Rice':21830, 'Maize':19620, 'Chickpea':54400,
    'KidneyBeans':60000, 'PigeonPeas':66000, 'MothBeans':66450,
    'MungBean':77550, 'Blackgram':66000, 'Lentil':55000,
    'Pomegranate':80000, 'Banana':20000, 'Mango':40000,
    'Grapes':60000, 'Watermelon':15000, 'Muskmelon':20000,
    'Apple':100000, 'Orange':30000, 'Papaya':15000,
    'Coconut':30000, 'Cotton':66200, 'Jute':45000, 'Coffee':110000
}

FARMING_COSTS_INR = {
    'Rice':25000, 'Maize':20000, 'Chickpea':18000,
    'KidneyBeans':16000, 'PigeonPeas':15000, 'MothBeans':14000,
    'MungBean':16000, 'Blackgram':15000, 'Lentil':17000,
    'Pomegranate':45000, 'Banana':50000, 'Mango':40000,
    'Grapes':60000, 'Watermelon':30000, 'Muskmelon':28000,
    'Apple':55000, 'Orange':40000, 'Papaya':35000,
    'Coconut':20000, 'Cotton':28000, 'Jute':22000, 'Coffee':35000
}

@app.route('/api/profit', methods=['POST'])
def estimate_profit():
    data = request.get_json()
    required = ['crop', 'yield_tonnes', 'land_hectares']

    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    crop          = data['crop']
    yield_tonnes  = float(data['yield_tonnes'])
    land_hectares = float(data['land_hectares'])

    if crop not in CROP_PRICES_INR:
        return jsonify({'error': f'Crop "{crop}" not found. Check /api/crops for valid names'}), 404

    if yield_tonnes <= 0 or land_hectares <= 0:
        return jsonify({'error': 'yield_tonnes and land_hectares must be greater than 0'}), 400

    # Rs. per tonne (price per quintal x 10)
    price_per_tonne    = CROP_PRICES_INR[crop]
    total_yield        = round(yield_tonnes * land_hectares, 2)
    total_revenue      = round(total_yield * price_per_tonne, 2)
    total_cost         = round(FARMING_COSTS_INR[crop] * land_hectares, 2)
    net_profit         = round(total_revenue - total_cost, 2)
    profit_margin      = round((net_profit / total_revenue * 100), 1) if total_revenue > 0 else 0
    roi                = round((net_profit / total_cost * 100), 1)    if total_cost > 0    else 0

    return jsonify({
        'crop'              : crop,
        'land_hectares'     : land_hectares,
        'total_yield_tonnes': total_yield,
        'price_per_tonne'   : price_per_tonne,
        'total_revenue_inr' : total_revenue,
        'total_cost_inr'    : total_cost,
        'net_profit_inr'    : net_profit,
        'profit_margin_pct' : profit_margin,
        'roi_percent'       : roi,
        'currency'          : 'INR (Rs.)',
        'verdict'           : 'Profitable' if net_profit > 0 else 'Loss'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
