# PISCI CARE - BACKEND API
# Water Quality Prediction + MongoDB Integration
# + Disease Detection + Feeding Prediction

from dotenv import load_dotenv
load_dotenv()
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from datetime import datetime
import pymongo

# ============================================
# PYTORCH IMPORTS FOR DISEASE MODEL
# ============================================
try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not installed. Disease detection will be disabled.")
    TORCH_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ============================================
# LOAD YOUR TRAINED MODELS
# ============================================
print("Loading models...")

# Path to your models
model_path = os.path.join('models', '4species_fixed_model.pkl')
encoder_path = os.path.join('models', '4species_fixed_encoder.pkl')
scaler_path = os.path.join('models', '4species_fixed_scaler.pkl')

# Load water quality models
water_model = joblib.load(model_path)
water_encoder = joblib.load(encoder_path)
water_scaler = joblib.load(scaler_path)

print("✅ Water quality model loaded successfully!")
print(f"   Species: {list(water_encoder.classes_)}")

# ============================================
# LOAD DISEASE MODEL (Complete .pth file)
# ============================================
disease_model = None
disease_model_loaded = False

try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    
    model_path = os.path.join('models', 'fish_disease_complete.pth')
    
    if os.path.exists(model_path):
        # Load complete model
        disease_model = torch.load(model_path, map_location='cpu', weights_only=False)
        disease_model.eval()
        disease_model_loaded = True
        print("✅ Disease detection model loaded successfully!")
    else:
        print(f"⚠️ Disease model not found at: {model_path}")
        
except Exception as e:
    print(f"⚠️ Disease model loading failed: {e}")

# Load feeding prediction model
feeding_model = None
feeding_model_loaded = False

feeding_model_path = os.path.join('models', 'feeding_prediction_model.pkl')
if os.path.exists(feeding_model_path):
    try:
        feeding_model = joblib.load(feeding_model_path)
        feeding_model_loaded = True
        print("✅ Feeding prediction model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Feeding model loading failed: {e}")
else:
    print(f"⚠️ Feeding model file not found at: {feeding_model_path}")

# Load species encoder
species_encoder = None
species_encoder_loaded = False

species_encoder_path = os.path.join('models', 'species_encoder.pkl')
if os.path.exists(species_encoder_path):
    try:
        species_encoder = joblib.load(species_encoder_path)
        species_encoder_loaded = True
        print("✅ Species encoder loaded successfully!")
    except Exception as e:
        print(f"⚠️ Species encoder loading failed: {e}")

# ============================================
# MONGODB CONNECTION
# ============================================

MONGO_URI = os.getenv('MONGO_URI')
try:
    mongo_client = pymongo.MongoClient(MONGO_URI)
    db = mongo_client['PisciCareDB']
    sensors_collection = db['sensordata']
    alerts_collection = db['alerts']
    history_collection = db['sensordata']
    print("✅ MongoDB Connected Successfully!")
    print(f"   Collections: {db.list_collection_names()}")
except Exception as e:
    print(f"❌ MongoDB Connection Failed: {e}")
    db = None

# Temporary storage for sensor data (fallback if MongoDB fails)
sensor_data_store = {
    'latest': {'ph': 7.2, 'temperature': 28.0, 'turbidity': 15.0},
    'history': []
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_treatment_recommendation(disease):
    """Get treatment recommendation based on disease"""
    recommendations = {
        "Healthy": "No treatment needed. Maintain good water quality.",
        "Bacterial Infection": "Use antibiotics. Consult a veterinarian for specific medication. Improve water quality.",
        "Fungal Infection": "Salt bath treatment (3% for 5-10 minutes). Improve water quality and reduce stress.",
        "Parasitic Infection": "Use anti-parasitic medication. Quarantine infected fish. Clean pond thoroughly.",
        "Viral Infection": "No specific treatment. Focus on water quality and fish immunity. Remove infected fish.",
        "Unknown": "Consult a fish health expert for proper diagnosis."
    }
    return recommendations.get(disease, recommendations["Unknown"])

def encode_species(species_name):
    """Convert species name to encoded value for feeding model"""
    if species_encoder_loaded and species_encoder is not None:
        try:
            return species_encoder.transform([species_name])[0]
        except:
            pass
    
    # Default mapping (if encoder not available)
    species_mapping = {
        "silver": 0,
        "silver fish": 0,
        "rohu": 1,
        "mahseer": 2,
        "sole": 3
    }
    return species_mapping.get(species_name.lower(), 0)

# ============================================
# ENDPOINT 1: Receive sensor data from Raspberry Pi
# ============================================
@app.route('/api/sensors', methods=['POST'])
def receive_sensor_data():
    """Raspberry Pi sends data here"""
    try:
        data = request.json
        sensor_data_store['latest'] = data
        
        # Save to history (temporary storage)
        sensor_data_store['history'].append(data)
        if len(sensor_data_store['history']) > 100:
            sensor_data_store['history'] = sensor_data_store['history'][-100:]
        
        # Save to MongoDB if connected
        if db is not None:
            data['timestamp'] = datetime.now()
            sensors_collection.insert_one(data)
            print(f"✅ Data saved to MongoDB: pH={data['ph']}, Temp={data['temperature']}°C, Turb={data['turbidity']}NTU")
        else:
            print(f"✅ Data received (MongoDB not connected): pH={data['ph']}, Temp={data['temperature']}°C, Turb={data['turbidity']}NTU")
        
        return jsonify({
            "status": "success", 
            "message": "Data received", 
            "data": data
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ============================================
# ENDPOINT 2: Get current sensor data
# ============================================
@app.route('/api/sensors/current', methods=['GET'])
def get_current_sensors():
    """Frontend calls this to get latest readings"""
    if sensor_data_store.get('latest'):
        return jsonify(sensor_data_store['latest'])
    else:
        return jsonify({
            "ph": 7.2,
            "temperature": 28.0,
            "turbidity": 15.0,
            "message": "No data from Raspberry Pi yet. Showing sample values."
        })

# ============================================
# ENDPOINT 3: Get sensor history from MongoDB
# ============================================
@app.route('/api/sensors/history', methods=['GET'])
def get_sensor_history():
    """Get last 100 sensor readings"""
    try:
        if db is not None:
            history = list(sensors_collection.find({}, {'_id': 0}).sort('timestamp', -1).limit(100))
            return jsonify(history)
        else:
            return jsonify(sensor_data_store.get('history', []))
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ============================================
# ENDPOINT 4: Water Quality Prediction (YOUR MODEL)
# ============================================
@app.route('/api/predict/water', methods=['POST'])
def predict_water():
    """Predict suitable fish based on water parameters"""
    try:
        data = request.json
        ph = float(data['ph'])
        temp = float(data['temperature'])
        turb = float(data['turbidity'])
        
        # Prepare input
        input_data = np.array([[ph, temp, turb]])
        input_scaled = water_scaler.transform(input_data)
        
        # Predict
        prediction = water_model.predict(input_scaled)
        fish = water_encoder.inverse_transform(prediction)[0]
        
        # Get confidence and probabilities
        probabilities = water_model.predict_proba(input_scaled)[0]
        confidence = np.max(probabilities) * 100
        
        # Create probability dictionary
        prob_dict = {}
        for i, species in enumerate(water_encoder.classes_):
            prob_dict[species] = round(probabilities[i] * 100, 1)
        
        # Save prediction to MongoDB (optional)
        if db is not None:
            prediction_data = {
                'timestamp': datetime.now(),
                'ph': ph,
                'temperature': temp,
                'turbidity': turb,
                'predicted_fish': fish,
                'confidence': confidence
            }
            history_collection.insert_one(prediction_data)
        
        return jsonify({
            "status": "success",
            "suitable_fish": fish,
            "confidence": round(confidence, 1),
            "probabilities": prob_dict,
            "input": {
                "ph": ph,
                "temperature": temp,
                "turbidity": turb
            }
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# =============================
# ENDPOINT 5: Disease Detection 
# =============================
@app.route('/api/predict/disease', methods=['POST'])
def predict_disease():
    """Detect disease from fish image"""
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image uploaded"}), 400
        
        file = request.files['image']
        
        if not disease_model_loaded:
            return jsonify({
                "status": "warning",
                "disease": "Unknown",
                "confidence": 0,
                "recommendation": "Model not loaded. Please check server logs."
            }), 200
        
        # Process image
        from PIL import Image
        import torchvision.transforms as transforms
        
        image = Image.open(file.stream).convert('RGB')
        
        # Standard preprocessing 
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = disease_model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item() * 100
        
        # Class mapping (0: Diseased, 1: Healthy )
        if predicted_class == 0:
            disease = "Diseased"
            recommendation = "Isolate fish immediately. Consult veterinarian. Check water quality."
        else:
            disease = "Healthy"
            recommendation = "No disease detected. Maintain current water quality and feeding."
        
        return jsonify({
            "status": "success",
            "disease": disease,
            "confidence": round(confidence_score, 2),
            "recommendation": recommendation,
            "probabilities": {
                "Diseased": round(probabilities[0][0].item() * 100, 2),
                "Healthy": round(probabilities[0][1].item() * 100, 2)
            }
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ============================================
# ENDPOINT 6: Feeding Prediction 
# ============================================
@app.route('/api/predict/feeding', methods=['POST'])
def predict_feeding():
    """Predict feeding amount based on water conditions"""
    try:
        data = request.json
        
        temperature = float(data.get('temperature', 28))
        ph = float(data.get('ph', 7.2))
        turbidity = float(data.get('turbidity', 15))
        species_name = str(data.get('fish_species', 'silver'))
        fish_weight = float(data.get('fish_weight', 100))
        
        # Simple feeding calculation
        if temperature > 28:
            feeding_percent = 2.0
        elif temperature < 22:
            feeding_percent = 1.5
        else:
            feeding_percent = 2.5
        
        # Species adjustment
        if species_name.lower() == 'silver':
            feeding_percent = feeding_percent * 1.0
        elif species_name.lower() == 'rohu':
            feeding_percent = feeding_percent * 1.1
        elif species_name.lower() == 'mahseer':
            feeding_percent = feeding_percent * 0.9
        elif species_name.lower() == 'sole':
            feeding_percent = feeding_percent * 0.8
        
        morning_feed = round(feeding_percent * 0.6, 2)
        evening_feed = round(feeding_percent * 0.4, 2)
        
        return jsonify({
            "status": "success",
            "feeding_amount_percent": float(round(feeding_percent, 2)),
            "feeding_schedule": "2 times per day",
            "morning_feed": f"{morning_feed}% of body weight",
            "evening_feed": f"{evening_feed}% of body weight",
            "total_daily_feed_g": float(round((feeding_percent / 100) * fish_weight, 2)),
            "recommendation": "Monitor fish behavior after feeding"
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ============================================
# ENDPOINT 8: Health check
# ============================================
@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({
        "status": "healthy",
        "models": {
            "water_quality": True,
            "disease_detection": disease_model_loaded,
            "feeding_prediction": feeding_model_loaded
        },
        "species": list(water_encoder.classes_),
        "mongodb_connected": db is not None,
        "timestamp": datetime.now().isoformat()
    })

# ============================================
# ENDPOINT 9: Get alerts (from MongoDB)
# ============================================
@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts from MongoDB"""
    try:
        if db is not None:
            alerts = list(alerts_collection.find({}, {'_id': 0}).sort('timestamp', -1).limit(50))
            return jsonify(alerts)
        else:
            return jsonify([])
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ============================================
# ENDPOINT 10: Save alert (called when threshold exceeded)
# ============================================
@app.route('/api/alerts', methods=['POST'])
def save_alert():
    """Save alert to MongoDB"""
    try:
        data = request.json
        if db is not None:
            data['timestamp'] = datetime.now()
            data['seen'] = False
            alerts_collection.insert_one(data)
            return jsonify({"status": "success", "message": "Alert saved"})
        else:
            return jsonify({"status": "warning", "message": "MongoDB not connected"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ============================================
# RUN THE APP
# ============================================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🐟 PISCI CARE BACKEND API")
    print(f"✅ Water Quality Model: {list(water_encoder.classes_)}")
    print(f"✅ Disease Detection Model: {'Loaded' if disease_model_loaded else 'Not loaded'}")
    print(f"✅ Feeding Prediction Model: {'Loaded' if feeding_model_loaded else 'Not loaded'}")
    print(f"✅ MongoDB: {'Connected' if db is not None else 'Not connected'}")

    print("🚀 Starting server at http://localhost:5000")
    print("\n📡 API Endpoints:")
    print("   POST   /api/sensors              - Receive sensor data from Pi")
    print("   GET    /api/sensors/current      - Get current sensor readings")
    print("   GET    /api/sensors/history      - Get sensor history")
    print("   POST   /api/predict/water        - Water quality prediction")
    print("   POST   /api/predict/disease      - Disease detection (upload image)")
    print("   POST   /api/predict/feeding      - Feeding prediction")
    print("   GET    /api/health               - Health check")
    print("   GET    /api/alerts               - Get alerts")
    print("   POST   /api/alerts               - Save alert")
    
    app.run(host='0.0.0.0', port=5000, debug=True)