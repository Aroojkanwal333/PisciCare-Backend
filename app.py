# PISCI CARE - BACKEND API
# Water Quality Prediction + MongoDB Integration
from dotenv import load_dotenv
load_dotenv()
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from datetime import datetime
import pymongo

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# LOAD YOUR TRAINED MODELS
print("Loading models...")

# Path to your models
model_path = os.path.join('models', '4species_fixed_model.pkl')
encoder_path = os.path.join('models', '4species_fixed_encoder.pkl')
scaler_path = os.path.join('models', '4species_fixed_scaler.pkl')

# Load models
water_model = joblib.load(model_path)
water_encoder = joblib.load(encoder_path)
water_scaler = joblib.load(scaler_path)

print("✅ Models loaded successfully!")
print(f"   Species: {list(water_encoder.classes_)}")

# MONGODB CONNECTION
# REPLACE WITH YOUR FRIEND'S CREDENTIALS
MONGO_URI = os.getenv('MONGO_URI')
try:
    mongo_client = pymongo.MongoClient(MONGO_URI)
    db = mongo_client['PisciCareDB']
    sensors_collection = db['sensordata']
    alerts_collection = db['alerts']
    history_collection = db['sensordata']  # or create separate collection
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

# ENDPOINT 1: Receive sensor data from Raspberry Pi
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

# ENDPOINT 2: Get current sensor data
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

# ENDPOINT 3: Get sensor history from MongoDB
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

# ENDPOINT 4: Water Quality Prediction (YOUR MODEL)
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

# ENDPOINT 5: Chatbot (Simple rule-based)
@app.route('/api/chat', methods=['POST'])
def chat():
    """Simple rule-based chatbot"""
    try:
        user_message = request.json.get('message', '').lower()
        
        # Simple response rules
        if 'ph' in user_message:
            response = "Ideal pH ranges:\n• Silver Fish: 6.8-7.5\n• Rohu: 6.5-8.0\n• Mahseer: 7.0-8.0\n• Sole: 7.2-8.2"
        elif 'temperature' in user_message or 'temp' in user_message:
            response = "Ideal temperature ranges:\n• Silver Fish: 25-28°C\n• Rohu: 26-30°C\n• Mahseer: 22-26°C\n• Sole: 20-24°C"
        elif 'feeding' in user_message or 'feed' in user_message:
            response = "Feeding guidelines:\n• Feed 2-3 times daily\n• Summer: 3-4% of body weight\n• Winter: 1-2% of body weight\n• Only feed what fish eat in 5 minutes"
        elif 'disease' in user_message:
            response = "Common fish diseases:\n• Bacterial infections\n• White Spot (Ich)\n• Fungal infections\n• Use Disease Detection feature for diagnosis"
        elif 'oxygen' in user_message:
            response = "Dissolved Oxygen (DO):\n• Maintain above 5 mg/L\n• Below 3 mg/L is dangerous\n• Add aeration if low"
        elif 'hello' in user_message or 'hi' in user_message:
            response = "Hello! I'm PisciCare AI Assistant. Ask me about pH, temperature, feeding, diseases, or water quality!"
        else:
            response = "I'm here to help! Ask me about:\n• pH levels\n• Temperature ranges\n• Feeding guidelines\n• Common diseases\n• Water quality parameters"
        
        return jsonify({
            "response": response,
            "confidence": 85
        })
        
    except Exception as e:
        return jsonify({"response": "Sorry, I couldn't process your request.", "error": str(e)}), 400

# ENDPOINT 6: Health check
@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({
        "status": "healthy",
        "models_loaded": True,
        "species": list(water_encoder.classes_),
        "mongodb_connected": db is not None
    })

# ENDPOINT 7: Get alerts (from MongoDB)
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

# ENDPOINT 8: Save alert (called when threshold exceeded)
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

# RUN THE APP
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🐟 PISCI CARE BACKEND API")
    print(f"✅ Models loaded: {list(water_encoder.classes_)}")
    print(f"✅ MongoDB: {'Connected' if db is not None else 'Not connected'}")
    print("🚀 Starting server at http://localhost:5000")
    print("📡 API Endpoints:")
    print("   POST   /api/sensors        - Receive sensor data from Pi")
    print("   GET    /api/sensors/current - Get current sensor readings")
    print("   GET    /api/sensors/history - Get sensor history")
    print("   POST   /api/predict/water  - Water quality prediction")
    print("   POST   /api/chat           - AI Assistant chat")
    print("   GET    /api/health         - Health check")
    print("   GET    /api/alerts         - Get alerts")
    print("   POST   /api/alerts         - Save alert")
    
    app.run(host='0.0.0.0', port=5000, debug=True)