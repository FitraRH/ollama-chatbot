from flask import Flask, request, jsonify
from pymongo import MongoClient
import datetime

app = Flask(__name__)

# === KONEKSI KE MONGODB ATLAS ===
MONGO_URI = "mongodb+srv://HSC046:RGFBAKSO@rgfbakso.mzcsy.mongodb.net/?retryWrites=true&w=majority&appName=RGFBAKSO"

try:
    client = MongoClient(MONGO_URI)
    db = client["sensordb"]
    collection = db["sensor_data"]
    print("✅ Connected to MongoDB Atlas!")
except Exception as e:
    print("❌ Failed to connect to MongoDB:", str(e))

@app.route('/', methods=['POST'])
def home():
    return "Hello World"

# === ENDPOINT UNTUK MENYIMPAN DATA SENSOR ===
@app.route('/ambatukam', methods=['POST'])
def receive_sensor_data():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data received"}), 400

        data["timestamp"] = datetime.datetime.now()
        collection.insert_one(data)
        return jsonify({"message": "✅ Data stored successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)