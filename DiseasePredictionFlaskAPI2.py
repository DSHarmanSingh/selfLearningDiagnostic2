import numpy as np
import joblib
import tensorflow.lite as tflite
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import datetime

# Load Label Encoder & TF-IDF Vectorizer
label_encoder = joblib.load("LabelEncoder.pkl")
tfidf = joblib.load("tfidf.pkl")

# Load TF-Lite Model
interpreter = tflite.Interpreter(model_path="DiseasePrediction_DeepLearning.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Connect to MongoDB Atlas (update with your credentials)
MONGO_URI = "mongodb+srv://DSHarman:harman.mongodb.ds@harmandiseaseprediction.wbo3b.mongodb.net/ai_medical_assistant?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI)
db = client["sldds"]
queries_collection = db["user_queries"]
feedback_collection = db["user_feedback"]

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to AI-Powered Healthcare API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms_text = data.get("symptoms", "").strip()

    if not symptoms_text:
        return jsonify({"error": "No symptoms provided!"}), 400

    # Convert symptoms to TF-IDF vector
    symptoms_vector = tfidf.transform([symptoms_text]).toarray().astype(np.float32)

    # Run TF-Lite model
    interpreter.set_tensor(input_details[0]['index'], symptoms_vector)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted disease
    predicted_label = np.argmax(prediction)
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0]

    # Save query in MongoDB (store prediction without feedback yet)
    queries_collection.insert_one({
        "symptoms": symptoms_text,
        "predicted_disease": predicted_disease,
        "timestamp": datetime.datetime.utcnow()
    })

    return jsonify({"predicted_disease": predicted_disease})

@app.route('/update', methods=['POST'])
def update():
    """
    Stores user feedback in a separate collection.
    This does NOT automatically update the model.
    You can review feedback later and trigger manual retraining.
    """
    data = request.json
    symptoms = data.get("symptoms", "").strip()
    correct_disease = data.get("correct_disease", "").strip()

    if not symptoms or not correct_disease:
        return jsonify({"error": "Symptoms or correct disease missing!"}), 400

    # Store feedback in MongoDB for later manual review
    feedback_collection.insert_one({
        "symptoms": symptoms,
        "correct_disease": correct_disease,
        "timestamp": datetime.datetime.utcnow()
    })

    return jsonify({"message": "Feedback stored successfully!"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=10000)
