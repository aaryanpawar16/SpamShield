import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("spam_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    message = data.get("message", "")

    # Prediction
    prediction = model.predict([message])[0]
    probability = model.predict_proba([message])[0][1]  # Spam probability

    return jsonify({
        "prediction": "Spam 🚨" if prediction == 1 else "Not Spam ✅",
        "confidence": f"{probability*100:.2f}%"
    })

if __name__ == "__main__":
    app.run(debug=True)
