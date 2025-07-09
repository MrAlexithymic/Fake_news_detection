from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Optional safety: try/except block for model loading
try:
    model = joblib.load("RandomForestClassifier_2.pkl")
    vectorizer = joblib.load("CountVectorizer_2.pkl")
except Exception as e:
    print(f"🔴 Failed to load model or vectorizer: {e}")
    model = None
    vectorizer = None


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return render_template("index.html", prediction="Error: Model not loaded")
    
    text = request.form['news']
    vect_text = vectorizer.transform([text])
    result = model.predict(vect_text)[0]
    label = "Real News" if result == 1 else "Fake News"
    return render_template("index.html", prediction=label)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # ✅ For Render
    app.run(host='0.0.0.0', port=port)
