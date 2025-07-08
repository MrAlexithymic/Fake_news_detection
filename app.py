from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load("RandomForestClassifier_2.pkl")
vectorizer = joblib.load("CountVectorizer_2.pkl")


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']
    vect_text = vectorizer.transform([text])
    result = model.predict(vect_text)[0]
    label = "Real News" if result == 1 else "Fake News"
    return render_template("index.html", prediction=label)

if __name__ == '__main__':
    app.run(debug=True)
