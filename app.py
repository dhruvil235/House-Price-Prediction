from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and columns
model = joblib.load("model/house_model.pkl")
model_columns = joblib.load("model/model_columns.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = [float(request.form[col]) for col in model_columns]
        df = pd.DataFrame([input_data], columns=model_columns)
        prediction = model.predict(df)[0]
        return render_template("index.html", prediction=f"{prediction:.2f}")
    except Exception as e:
        return render_template("index.html", prediction="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
