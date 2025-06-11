# 🏡 Boston House Price Predictor

A simple and interactive web app built with **Flask** that predicts housing prices in Boston based on 13 key features using a machine learning model.

---

## 🚀 Live Demo

🌐 [Click here to view the app](https://house-price-prediction-ie6r.onrender.com)

---

## 📂 Project Structure

.
├── app.py # Flask application
├── model.pkl # Trained ML model
├── requirements.txt # Python dependencies
├── templates/
│ └── index.html # Frontend UI
├── static/ # (Optional) CSS or assets
└── housing.csv # Dataset used for training


---

## 📊 Features Used for Prediction

- `CRIM`: Per capita crime rate
- `ZN`: Proportion of residential land zoned
- `INDUS`: Proportion of non-retail business acres
- `CHAS`: Charles River (1 = bounds river; 0 = otherwise)
- `NOX`: Nitric oxides concentration
- `RM`: Avg number of rooms
- `AGE`: Proportion of owner-occupied units built before 1940
- `DIS`: Weighted distances to employment centers
- `RAD`: Index of accessibility to radial highways
- `TAX`: Property-tax rate
- `PTRATIO`: Pupil-teacher ratio
- `B`: Proportion of African Americans
- `LSTAT`: % lower status population

---

## 🧠 How It Works

1. User inputs values for 13 features
2. Flask passes inputs to a trained Linear Regression model
3. Model predicts the price of the house
4. Result is displayed in ₹ Lakhs (Indian currency formatting)

---

## ⚙️ Installation & Local Setup

### 🔧 Prerequisites

- Python 3.8+
- pip (Python package manager)

### 📥 Installation

```bash
# Clone the repo
git clone https://github.com/dhruvil235/House-Price-Prediction
cd boston-house-price-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
