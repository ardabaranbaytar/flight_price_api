from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import xgboost as xgb
import pandas as pd
import os

app = Flask(__name__)

# Veritabanı ayarı
basedir = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, "flights.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Modeli yükle
model = xgb.XGBRegressor()
model.load_model("flight_price_xgb_model.json")

# SQLAlchemy modeli
class FlightPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    airline = db.Column(db.String(50))
    source_city = db.Column(db.String(50))
    departure_time = db.Column(db.String(50))
    stops = db.Column(db.String(20))
    arrival_time = db.Column(db.String(50))
    destination_city = db.Column(db.String(50))
    duration = db.Column(db.Float)
    days_left = db.Column(db.Integer)
    predicted_price = db.Column(db.Float)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = pd.DataFrame([data])

        prediction = round(model.predict(features)[0], 2)

        record = FlightPrediction(**data, predicted_price=prediction)
        db.session.add(record)
        db.session.commit()

        return jsonify({"predicted_price": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "✈️ Flight Price Prediction API is up and running!"

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
