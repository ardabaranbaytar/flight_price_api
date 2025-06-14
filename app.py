from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Veritabanı konfigürasyonu
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///flights.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Tahmin modeli yükle
model = joblib.load("xgboost_flight_price_model.pkl")

# Veritabanı modeli tanımı
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

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Form verilerini al
        input_data = {
            "airline": request.form["airline"],
            "source_city": request.form["source_city"],
            "departure_time": request.form["departure_time"],
            "stops": request.form["stops"],
            "arrival_time": request.form["arrival_time"],
            "destination_city": request.form["destination_city"],
            "duration": float(request.form["duration"]),
            "days_left": int(request.form["days_left"])
        }

        # Tahmini yap
        df = pd.DataFrame([input_data])
        prediction = round(model.predict(df)[0], 2)

        # Aynı veri daha önce eklenmiş mi kontrol et
        existing = FlightPrediction.query.filter_by(
            airline=input_data["airline"],
            source_city=input_data["source_city"],
            departure_time=input_data["departure_time"],
            stops=input_data["stops"],
            arrival_time=input_data["arrival_time"],
            destination_city=input_data["destination_city"],
            duration=input_data["duration"],
            days_left=input_data["days_left"],
            predicted_price=prediction
        ).first()

        if not existing:
            new_prediction = FlightPrediction(**input_data, predicted_price=prediction)
            db.session.add(new_prediction)
            db.session.commit()

        return render_template("index.html", prediction=prediction)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)


