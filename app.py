from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import xgboost as xgb
import pandas as pd
import os

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, "flights.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

model = xgb.XGBRegressor()
model.load_model("flight_price_xgb_model.json")

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

@app.route("/", methods=["GET"])
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "airline": request.form.get("airline"),
            "source_city": request.form.get("source_city"),
            "departure_time": request.form.get("departure_time"),
            "stops": request.form.get("stops"),
            "arrival_time": request.form.get("arrival_time"),
            "destination_city": request.form.get("destination_city"),
            "duration": float(request.form.get("duration")),
            "days_left": int(request.form.get("days_left"))
        }

        features = pd.DataFrame([data])
        prediction = round(model.predict(features)[0], 2)

        new_record = FlightPrediction(**data, predicted_price=prediction)
        db.session.add(new_record)
        db.session.commit()

        return render_template("result.html", prediction=prediction)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
