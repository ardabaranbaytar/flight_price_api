from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import os

app = Flask(__name__)

# Model yükle
model = xgb.XGBRegressor()
model.load_model("flight_price_xgb_model.json")

# HTML form sayfası
@app.route("/", methods=["GET"])
def home():
    return """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <title>Uçuş Tahmini</title>
        <link rel="stylesheet" href="/static/style.css">
    </head>
    <body>
        <h1>Uçuş Bilgilerini Gir</h1>
        <form method="POST" action="/predict">
            <label>Havayolu:</label><input type="text" name="airline" required><br>
            <label>Kalkış Şehri:</label><input type="text" name="source_city" required><br>
            <label>Kalkış Saati:</label><input type="text" name="departure_time" required><br>
            <label>Aktarma Sayısı:</label><input type="text" name="stops" required><br>
            <label>Varış Saati:</label><input type="text" name="arrival_time" required><br>
            <label>Varış Şehri:</label><input type="text" name="destination_city" required><br>
            <label>Süre (dk):</label><input type="number" name="duration" required><br>
            <label>Gün Sayısı (days_left):</label><input type="number" name="days_left" required><br>
            <button type="submit">Tahmin Et</button>
        </form>
    </body>
    </html>
    """

# Tahmin işlemi
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "airline": request.form["airline"],
            "source_city": request.form["source_city"],
            "departure_time": request.form["departure_time"],
            "stops": request.form["stops"],
            "arrival_time": request.form["arrival_time"],
            "destination_city": request.form["destination_city"],
            "duration": float(request.form["duration"]),
            "days_left": int(request.form["days_left"]),
        }
        df = pd.DataFrame([data])
        prediction = round(model.predict(df)[0], 2)
        return f"<h2>Tahmin Edilen Fiyat: {prediction} TL</h2>"
    except Exception as e:
        return f"<p>Hata oluştu: {str(e)}</p>"

# Çalıştır
if __name__ == "__main__":
    app.run(debug=True)
