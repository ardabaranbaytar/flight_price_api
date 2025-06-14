from flask import Flask, request
import xgboost as xgb
import pandas as pd
import pickle

app = Flask(__name__)

# Modeli yükle
model = xgb.XGBRegressor()
model.load_model("flight_price_xgb_model.json")

# Encoder'ı yükle
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return """
    <html>
    <head><title>Uçuş Tahmini</title></head>
    <body>
        <h1>Uçuş Bilgilerini Gir</h1>
        <form method="POST" action="/predict">
            <label>Havayolu:</label><input type="text" name="airline"><br>
            <label>Kalkış Şehri:</label><input type="text" name="source_city"><br>
            <label>Kalkış Saati:</label><input type="text" name="departure_time"><br>
            <label>Aktarma:</label><input type="text" name="stops"><br>
            <label>Varış Saati:</label><input type="text" name="arrival_time"><br>
            <label>Varış Şehri:</label><input type="text" name="destination_city"><br>
            <label>Süre (dk):</label><input type="number" name="duration"><br>
            <label>Gün Sayısı:</label><input type="number" name="days_left"><br>
            <button type="submit">Tahmin Et</button>
        </form>
    </body>
    </html>
    """

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = {
            "airline": request.form["airline"],
            "source_city": request.form["source_city"],
            "departure_time": request.form["departure_time"],
            "stops": request.form["stops"],
            "arrival_time": request.form["arrival_time"],
            "destination_city": request.form["destination_city"],
            "duration": float(request.form["duration"]),
            "days_left": int(request.form["days_left"]),
        }

        df = pd.DataFrame([form_data])

        # Encode et
        features = encoder.transform(df)

        # Tahmin
        prediction = round(model.predict(features)[0], 2)

        return f"<h2>Tahmin Edilen Fiyat: {prediction} TL</h2>"

    except Exception as e:
        return f"<p>Hata oluştu: {str(e)}</p>"

if __name__ == "__main__":
    app.run(debug=True)
