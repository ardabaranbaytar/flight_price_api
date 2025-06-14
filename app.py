from flask import Flask, request
import xgboost as xgb
import pandas as pd

app = Flask(__name__)
model = xgb.XGBRegressor()
model.load_model("flight_price_xgb_model.json")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
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
        features = pd.DataFrame([data])
        prediction = round(model.predict(features)[0], 2)
        return f"<h2>Tahmini Bilet Fiyatı: ₹{prediction}</h2>"

    return '''
        <h1>Uçuş Bilgilerini Gir</h1>
        <form method="POST">
            Havayolu: <input name="airline"><br>
            Kalkış Şehri: <input name="source_city"><br>
            Kalkış Saati: <input name="departure_time"><br>
            Aktarma Sayısı: <input name="stops"><br>
            Varış Saati: <input name="arrival_time"><br>
            Varış Şehri: <input name="destination_city"><br>
            Süre (dk): <input name="duration" type="number"><br>
            Gün Sayısı (days_left): <input name="days_left" type="number"><br>
            <input type="submit" value="Tahmin Et">
        </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)
