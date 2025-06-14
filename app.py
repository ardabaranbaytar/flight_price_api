from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Pipeline model yükleniyor
model = joblib.load("flight_pipeline.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Form verilerini al
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

        df = pd.DataFrame([form_data])  # Tek satırlık DataFrame oluştur
        prediction = round(model.predict(df)[0], 2)  # Tahmin yap

        return render_template("form.html", prediction=prediction)

    except Exception as e:
        return f"<p>Hata oluştu: {str(e)}</p>"

if __name__ == "__main__":
    app.run(debug=True)

