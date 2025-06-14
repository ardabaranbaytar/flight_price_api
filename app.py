from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Modeli güvenli şekilde yükle
model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "flight_pipeline.pkl")
model = joblib.load(model_path)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Formdan gelen verileri oku (arrival_time ve departure_time eklendi!)
            form_data = {
                "airline": request.form.get("airline"),
                "source_city": request.form.get("source_city"),
                "destination_city": request.form.get("destination_city"),
                "stops": request.form.get("stops"),
                "departure_time": request.form.get("departure_time"),
                "arrival_time": request.form.get("arrival_time"),
                "duration": float(request.form.get("duration")),
                "days_left": int(request.form.get("days_left")),
            }

            # Veriyi DataFrame'e çevir
            df = pd.DataFrame([form_data])
            print("Formdan gelen veriler:")
            print(df)

            # Tahmin
            prediction = round(model.predict(df)[0], 2)
            print("Tahmin sonucu:", prediction)

        except Exception as e:
            print("Tahmin sırasında hata:", str(e))
            error = f"Bir hata oluştu: {str(e)}"

    return render_template("form.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)








