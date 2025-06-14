from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Model yükleniyor
model = joblib.load("flight_pipeline.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            # Form verilerini al
            form_data = {
                "airline": request.form.get("airline"),
                "source_city": request.form.get("source_city"),
                "destination_city": request.form.get("destination_city"),
                "stops": request.form.get("stops"),
                "duration": float(request.form.get("duration")),
                "days_left": int(request.form.get("days_left")),
            }

            df = pd.DataFrame([form_data])
            prediction = round(model.predict(df)[0], 2)

        except Exception as e:
            return f"<h3>Hata oluştu: {str(e)}</h3>"

    return render_template("form.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)


