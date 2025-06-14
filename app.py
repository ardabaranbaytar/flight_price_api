from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Modeli yükle
model_path = os.path.join(os.path.dirname(__file__), "flight_pipeline.pkl")
model = joblib.load(model_path)

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

            # DataFrame oluştur ve tahmin yap
            df = pd.DataFrame([form_data])
            prediction = round(model.predict(df)[0], 2)

        except Exception as e:
            return render_template("form.html", prediction=None, error=str(e))

    return render_template("form.html", prediction=prediction)

# 🚫 BURAYI RENDER'DA KULLANMA
# if __name__ == "__main__":
#     app.run(debug=True)




