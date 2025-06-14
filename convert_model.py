import joblib
import xgboost as xgb

# 1. Pickle modelini yükle
model = joblib.load("xgboost_flight_price_model.pkl")

# 2. Pipeline içindeki XGBoost modelini al
xgb_model = model.named_steps["xgb"]

# 3. Booster nesnesini al ve JSON formatında kaydet
booster = xgb_model.get_booster()
booster.save_model("flight_price_xgb_model.json")

print("✅ JSON formatında model başarıyla kaydedildi: flight_price_xgb_model.json")




