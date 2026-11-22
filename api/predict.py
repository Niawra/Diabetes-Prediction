import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "diabetes_model.pkl"
_artifacts = joblib.load(MODEL_PATH)
_model = _artifacts["model"]
_scaler = _artifacts["scaler"]
_feature_columns = _artifacts["feature_columns"]


def _json_response(handler, status, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.rstrip("/") != "/api/predict":
            return _json_response(self, 404, {"error": "Endpoint tidak ditemukan."})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 10_000:
                return _json_response(self, 400, {"error": "Data permintaan tidak valid."})
            data = json.loads(self.rfile.read(length).decode("utf-8"))

            gender = data.get("gender")
            smoking = data.get("smoking_history")
            if gender not in ["Female", "Male", "Other"]:
                raise ValueError("Pilih gender yang tersedia.")
            if smoking not in ["never", "current", "former", "ever", "not current", "No Info"]:
                raise ValueError("Pilih riwayat merokok yang tersedia.")

            age = float(data.get("age"))
            bmi = float(data.get("bmi"))
            hba1c = float(data.get("hba1c"))
            glucose = float(data.get("glucose"))
            hypertension = int(bool(data.get("hypertension")))
            heart_disease = int(bool(data.get("heart_disease")))

            if not 1 <= age <= 100:
                raise ValueError("Umur harus antara 1 sampai 100 tahun.")
            if not 10 <= bmi <= 80:
                raise ValueError("BMI harus antara 10 sampai 80.")
            if not 3 <= hba1c <= 20:
                raise ValueError("HbA1c harus antara 3 sampai 20.")
            if not 40 <= glucose <= 500:
                raise ValueError("Blood glucose harus antara 40 sampai 500.")

            row = {
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "bmi": bmi,
                "HbA1c_level": hba1c,
                "blood_glucose_level": glucose,
                "gender_Female": int(gender == "Female"),
                "gender_Male": int(gender == "Male"),
                "gender_Other": int(gender == "Other"),
                "smoking_history_No Info": int(smoking == "No Info"),
                "smoking_history_current": int(smoking == "current"),
                "smoking_history_ever": int(smoking == "ever"),
                "smoking_history_former": int(smoking == "former"),
                "smoking_history_never": int(smoking == "never"),
                "smoking_history_not current": int(smoking == "not current"),
            }
            input_frame = pd.DataFrame([row]).reindex(columns=_feature_columns, fill_value=0)
            scaled = _scaler.transform(input_frame)
            prediction = int(_model.predict(scaled)[0])
            score = float(_model.predict_proba(scaled)[0][1])

            return _json_response(self, 200, {
                "prediction": prediction,
                "label": "Kelas 1 — terdeteksi oleh model" if prediction == 1 else "Kelas 0 — tidak terdeteksi oleh model",
                "score": round(score * 100, 2),
                "message": "Hasil ini hanya keluaran model pembelajaran mesin untuk keperluan edukasi, bukan diagnosis atau penilaian medis."
            })
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return _json_response(self, 400, {"error": str(exc) or "Periksa kembali data yang dimasukkan."})
        except Exception:
            return _json_response(self, 500, {"error": "Terjadi kesalahan pada server saat memproses prediksi."})

    def do_GET(self):
        return _json_response(self, 405, {"error": "Gunakan metode POST untuk endpoint prediksi."})
