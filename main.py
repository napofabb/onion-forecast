import os
import json
import pandas as pd
from prophet.serialize import model_from_json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # <--- PENTING UNTUK SISDA
from pydantic import BaseModel
from supabase import create_client, Client
import google.generativeai as genai
from datetime import datetime

app = FastAPI()

# ======================================================
# ▶️ 1. SETUP CORS (PASSPORT UNTUK SISDA)
# ======================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Membenarkan semua website (termasuk SISDA) akses
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- LOAD MODEL ---
try:
    with open('fama_forecast_model.json', 'r') as fin:
        model = model_from_json(fin.read())
    print("✅ Model forecasting berjaya diload!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- SETUP CLIENTS ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- DATA STRUCTURE ---
class ForecastRequest(BaseModel):
    days: int = 30  # Default forecast 30 hari

# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    import os
    all_keys = list(os.environ.keys())
    gemini_related = [k for k in all_keys if "GEMINI" in k.upper()]
    return {
        "status": "Server FAMA Online (SISDA Ready ✅)",
        "adakah_kunci_ditemui": "YA" if os.environ.get("GEMINI_API_KEY") else "TIDAK",
        "cors_enabled": True
    }

@app.post("/predict")
def predict_price(req: ForecastRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # 1. Prophet Calculation
    future = model.make_future_dataframe(periods=req.days)
    forecast = model.predict(future)
    next_days = forecast.tail(req.days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    results = []
    average_predicted_price = next_days['yhat'].mean()
    min_predicted_price = next_days['yhat'].min()
    max_predicted_price = next_days['yhat'].max()
    
    for _, row in next_days.iterrows():
        results.append({
            "date": row['ds'].strftime('%Y-%m-%d'),
            "predicted_price": row['yhat']
        })

    # 2. Gemini Reasoning
    ai_insight = "Gemini key not found."
    if GEMINI_API_KEY:
        try:
            # Guna flash-latest
            gemini_model = genai.GenerativeModel('gemini-flash-latest')
            prompt = f"""
            Bertindak sebagai Senior Data Analyst FAMA Malaysia.
            Analisa data ramalan HARGA Bawang Besar India untuk {req.days} hari akan datang.
            
            DATA RAMALAN:
            - Purata Harga: RM {average_predicted_price:.2f} / kg
            - Harga Tertinggi: RM {max_predicted_price:.2f} / kg
            - Harga Terendah: RM {min_predicted_price:.2f} / kg
            
            TUGAS:
            1. Nyatakan trend harga (Menaik/Menurun).
            2. Apa implikasi kepada pengguna/peniaga?
            3. Cadangkan satu tindakan untuk FAMA.
            
            JAWAPAN (Bahasa Melayu Professional):
            """
            ai_response = gemini_model.generate_content(prompt)
            ai_insight = ai_response.text
        except Exception as e:
            ai_insight = f"Error generating AI insight: {str(e)}"

    # ======================================================
    # ▶️ 2. AUTO-SAVE KE SUPABASE (UNTUK REKOD)
    # ======================================================
    if supabase:
        try:
            data_to_save = {
                "forecast_date": datetime.now().strftime('%Y-%m-%d'),
                "item_name": "Bawang Besar India", 
                "predicted_value": float(f"{average_predicted_price:.2f}"),
                "ai_analysis": ai_insight
            }
            supabase.table('predictions').insert(data_to_save).execute()
            print("✅ Data berjaya disimpan ke Supabase!")
        except Exception as e:
            print(f"⚠️ Gagal simpan ke DB: {e}")

    return {
        "forecast_type": "Price (RM/kg)",
        "average_price": average_predicted_price,
        "min_price": min_predicted_price,
        "max_price": max_predicted_price,
        "ai_analysis": ai_insight,
        "daily_data": results
    }

# --- CHECKER ---
@app.get("/check-models")
def check_models():
    try:
        senarai_model = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                senarai_model.append(m.name)
        return {"status": "OK", "models": senarai_model}
    except Exception as e:
        return {"status": "ERROR", "detail": str(e)}
