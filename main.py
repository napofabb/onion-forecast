import os
import json
import pandas as pd
from prophet.serialize import model_from_json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
import google.generativeai as genai

app = FastAPI()

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
    # Guna alias 'latest' atau model yang available
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
        "status": "Server FAMA Online (Mode: Price Forecasting)",
        "adakah_kunci_ditemui": "YA" if os.environ.get("GEMINI_API_KEY") else "TIDAK",
        "variable_gemini_yang_dijumpai": gemini_related
    }

@app.post("/predict")
def predict_price(req: ForecastRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # 1. Buat dataframe masa depan
    future = model.make_future_dataframe(periods=req.days)
    
    # 2. Run prediction
    forecast = model.predict(future)
    
    # 3. Ambil data masa depan sahaja
    next_days = forecast.tail(req.days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    results = []
    
    # KIRAAN KHAS UNTUK HARGA: Kita cari Purata (Average), bukan Total
    average_predicted_price = next_days['yhat'].mean()
    min_predicted_price = next_days['yhat'].min()
    max_predicted_price = next_days['yhat'].max()
    
    for _, row in next_days.iterrows():
        val = row['yhat']
        results.append({
            "date": row['ds'].strftime('%Y-%m-%d'),
            "predicted_price": val
        })

    # 4. Generate AI Insight (Gemini) - Context Harga
    ai_insight = "Gemini key not found."
    
    if GEMINI_API_KEY:
        try:
            # Guna flash-latest untuk elak error 404/Quota
            gemini_model = genai.GenerativeModel('gemini-flash-latest')
            
            prompt = f"""
            Bertindak sebagai Senior Data Analyst FAMA Malaysia.
            Analisa data ramalan HARGA (Price Forecasting) untuk Bawang Besar India bagi {req.days} hari akan datang.
            
            DATA RAMALAN:
            - Purata Harga Dijangka: RM {average_predicted_price:.2f} / kg
            - Harga Tertinggi: RM {max_predicted_price:.2f} / kg
            - Harga Terendah: RM {min_predicted_price:.2f} / kg
            
            TUGAS:
            1. Nyatakan trend harga secara ringkas (Menaik/Menurun/Stabil).
            2. Apa implikasi harga ini kepada pengguna atau peniaga?
            3. Cadangkan satu tindakan untuk FAMA (contoh: Jualan Agro Madani, kawalan siling, atau pemantauan).
            
            JAWAPAN (Bahasa Melayu Professional):
            """
            
            ai_response = gemini_model.generate_content(prompt)
            ai_insight = ai_response.text
        except Exception as e:
            ai_insight = f"Error generating AI insight: {str(e)}"

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
