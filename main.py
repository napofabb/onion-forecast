import os
import json
import pandas as pd
from prophet.serialize import model_from_json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
import google.generativeai as genai

app = FastAPI()

# --- CONFIGURATION (Akan ditarik dari Railway Environment Variables nanti) ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- LOAD MODEL (Hanya sekali bila server start) ---
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
    return {"status": "Server FAMA Online", "model_loaded": model is not None}

@app.post("/predict")
def predict_supply(req: ForecastRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # 1. Buat dataframe masa depan
    future = model.make_future_dataframe(periods=req.days)
    
    # 2. Run prediction
    forecast = model.predict(future)
    
    # 3. Ambil data 30 hari terakhir sahaja (masa depan)
    next_30_days = forecast.tail(req.days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    # Format date jadi string
    results = []
    total_predicted_supply = 0
    
    for _, row in next_30_days.iterrows():
        val = row['yhat']
        results.append({
            "date": row['ds'].strftime('%Y-%m-%d'),
            "prediction": val
        })
        total_predicted_supply += val

    # 4. Generate AI Insight (Gemini)
    ai_insight = "Gemini key not found."
    if GEMINI_API_KEY:
        try:
            gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            prompt = f"""
            Sebagai Data Analyst FAMA, analisa ramalan ini:
            - Item: Bawang Besar India
            - Total Ramalan 30 Hari: {total_predicted_supply:.2f} Tan
            
            Berikan 1 ayat ringkas trend supply ini dan 1 cadangan tindakan.
            """
            ai_response = gemini_model.generate_content(prompt)
            ai_insight = ai_response.text
        except Exception as e:
            ai_insight = f"Error generating AI insight: {str(e)}"

    return {
        "total_forecast": total_predicted_supply,
        "ai_analysis": ai_insight,
        "daily_data": results
    }