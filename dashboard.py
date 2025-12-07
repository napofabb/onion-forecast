import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="FAMA AI Forecasting",
    page_icon="📊",
    layout="wide"
)

# --- TAJUK & HEADER ---
st.title("🌾 FAMA Intelligent Forecasting Dashboard")
st.markdown("Sistem peramalan harga pasaran menggunakan **Prophet (Time-series)** dan **Gemini (AI Reasoning)**.")
st.divider()

# --- SIDEBAR (INPUT) ---
with st.sidebar:
    st.header("Tetapan Ramalan")
    item = st.selectbox("Pilih Komoditi:", ["Bawang Besar India", "Cili Merah", "Kobis Bulat"])
    days = st.slider("Tempoh Ramalan (Hari):", 7, 90, 30)
    
    st.markdown("---")
    predict_btn = st.button("Jana Ramalan 🚀", type="primary")
    
    st.info("💡 Data sejarah diambil dari pangkalan data Supabase FAMA.")

# --- LOGIC API ---
# ⚠️ GANTI LINK NI DENGAN LINK RAILWAY API KAU (YANG ADA /predict)
API_URL = "https://onion-forecast-production.up.railway.app/predict"

if predict_btn:
    with st.spinner(f"Sedang memproses ramalan harga {item} untuk {days} hari..."):
        try:
            # Hantar request ke Backend Railway
            payload = {"days": days}
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # --- BAHAGIAN 1: KPI CARDS ---
                st.subheader(f"📊 Hasil Ramalan: {item}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(label="Purata Harga (Jangkaan)", value=f"RM {data.get('average_price', 0):.2f}")
                with col2:
                    st.metric(label="Harga Tertinggi", value=f"RM {data.get('max_price', 0):.2f}")
                with col3:
                    st.metric(label="Harga Terendah", value=f"RM {data.get('min_price', 0):.2f}")
                
                # --- BAHAGIAN 2: GRAF INTERAKTIF ---
                st.markdown("### 📈 Trend Harga Harian")
                
                # Convert JSON data ke Pandas DataFrame
                df_chart = pd.DataFrame(data['daily_data'])
                
                # Buat graf cantik guna Plotly
                fig = px.line(df_chart, x='date', y='predicted_price', 
                              markers=True, title=f"Unjuran Harga {days} Hari Ke Depan")
                
                fig.update_layout(xaxis_title="Tarikh", yaxis_title="Harga (RM/kg)")
                fig.update_traces(line_color='#E74C3C') # Warna merah FAMA sikit
                
                st.plotly_chart(fig, use_container_width=True)
                
                # --- BAHAGIAN 3: AI INSIGHT (GEMINI) ---
                st.markdown("### 🤖 Analisis Pintar (AI Insight)")
                
                with st.chat_message("assistant"):
                    st.write(data['ai_analysis'])
                    
            else:
                st.error(f"Gagal mendapat data dari server. Error: {response.text}")
                
        except Exception as e:
            st.error(f"Terjadi ralat sambungan: {e}")

else:
    st.info("👈 Sila tekan butang 'Jana Ramalan' di sebelah kiri untuk bermula.")