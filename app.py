import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Kualitas Red Wine", page_icon="🍷")

@st.cache_resource
def load_all():
    model = joblib.load("wine_model.joblib")
    with open("meta.json") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_all()
features = meta["features"]
stats = meta["stats"]

st.title("🍷 Prediksi Kualitas Red Wine")
st.write("Masukkan sifat fisikokimia wine. Model memprediksi skor kualitas (skala 0-10).")

inputs = {}
cols = st.columns(2)
for i, f in enumerate(features):
    lo, hi, med = float(stats[f]["min"]), float(stats[f]["max"]), float(stats[f]["50%"])
    inputs[f] = cols[i % 2].number_input(
        f, min_value=lo, max_value=hi, value=med, step=(hi - lo) / 100, format="%.4f"
    )

if st.button("Prediksi kualitas"):
    X_new = pd.DataFrame([inputs])[features]   # urutan kolom harus sama dengan saat training
    pred = float(model.predict(X_new)[0])
    st.metric("Prediksi skor kualitas", f"{pred:.2f}")
    st.info(
        f"Kesalahan rata-rata model sekitar ±{meta['mae_test']} poin pada data uji. "
        "Model ini alat bantu penyaring, bukan pengganti penilaian ahli. "
        "Prediksi cenderung mendekati 5-6 sehingga kurang andal untuk wine yang sangat buruk atau sangat bagus."
    )