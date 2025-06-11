import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open('penyakit_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Judul aplikasi
st.title('Prediksi Penyakit Jantung')
st.write("Masukkan data pasien di bawah ini untuk memprediksi apakah pasien berpotensi terkena penyakit jantung.")

# Form input pengguna
age = st.number_input('Usia', min_value=1, max_value=120, value=30)
sex = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
cp = st.selectbox('Tipe Nyeri Dada (Chest Pain Type)', [0, 1, 2, 3])
trestbps = st.number_input('Tekanan Darah (resting blood pressure)', min_value=80, max_value=200, value=120)
chol = st.number_input('Kolesterol', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl?', ['Ya', 'Tidak'])
restecg = st.selectbox('Hasil EKG saat istirahat', [0, 1, 2])
thalach = st.number_input('Denyut Jantung Maksimum (thalach)', min_value=60, max_value=220, value=150)
exang = st.selectbox('Angina saat olahraga (exercise induced angina)', ['Ya', 'Tidak'])
oldpeak = st.number_input('Oldpeak (depresi ST)', min_value=0.0, max_value=6.0, value=1.0, step=0.1)
slope = st.selectbox('Kemiringan segmen ST', [0, 1, 2])
ca = st.selectbox('Jumlah pembuluh darah utama (0–3)', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia', [0, 1, 2, 3])

# Konversi input ke format numerik sesuai model
input_data = np.array([[
    age,
    1 if sex == 'Laki-laki' else 0,
    cp,
    trestbps,
    chol,
    1 if fbs == 'Ya' else 0,
    restecg,
    thalach,
    1 if exang == 'Ya' else 0,
    oldpeak,
    slope,
    ca,
    thal
]])

# Tombol prediksi
if st.button('Prediksi'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error('Pasien berisiko terkena penyakit jantung.')
    else:
        st.success('Pasien tidak berisiko terkena penyakit jantung.')

# Opsional: tampilkan dataset
if st.checkbox('Tampilkan Data Awal'):
    df = pd.read_csv('heart.csv')
    st.dataframe(df.head())
