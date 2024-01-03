from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__,template_folder='template')
df = pd.read_csv('Harga BawangTest.csv', sep=';')
# Drop rows with missing values
df = df.dropna()
data = df[['Harga']].values.astype(float)

scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)
sequence_length = 10

# Take the last part of the dataset for prediction
data_last = data_normalized[-sequence_length:]
data_last = np.reshape(data_last, (1, sequence_length, 1))

# Load the pre-trained model
loaded_model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        tanggal_terakhir = df['Tanggal'].iloc[-1]
        tanggal_terakhir = datetime.strptime(tanggal_terakhir, '%d/%m/%Y')
# Meminta pengguna memasukkan tanggal prediksi
        # tanggal_prediksi_input = input("Masukkan tanggal prediksi (format: dd/mm/yyyy): ")
        tanggal_prediksi_input = str(request.form['tanggal_terakhir'])
        tanggal_prediksi = datetime.strptime(tanggal_prediksi_input, '%d/%m/%Y')

        selisih_hari = (tanggal_prediksi - tanggal_terakhir).days
        jumlah_langkah_waktu_ke_depan = selisih_hari
        data_akhir = data_normalized[-sequence_length:]
        data_akhir = np.reshape(data_akhir, (1, sequence_length, 1))
        hasil_prediksi = None
# Ambil sebagian terakhir dari dataset untuk digunakan sebagai awal prediksi
        for _ in range(jumlah_langkah_waktu_ke_depan):
            prediksi_langkah_ini = loaded_model.predict(data_akhir)
            prediksi_langkah_ini_denormalized = scaler.inverse_transform(prediksi_langkah_ini.reshape(-1, 1))
            data_akhir = np.append(data_akhir[:, 1:, :], prediksi_langkah_ini.reshape(1, 1, 1), axis=1)
            hasil_prediksi = prediksi_langkah_ini_denormalized[0, 0]
        return render_template('index.html', prediction=hasil_prediksi)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
