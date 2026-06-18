from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Memuat model dan scaler (Pastikan file model.pkl hanya berisi 1 objek model)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f) 
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Mengambil data dari form HTML
            # Pastikan kunci di sini ('pregnancies', 'glucose', dll)
            # SAMA PERSIS dengan atribut name di <input> pada index.html
            data = {
                'Pregnancies': int(request.form['pregnancies']),
                'Glucose': int(request.form['glucose']),
                'BloodPressure': int(request.form['blood_pressure']),
                'SkinThickness': int(request.form['skin_thickness']),
                'Insulin': int(request.form['insulin']),
                'BMI': float(request.form['bmi']),
                'DiabetesPedigreeFunction': float(request.form['diabetes_pedigree_function']),
                'Age': int(request.form['age'])
            }
            
            # Mengubah data input menjadi DataFrame
            df = pd.DataFrame(data, index=[0])
            
            # Normalisasi data menggunakan scaler
            X_scaled = scaler.transform(df)
            
            # Melakukan prediksi
            # Langsung menggunakan model karena sekarang sudah objek tunggal
            prediction_value = model.predict(X_scaled)[0]
            
            # Menerjemahkan output
            hasil = "Diabetic" if int(prediction_value) == 1 else "Non-Diabetic"
            
            return render_template('index.html', prediction=hasil)
            
        except Exception as e:
            return f"Terjadi kesalahan: {str(e)}"

if __name__ == '__main__':
    # Ubah debug menjadi False saat di server produksi
    app.run(debug=True, host='0.0.0.0')