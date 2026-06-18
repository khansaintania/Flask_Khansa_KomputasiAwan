from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Memuat model dan scaler (Berisi list 2 model dari proses training)
with open('model.pkl', 'rb') as f:
    models = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Definisikan nama model untuk ditampilkan di dropdown HTML
model_names = ['Decision Tree', 'SVC']

@app.route('/')
def index():
    # Kirim variabel model_names ke index.html agar dropdown terisi
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Mengambil data dari form HTML
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

            # Mengambil model pilihan user dari dropdown HTML
            selected_model_name = request.form['model']
            model_idx = model_names.index(selected_model_name)
            clf = models[model_idx]

            # Melakukan prediksi menggunakan model yang dipilih
            prediction_value = clf.predict(X_scaled)[0]

            # Menerjemahkan output
            hasil = "Diabetic" if int(prediction_value) == 1 else "Non-Diabetic"

            # Mengembalikan hasil dan jangan lupa kirim ulang model_names-nya
            return render_template('index.html', model_names=model_names, prediction=hasil)

        except Exception as e:
            return f"Terjadi kesalahan: {str(e)}"

if __name__ == '__main__':
    # Ubah debug menjadi False saat di server produksi
    app.run(debug=True, host='0.0.0.0')