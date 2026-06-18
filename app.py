from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Memuat model dan scaler (pastikan file ini ada di folder yang sama)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route("/")
def index():
    # Menampilkan halaman form awal
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Mengambil data dari form HTML
            pregnancies = int(request.form["pregnancies"])
            glucose = int(request.form["glucose"])
            blood_pressure = int(request.form["blood_pressure"])
            skin_thickness = int(request.form["skin_thickness"])
            insulin = int(request.form["insulin"])
            bmi = float(request.form["bmi"])
            dpf = float(request.form["dpf"])
            age = int(request.form["age"])

            # Menyusun data ke dalam dictionary sesuai format dataset
            data = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age': age
            }

            # Mengubah dictionary menjadi DataFrame Pandas
            df = pd.DataFrame(data, index=[0])

            # Normalisasi data menggunakan scaler
            scaled_data = scaler.transform(df)

            # Melakukan prediksi menggunakan model Decision Tree
            prediction = model.predict(scaled_data)

            # Menerjemahkan output numerik menjadi teks
            if prediction[0] == 1:
                hasil = "Diabetic"
            else:
                hasil = "Non-Diabetic"

            # Mengembalikan halaman index beserta hasil prediksi
            return render_template("index.html", prediction=hasil)
            
        except Exception as e:
            return f"Terjadi kesalahan saat memproses data: {e}"

if __name__ == "__main__":
    app.run(debug=True)