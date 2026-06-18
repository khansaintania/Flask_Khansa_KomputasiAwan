from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Pastikan file model.pkl dan scaler.pkl ada di folder yang sama
with open('model.pkl', 'rb') as f:
    models = pickle.load(f) # Berisi list model: [Decision Tree, SVC]
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model_names = ['Decision Tree', 'SVC']

@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
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
        
        # Melakukan scaling pada data
        X_scaled = scaler.transform(df)
        
        # Memilih model berdasarkan pilihan dropdown user
        selected_model_name = request.form['model']
        model_idx = model_names.index(selected_model_name)
        clf = models[model_idx]
        
        # Melakukan prediksi
        y_pred = clf.predict(X_scaled)
        prediction = 'Diabetic' if int(y_pred[0]) == 1 else 'Non-Diabetic'
        
        # Mengembalikan hasil ke halaman utama
        return render_template('index.html', model_names=model_names, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')