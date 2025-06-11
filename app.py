from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('penyakit_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data input dari form HTML
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])

        # Lakukan prediksi
        prediction = model.predict(final_features)

        # Format hasil
        result = f'Prediksi: {prediction[0]}'

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
