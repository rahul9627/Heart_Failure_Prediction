from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load('heart_failure_model.pkl')
print("Model loaded successfully.")

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve input data from the form
            inputs = [
                float(request.form['height']),
                float(request.form['weight']),
                float(request.form['ap_hi']),
                float(request.form['ap_lo']),
                float(request.form['cholesterol']),
                float(request.form['gluc']),
                int(request.form['smoke']),
                int(request.form['alco']),
                int(request.form['active']),
                float(request.form['ageYears'])
            ]

            print("Inputs received:", inputs)

            # Convert inputs into a NumPy array for prediction
            inputs_array = np.array([inputs])
            print("Input array shape:", inputs_array.shape)

            # Make prediction
            prediction = model.predict(inputs_array)
            print("Prediction result:", prediction)

            # Interpret prediction result
            result = 'Heart Failure' if prediction[0] == 1 else 'Healthy Heart'
            return render_template('index.html', result=result)
        
        except Exception as e:
            print("Error during prediction:", str(e))
            return render_template('index.html', result='Error in prediction.')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)