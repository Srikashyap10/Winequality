from flask import Flask, render_template, request, redirect, url_for, session
from flask import Flask, render_template, send_from_directory
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = 'Naanu_bekku' 

# Load the PCA, scaler, and quality prediction model
with open("pca.pkl", 'rb') as f:
    pca = joblib.load(f)
with open("scaler_11_features.pkl", 'rb') as f:
    scaler = joblib.load(f)
with open("wine_quality_prediction.pkl", 'rb') as f:
    model = joblib.load(f)

# Load the age prediction model and scaler
with open("age_model.pkl", 'rb') as f:
    age_model = joblib.load(f)
with open("age_scaler.pkl", 'rb') as f:
    age_scaler = joblib.load(f)

# User credentials 
users = {
    'admin': 'password123',
    'user1': 'pass123',
    'user2': 'abc456'
}

@app.route('/')
def home():
    if 'username' in session:
        return render_template('home.html')
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', message='Invalid credentials. Please try again.')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)
        
        # Apply transformations for quality prediction
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        # Predict quality
        quality_prediction = model.predict(features_pca)
        quality_result = "Good Quality Wine" if quality_prediction[0] == 1 else "Bad Quality Wine"
        
        # Apply transformations for age prediction
        age_features_scaled = age_scaler.transform(features)
        
        # Predict age
        age_prediction = age_model.predict(age_features_scaled)[0]
        
        if quality_prediction[0] == 1:
            reason = 'The wine is good due to balanced acidity, adequate sugar levels, and optimal sulphate and alcohol content.'
        else:
            reason = 'The wine is bad due to imbalanced acidity, inadequate sugar levels, or suboptimal sulphate and alcohol content.'
        
        return render_template('predict.html', prediction_text=f'Prediction: {quality_result}', age_text=f'Age: {round(age_prediction, 2)} years', reason_text=f'Reason: {reason}')

    return render_template('predict.html')

@app.route('/assets/<path:filename>')
def assets(filename):
    return send_from_directory('assets', filename)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == "__main__":
    app.run(debug=True)