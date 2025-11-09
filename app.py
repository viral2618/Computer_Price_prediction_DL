from flask import Flask, request, render_template, redirect, url_for, session, flash
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
app.secret_key = 'my_secret'

model = load_model('computer_price_model.h5')
scaler = joblib.load('Scaler.pkl')

@app.route('/', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        name = request.form.get('name')
        password = request.form.get('password')
        
        if name == 'viral' and password == '1234':
            session['name'] = name
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials. Please try again.', 'warning')
            return redirect(url_for('login'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('name', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))


@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'name' not in session:
        flash('please login first', 'info')
        return redirect(url_for('login'))
    
    prediction = None  # define upfront
    
    if request.method == 'POST':
        try:
            features = [
                request.form['device_type'],
                request.form['brand'],
                request.form['model'],
                request.form['release_year'],
                request.form['os'],
                request.form['cpu_brand'],
                request.form['cpu_model'],
                request.form['cpu_tier'],
                request.form['cpu_cores'],
                request.form['cpu_threads'],
                request.form['cpu_base_ghz'],
                request.form['cpu_boost_ghz'],
                request.form['gpu_brand'],
                request.form['gpu_model'],
                request.form['gpu_tier'],
                request.form['vram_gb'],
                request.form['ram_gb'],
                request.form['storage_gb'],
                request.form['storage_drive_count'],
                request.form['display_type'],
                request.form['display_size_in'],
                request.form['resolution'],
                request.form['refresh_hz'],
                request.form['battery_wh'],
                request.form['charger_watts']
            ]

            # ✅ Extract numeric part only (from cpu_cores onward)
            numeric_features = list(map(float, [
                request.form['cpu_cores'],
                request.form['cpu_threads'],
                request.form['cpu_base_ghz'],
                request.form['cpu_boost_ghz'],
                request.form['vram_gb'],
                request.form['ram_gb'],
                request.form['storage_gb'],
                request.form['storage_drive_count'],
                request.form['display_size_in'],
                request.form['refresh_hz'],
                request.form['battery_wh'],
                request.form['charger_watts']
            ]))

            # ⚠️ Pad dummy zeros to match 25 features
            while len(numeric_features) < 25:
                numeric_features.append(0.0)

            scaled_data = scaler.transform([numeric_features])
            prediction = model.predict(scaled_data)[0][0]

        except Exception as e:
            flash(f"Error: {str(e)}", 'danger')

    return render_template('home.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
