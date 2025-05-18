import subprocess
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)
app.secret_key = b'\xd2\x11!|\xb8}M\x94\xa2\x1d.\xcepKy\x14\xe1\x85\xdf\xbbo\t|H'

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["chatbot-db"]
user_collection = db["user-db"]

# Load the trained model and tokenizer
model_path = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Ensure 'username' is a unique field (acting as the primary key)
user_collection.create_index("username", unique=True)

def run_streamlit_app():
    # Run the Streamlit app in a separate process
    subprocess.Popen(["streamlit", "run", "main_exercise_tracker.py"])

@app.route('/')
def login():
    return render_template('login.html', error=None)

@app.route('/login', methods=['POST'])
def login_post():
    username = request.form['username']
    password = request.form['password']
    user = user_collection.find_one({"username": username})
    if user and user['password'] == password:
        session['username'] = username
        return redirect(url_for('dashboard'))
    else:
        error = "Invalid username or password"
        return render_template('login.html', error=error)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form['fullname']
        username = request.form['username']
        password = request.form['password']
        gender = request.form['gender']
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        terms = request.form.get('terms')

        if not terms:
            flash("Please agree to the Terms and Conditions.")
            return render_template('signup.html')

        # Calculate BMI and category
        bmi, weight_category = calculate_bmi(weight, height)

        # Store the new user in the database
        new_user = {
            "fullname": fullname,
            "username": username,
            "password": password,
            "gender": gender,
            "weight": weight,
            "height": height,
            "bmi": bmi,
            "weight_category": weight_category  # Use the correct field name
        }

        try:
            user_collection.insert_one(new_user)
            session['username'] = username
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash("Username already exists or an error occurred.")
            return render_template('signup.html')

    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        # Start the Streamlit app when the dashboard is accessed
        run_streamlit_app()
        return render_template('dashboard.html', username=session['username'])
    else:
        return redirect(url_for('login'))

@app.route('/chatbot')
def chatbot():
    if 'username' in session:
        return render_template('chatbot_ui.html', username=session['username'])
    else:
        return redirect(url_for('login'))

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    username = session.get('username')
    user = user_collection.find_one({"username": username})

    if user:
        gender = user['gender']
        weight_category = user['weight_category']  # Use the correct field name
        input_text = f"Question: {user_input}\nCategory: {gender}_{weight_category}"

        inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_length=32, num_beams=4, early_stopping=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"response": response})
    else:
        return jsonify({"response": "Error: User not found."})

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

def calculate_bmi(weight, height):
    bmi = round(weight / (height ** 2), 2)
    if bmi < 18.5:
        weight_category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        weight_category = "Normal weight"
    elif 25 <= bmi < 29.9:
        weight_category = "Overweight"
    else:
        weight_category = "Obese"
    return bmi, weight_category

if __name__ == '__main__':
    app.run(debug=True)
