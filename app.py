from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from datetime import datetime 

app = Flask(__name__)

# Load the trained model
with open('credit_card_fraud_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict(): 
    try:
        data = request.get_json()
        credit_card_number = data['credit_card_number']
    
        # Convert the transaction time (HH:MM:SS) to total seconds since midnight
        transaction_time_str = data['transaction_time']
        transaction_time = datetime.strptime(transaction_time_str, '%H:%M:%S')
        total_seconds = transaction_time.hour * 3600 + transaction_time.minute * 60 + transaction_time.second

        transaction_amount = float(data['transaction_amount'])
        input_data = np.array([[credit_card_number, total_seconds, transaction_amount]])
        
        prediction = model.predict(input_data) 
        result = "Fraudulent Transaction Detected" if prediction[0] == 1 else "Transaction is Safe"
        
        return jsonify({'message': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
