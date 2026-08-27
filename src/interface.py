from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.predict import predict_fare
from src.predict import optional_tip

app = FastAPI()

# 1. Pydantic model matching your exact function signature
class FarePredictionInput(BaseModel):
    trip_duration: float
    passenger_count: int
    trip_distance: float
    pickup_borough: str
    dropoff_borough: str
    rate_category: str
    pickup_hour: int
    pickup_dayofweek: int
    tip_percentage: float = 0.0

# Import or define your predict_fare function here
# from your_model_module import predict_fare

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Taxi Fare Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 650px; margin: 30px auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
            .grid-container { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
            .form-group { margin-bottom: 10px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; font-size: 14px; }
            input, select { width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
            button { width: 100%; padding: 12px; background-color: #007bff; color: white; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; margin-top: 15px; grid-column: span 2; }
            button:hover { background-color: #0056b3; }
            #result { margin-top: 20px; font-size: 16px; text-align: center; line-height: 1.6; }
            .total-fare { font-size: 20px; font-weight: bold; color: #28a745; margin-top: 5px; }
        </style>
    </head>
    <body>
        <h2>Taxi Fare Predictor</h2>
        <div class="grid-container">
            <div class="form-group">
                <label for="trip_duration">Trip Duration (minutes)</label>
                <input type="number" id="trip_duration" step="0.1" value="34.0">
            </div>
            <div class="form-group">
                <label for="passenger_count">Passenger Count</label>
                <input type="number" id="passenger_count" value="2" min="1" max="6">
            </div>
            <div class="form-group">
                <label for="trip_distance">Trip Distance (miles)</label>
                <input type="number" id="trip_distance" step="0.1" value="8.3">
            </div>
            <div class="form-group">
                <label for="pickup_borough">Pickup Borough</label>
                <select id="pickup_borough">
                    <option value="Manhattan">Manhattan</option>
                    <option value="Brooklyn">Brooklyn</option>
                    <option value="Queens">Queens</option>
                    <option value="Bronx">Bronx</option>
                    <option value="Staten Island">Staten Island</option>
                </select>
            </div>
            <div class="form-group">
                <label for="dropoff_borough">Dropoff Borough</label>
                <select id="dropoff_borough">
                    <option value="Manhattan">Manhattan</option>
                    <option value="Brooklyn">Brooklyn</option>
                    <option value="Queens">Queens</option>
                    <option value="Bronx">Bronx</option>
                    <option value="Staten Island">Staten Island</option>
                </select>
            </div>
            <div class="form-group">
                <label for="rate_category">Rate Category</label>
                <select id="rate_category">
                    <option value="standard">standard</option>
                    <option value="JFK">JFK</option>
                    <option value="negotiated">negotiated</option>
                </select>
            </div>
            <div class="form-group">
                <label for="pickup_hour">Pickup Hour (0-23)</label>
                <input type="number" id="pickup_hour" min="0" max="23" value="11">
            </div>
            <div class="form-group">
                <label for="pickup_dayofweek">Day of Week (0=Mon, 6=Sun)</label>
                <input type="number" id="pickup_dayofweek" min="0" max="6" value="4">
            </div>
            <div class="form-group" style="grid-column: span 2;">
                <label for="tip_percentage">Tip Percentage (%)</label>
                <input type="number" id="tip_percentage" min="0" max="100" step="1" value="15">
            </div>
            <button onclick="sendData()">Estimate Fare</button>
        </div>

        <div id="result"></div>

        <script>
            async function sendData() {
                const resultDiv = document.getElementById('result');
                resultDiv.style.color = '#333';
                resultDiv.innerText = 'Calculating...';

                try {
                    const tipVal = parseFloat(document.getElementById('tip_percentage').value);
                    const payload = {
                        trip_duration: parseFloat(document.getElementById('trip_duration').value),
                        passenger_count: parseInt(document.getElementById('passenger_count').value),
                        trip_distance: parseFloat(document.getElementById('trip_distance').value),
                        pickup_borough: document.getElementById('pickup_borough').value,
                        dropoff_borough: document.getElementById('dropoff_borough').value,
                        rate_category: document.getElementById('rate_category').value,
                        pickup_hour: parseInt(document.getElementById('pickup_hour').value),
                        pickup_dayofweek: parseInt(document.getElementById('pickup_dayofweek').value),
                        tip_percentage: isNaN(tipVal) ? 0.0 : tipVal
                    };

                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        resultDiv.style.color = '#dc3545';
                        resultDiv.innerText = `Error: ${JSON.stringify(data.detail || data)}`;
                        return;
                    }

                    resultDiv.innerHTML = `
                        Base Fare: <strong>$${data.base_fare.toFixed(2)}</strong> | 
                        Tip (${payload.tip_percentage}%): <strong>$${data.tip_amount.toFixed(2)}</strong>
                        <div class="total-fare">Total: $${data.total_fare.toFixed(2)}</div>
                    `;
                } catch (err) {
                    resultDiv.style.color = '#dc3545';
                    resultDiv.innerText = `Network Error: ${err.message}`;
                }
            }
        </script>
    </body>
    </html>
    """

# 2. Directly unpack parameters into your function
@app.post("/predict")
def predict(data: FarePredictionInput):
    input_dict = data.model_dump()
    
    # Extract tip_percentage so it isn't passed into predict_fare()
    tip_percent = input_dict.pop("tip_percentage", 0.0)
    
    # Model prediction for base fare
    base_fare = float(predict_fare(**input_dict))
    
    # Tip and Total calculations
    tip_amount = optional_tip(fare=base_fare, tip_percentage=tip_percent)
    total_fare = round(base_fare + tip_amount, 2)
    
    return {
        "base_fare": round(base_fare, 2),
        "tip_amount": tip_amount,
        "total_fare": total_fare
    }