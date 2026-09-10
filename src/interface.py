from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.predict import predict_fare, optional_tip, get_zone_name_to_id

app = FastAPI()

class FarePredictionInput(BaseModel):
    passenger_count: int
    pickup_zone: str
    dropoff_zone: str
    rate_category: str
    pickup_hour: int
    pickup_dayofweek: int
    tip_percentage: float = 0.0


@app.get("/zones")
def get_zones():
    zone_names = sorted(get_zone_name_to_id().keys())
    return {"zones": zone_names}


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
                <label for="passenger_count">Passenger Count</label>
                <input type="number" id="passenger_count" value="2" min="1" max="6">
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
                <label for="pickup_zone">Pickup Zone</label>
                <select id="pickup_zone"><option value="">Loading...</option></select>
            </div>
            <div class="form-group">
                <label for="dropoff_zone">Dropoff Zone</label>
                <select id="dropoff_zone"><option value="">Loading...</option></select>
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
            async function loadZones() {
                const response = await fetch('/zones');
                const data = await response.json();
                const pickupSelect = document.getElementById('pickup_zone');
                const dropoffSelect = document.getElementById('dropoff_zone');

                pickupSelect.innerHTML = '';
                dropoffSelect.innerHTML = '';

                data.zones.forEach(zone => {
                    const option1 = document.createElement('option');
                    option1.value = zone;
                    option1.textContent = zone;
                    pickupSelect.appendChild(option1);

                    const option2 = document.createElement('option');
                    option2.value = zone;
                    option2.textContent = zone;
                    dropoffSelect.appendChild(option2);
                });
            }

            window.onload = loadZones;

            async function sendData() {
                const resultDiv = document.getElementById('result');
                resultDiv.style.color = '#333';
                resultDiv.innerText = 'Calculating...';

                try {
                    const tipVal = parseFloat(document.getElementById('tip_percentage').value);
                    const payload = {
                        passenger_count: parseInt(document.getElementById('passenger_count').value),
                        pickup_zone: document.getElementById('pickup_zone').value,
                        dropoff_zone: document.getElementById('dropoff_zone').value,
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


@app.post("/predict")
def predict(data: FarePredictionInput):
    input_dict = data.model_dump()

    tip_percent = input_dict.pop("tip_percentage", 0.0)

    try:
        base_fare = float(predict_fare(**input_dict))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    tip_amount = optional_tip(fare=base_fare, tip_percentage=tip_percent)
    total_fare = round(base_fare + tip_amount, 2)

    return {
        "base_fare": round(base_fare, 2),
        "tip_amount": tip_amount,
        "total_fare": total_fare
    }