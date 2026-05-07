from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")


app = FastAPI(
    title="Airline Passenger Satisfaction API",
    description="Predicts whether a passenger is satisfied or not using Random Forest.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PassengerInput(BaseModel):
    Gender: str = Field(..., json_schema_extra={"example": "Male"}, description="Male or Female")
    Customer_Type: str = Field(
        ...,
        json_schema_extra={"example": "Loyal Customer"},
        description="Loyal Customer or disloyal Customer",
    )
    Age: int = Field(..., json_schema_extra={"example": 35}, ge=1, le=120)
    Type_of_Travel: str = Field(
        ...,
        json_schema_extra={"example": "Business travel"},
        description="Business travel or Personal Travel",
    )
    Class: str = Field(
        ...,
        json_schema_extra={"example": "Business"},
        description="Business, Eco, or Eco Plus",
    )
    Flight_Distance: int = Field(..., json_schema_extra={"example": 1500}, ge=0)
    Inflight_wifi_service: int = Field(..., json_schema_extra={"example": 3}, ge=0, le=5)
    Departure_Arrival_time_convenient: int = Field(
        ...,
        json_schema_extra={"example": 4},
        ge=0,
        le=5,
    )
    Ease_of_Online_booking: int = Field(..., json_schema_extra={"example": 3}, ge=0, le=5)
    Gate_location: int = Field(..., json_schema_extra={"example": 3}, ge=0, le=5)
    Food_and_drink: int = Field(..., json_schema_extra={"example": 4}, ge=0, le=5)
    Online_boarding: int = Field(..., json_schema_extra={"example": 4}, ge=0, le=5)
    Seat_comfort: int = Field(..., json_schema_extra={"example": 4}, ge=0, le=5)
    Inflight_entertainment: int = Field(..., json_schema_extra={"example": 3}, ge=0, le=5)
    On_board_service: int = Field(..., json_schema_extra={"example": 4}, ge=0, le=5)
    Leg_room_service: int = Field(..., json_schema_extra={"example": 3}, ge=0, le=5)
    Baggage_handling: int = Field(..., json_schema_extra={"example": 4}, ge=0, le=5)
    Checkin_service: int = Field(..., json_schema_extra={"example": 4}, ge=0, le=5)
    Inflight_service: int = Field(..., json_schema_extra={"example": 4}, ge=0, le=5)
    Cleanliness: int = Field(..., json_schema_extra={"example": 4}, ge=0, le=5)
    Departure_Delay_in_Minutes: float = Field(..., json_schema_extra={"example": 10.0}, ge=0)
    Arrival_Delay_in_Minutes: float = Field(..., json_schema_extra={"example": 5.0}, ge=0)


class PredictionOutput(BaseModel):
    prediction: int
    result: str
    confidence: float


def preprocess(data: PassengerInput) -> pd.DataFrame:
    """Convert raw input into a model-ready DataFrame."""
    raw = {
        "Age": data.Age,
        "Flight Distance": data.Flight_Distance,
        "Inflight wifi service": data.Inflight_wifi_service,
        "Departure/Arrival time convenient": data.Departure_Arrival_time_convenient,
        "Ease of Online booking": data.Ease_of_Online_booking,
        "Gate location": data.Gate_location,
        "Food and drink": data.Food_and_drink,
        "Online boarding": data.Online_boarding,
        "Seat comfort": data.Seat_comfort,
        "Inflight entertainment": data.Inflight_entertainment,
        "On-board service": data.On_board_service,
        "Leg room service": data.Leg_room_service,
        "Baggage handling": data.Baggage_handling,
        "Checkin service": data.Checkin_service,
        "Inflight service": data.Inflight_service,
        "Cleanliness": data.Cleanliness,
        "Departure Delay in Minutes": data.Departure_Delay_in_Minutes,
        "Arrival Delay in Minutes": data.Arrival_Delay_in_Minutes,
        "Gender": data.Gender,
        "Customer Type": data.Customer_Type,
        "Type of Travel": data.Type_of_Travel,
        "Class": data.Class,
    }

    df = pd.DataFrame([raw])
    categorical_columns = ["Gender", "Customer Type", "Type of Travel", "Class"]
    encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)
    aligned = encoded.reindex(columns=feature_columns, fill_value=0)
    scaled_values = scaler.transform(aligned)

    return pd.DataFrame(scaled_values, columns=feature_columns)


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Airline Passenger Satisfaction Predictor</title>
        <style>
            * {
                box-sizing: border-box;
            }
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                min-height: 100vh;
                background: #eef2f7;
                color: #172033;
            }
            main {
                width: min(1180px, calc(100% - 32px));
                margin: 0 auto;
                padding: 28px 0;
            }
            header {
                display: flex;
                align-items: flex-end;
                justify-content: space-between;
                gap: 16px;
                margin-bottom: 18px;
            }
            h1 {
                margin: 0;
                font-size: clamp(24px, 3vw, 34px);
                line-height: 1.15;
            }
            .subtitle {
                margin: 8px 0 0;
                max-width: 680px;
                color: #566276;
                line-height: 1.5;
            }
            .docs-link {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-height: 40px;
                padding: 10px 14px;
                color: #0f3f7a;
                background: #dceafe;
                border: 1px solid #b9d5fb;
                border-radius: 8px;
                font-weight: 700;
                text-decoration: none;
                white-space: nowrap;
            }
            .layout {
                display: grid;
                grid-template-columns: minmax(0, 1fr) 320px;
                gap: 18px;
                align-items: start;
            }
            form,
            .result-panel {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                box-shadow: 0 14px 30px rgba(23, 32, 51, 0.08);
            }
            form {
                padding: 20px;
            }
            fieldset {
                margin: 0 0 18px;
                padding: 0;
                border: 0;
            }
            fieldset:last-of-type {
                margin-bottom: 0;
            }
            legend {
                width: 100%;
                margin-bottom: 12px;
                padding-bottom: 8px;
                border-bottom: 1px solid #edf0f4;
                color: #172033;
                font-size: 16px;
                font-weight: 700;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 12px;
            }
            label {
                display: grid;
                gap: 6px;
                color: #445064;
                font-size: 13px;
                font-weight: 700;
            }
            input,
            select {
                width: 100%;
                min-height: 40px;
                padding: 9px 10px;
                border: 1px solid #ccd4df;
                border-radius: 6px;
                background: #ffffff;
                color: #172033;
                font: inherit;
            }
            input:focus,
            select:focus {
                border-color: #2563eb;
                outline: 3px solid #dbeafe;
            }
            .actions {
                display: flex;
                align-items: center;
                justify-content: flex-end;
                gap: 10px;
                margin-top: 18px;
                padding-top: 18px;
                border-top: 1px solid #edf0f4;
            }
            button {
                min-height: 42px;
                padding: 10px 16px;
                border: 0;
                border-radius: 8px;
                font: inherit;
                font-weight: 700;
                cursor: pointer;
            }
            button[type="reset"] {
                color: #354158;
                background: #e9eef5;
            }
            button[type="submit"] {
                color: #ffffff;
                background: #1d4ed8;
            }
            button:disabled {
                cursor: wait;
                opacity: 0.72;
            }
            .result-panel {
                position: sticky;
                top: 16px;
                padding: 18px;
            }
            .result-panel h2 {
                margin: 0 0 12px;
                font-size: 18px;
            }
            .status-box {
                min-height: 168px;
                padding: 16px;
                border: 1px dashed #cfd8e5;
                border-radius: 8px;
                background: #f8fafc;
            }
            .status-box.success {
                border-color: #86efac;
                background: #f0fdf4;
            }
            .status-box.error {
                border-color: #fca5a5;
                background: #fff1f2;
            }
            .result-label {
                margin: 0;
                color: #566276;
                font-size: 13px;
                font-weight: 700;
                text-transform: uppercase;
            }
            .result-value {
                margin: 8px 0 14px;
                color: #122034;
                font-size: 28px;
                font-weight: 800;
                line-height: 1.15;
            }
            .confidence {
                display: grid;
                gap: 8px;
            }
            .meter {
                height: 10px;
                overflow: hidden;
                border-radius: 999px;
                background: #dbe3ef;
            }
            .meter span {
                display: block;
                width: 0%;
                height: 100%;
                background: #16a34a;
                transition: width 180ms ease;
            }
            .meta {
                margin: 14px 0 0;
                color: #566276;
                font-size: 13px;
                line-height: 1.45;
            }
            @media (max-width: 900px) {
                header,
                .actions {
                    align-items: stretch;
                    flex-direction: column;
                }
                .layout {
                    grid-template-columns: 1fr;
                }
                .result-panel {
                    position: static;
                }
                .grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
            @media (max-width: 560px) {
                main {
                    width: min(100% - 20px, 1180px);
                    padding: 16px 0;
                }
                form {
                    padding: 14px;
                }
                .grid {
                    grid-template-columns: 1fr;
                }
                button,
                .docs-link {
                    width: 100%;
                }
            }
        </style>
    </head>
    <body>
        <main>
            <header>
                <div>
                    <h1>Airline Passenger Satisfaction Predictor</h1>
                    <p class="subtitle">Enter passenger and flight details, then generate a satisfaction prediction.</p>
                </div>
                <a class="docs-link" href="/docs">API Docs</a>
            </header>

            <div class="layout">
                <form id="prediction-form">
                    <fieldset>
                        <legend>Passenger</legend>
                        <div class="grid">
                            <label>
                                Gender
                                <select name="Gender" required>
                                    <option>Male</option>
                                    <option>Female</option>
                                </select>
                            </label>
                            <label>
                                Customer Type
                                <select name="Customer_Type" required>
                                    <option>Loyal Customer</option>
                                    <option>disloyal Customer</option>
                                </select>
                            </label>
                            <label>
                                Age
                                <input name="Age" type="number" min="1" max="120" value="35" required>
                            </label>
                            <label>
                                Travel Type
                                <select name="Type_of_Travel" required>
                                    <option>Business travel</option>
                                    <option>Personal Travel</option>
                                </select>
                            </label>
                            <label>
                                Class
                                <select name="Class" required>
                                    <option>Business</option>
                                    <option>Eco</option>
                                    <option>Eco Plus</option>
                                </select>
                            </label>
                            <label>
                                Flight Distance
                                <input name="Flight_Distance" type="number" min="0" value="1500" required>
                            </label>
                        </div>
                    </fieldset>

                    <fieldset>
                        <legend>Service Ratings</legend>
                        <div class="grid">
                            <label>
                                Inflight Wifi
                                <input name="Inflight_wifi_service" type="number" min="0" max="5" value="3" required>
                            </label>
                            <label>
                                Time Convenient
                                <input name="Departure_Arrival_time_convenient" type="number" min="0" max="5" value="4" required>
                            </label>
                            <label>
                                Online Booking
                                <input name="Ease_of_Online_booking" type="number" min="0" max="5" value="3" required>
                            </label>
                            <label>
                                Gate Location
                                <input name="Gate_location" type="number" min="0" max="5" value="3" required>
                            </label>
                            <label>
                                Food and Drink
                                <input name="Food_and_drink" type="number" min="0" max="5" value="4" required>
                            </label>
                            <label>
                                Online Boarding
                                <input name="Online_boarding" type="number" min="0" max="5" value="4" required>
                            </label>
                            <label>
                                Seat Comfort
                                <input name="Seat_comfort" type="number" min="0" max="5" value="4" required>
                            </label>
                            <label>
                                Entertainment
                                <input name="Inflight_entertainment" type="number" min="0" max="5" value="3" required>
                            </label>
                            <label>
                                On-board Service
                                <input name="On_board_service" type="number" min="0" max="5" value="4" required>
                            </label>
                            <label>
                                Leg Room
                                <input name="Leg_room_service" type="number" min="0" max="5" value="3" required>
                            </label>
                            <label>
                                Baggage Handling
                                <input name="Baggage_handling" type="number" min="0" max="5" value="4" required>
                            </label>
                            <label>
                                Check-in Service
                                <input name="Checkin_service" type="number" min="0" max="5" value="4" required>
                            </label>
                            <label>
                                Inflight Service
                                <input name="Inflight_service" type="number" min="0" max="5" value="4" required>
                            </label>
                            <label>
                                Cleanliness
                                <input name="Cleanliness" type="number" min="0" max="5" value="4" required>
                            </label>
                        </div>
                    </fieldset>

                    <fieldset>
                        <legend>Delays</legend>
                        <div class="grid">
                            <label>
                                Departure Delay
                                <input name="Departure_Delay_in_Minutes" type="number" min="0" step="0.1" value="10" required>
                            </label>
                            <label>
                                Arrival Delay
                                <input name="Arrival_Delay_in_Minutes" type="number" min="0" step="0.1" value="5" required>
                            </label>
                        </div>
                    </fieldset>

                    <div class="actions">
                        <button type="reset">Reset</button>
                        <button id="submit-button" type="submit">Predict Satisfaction</button>
                    </div>
                </form>

                <aside class="result-panel" aria-live="polite">
                    <h2>Prediction</h2>
                    <div id="result-box" class="status-box">
                        <p class="result-label">Status</p>
                        <p id="result-value" class="result-value">Ready</p>
                        <div class="confidence">
                            <span id="confidence-text">Confidence: --</span>
                            <div class="meter"><span id="confidence-meter"></span></div>
                        </div>
                        <p id="result-meta" class="meta">Submit the form to see the result.</p>
                    </div>
                </aside>
            </div>
        </main>

        <script>
            const form = document.getElementById("prediction-form");
            const submitButton = document.getElementById("submit-button");
            const resultBox = document.getElementById("result-box");
            const resultValue = document.getElementById("result-value");
            const resultMeta = document.getElementById("result-meta");
            const confidenceText = document.getElementById("confidence-text");
            const confidenceMeter = document.getElementById("confidence-meter");

            const integerFields = new Set([
                "Age",
                "Flight_Distance",
                "Inflight_wifi_service",
                "Departure_Arrival_time_convenient",
                "Ease_of_Online_booking",
                "Gate_location",
                "Food_and_drink",
                "Online_boarding",
                "Seat_comfort",
                "Inflight_entertainment",
                "On_board_service",
                "Leg_room_service",
                "Baggage_handling",
                "Checkin_service",
                "Inflight_service",
                "Cleanliness"
            ]);
            const floatFields = new Set([
                "Departure_Delay_in_Minutes",
                "Arrival_Delay_in_Minutes"
            ]);

            function setLoading(isLoading) {
                submitButton.disabled = isLoading;
                submitButton.textContent = isLoading ? "Predicting..." : "Predict Satisfaction";
            }

            function buildPayload() {
                const data = new FormData(form);
                const payload = {};

                for (const [key, value] of data.entries()) {
                    if (integerFields.has(key)) {
                        payload[key] = parseInt(value, 10);
                    } else if (floatFields.has(key)) {
                        payload[key] = parseFloat(value);
                    } else {
                        payload[key] = value;
                    }
                }

                return payload;
            }

            function showResult(data) {
                const confidence = Math.round(data.confidence * 100);
                resultBox.className = "status-box success";
                resultValue.textContent = data.result;
                confidenceText.textContent = `Confidence: ${confidence}%`;
                confidenceMeter.style.width = `${confidence}%`;
                resultMeta.textContent = `Prediction code: ${data.prediction}`;
            }

            function showError(message) {
                resultBox.className = "status-box error";
                resultValue.textContent = "Error";
                confidenceText.textContent = "Confidence: --";
                confidenceMeter.style.width = "0%";
                resultMeta.textContent = message;
            }

            form.addEventListener("submit", async (event) => {
                event.preventDefault();
                setLoading(true);

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify(buildPayload())
                    });
                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.detail || "Prediction failed.");
                    }

                    showResult(data);
                } catch (error) {
                    showError(error.message);
                } finally {
                    setLoading(false);
                }
            });

            form.addEventListener("reset", () => {
                resultBox.className = "status-box";
                resultValue.textContent = "Ready";
                confidenceText.textContent = "Confidence: --";
                confidenceMeter.style.width = "0%";
                resultMeta.textContent = "Submit the form to see the result.";
            });
        </script>
    </body>
    </html>
    """


@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput):
    try:
        processed = preprocess(passenger)
        prediction = int(model.predict(processed)[0])
        prediction_index = list(model.classes_).index(prediction)
        confidence = float(model.predict_proba(processed)[0][prediction_index])
        result = "Satisfied" if prediction == 1 else "Neutral or Dissatisfied"

        return PredictionOutput(
            prediction=prediction,
            result=result,
            confidence=round(confidence, 4),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok"}
