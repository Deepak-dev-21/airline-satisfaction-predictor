# Airline Satisfaction Predictor

A FastAPI-based machine learning web application that predicts whether an airline passenger is satisfied or neutral/dissatisfied based on travel details, service ratings, and delay information.

## Features

- Passenger satisfaction prediction
- Simple web user interface
- FastAPI backend
- Interactive Swagger API docs
- Machine learning model loaded from saved `.pkl` files
- Health check endpoint
- Unit tests included

## Tech Stack

- Python
- FastAPI
- Pandas
- Scikit-learn
- Joblib
- HTML, CSS, JavaScript

## Project Structure

```txt
airline/
├── app.py
├── model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
├── tests/
│   ├── __init__.py
│   └── test_app.py
└── README.md
