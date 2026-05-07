# Airline Passenger Satisfaction API

A FastAPI project that predicts airline passenger satisfaction using a saved machine learning model.

## Setup

```bash
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```bash
.\venv\Scripts\python.exe -m uvicorn app:app --reload
```

Open the user interface at:

```txt
http://127.0.0.1:8000/
```

Interactive API docs are available at:

```txt
http://127.0.0.1:8000/docs
```

## Test

```bash
.\venv\Scripts\python.exe -W error -m unittest discover -v
```
