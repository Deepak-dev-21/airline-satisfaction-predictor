import unittest

from fastapi.testclient import TestClient

import app as airline_app


VALID_PAYLOAD = {
    "Gender": "Male",
    "Customer_Type": "Loyal Customer",
    "Age": 35,
    "Type_of_Travel": "Business travel",
    "Class": "Business",
    "Flight_Distance": 1500,
    "Inflight_wifi_service": 3,
    "Departure_Arrival_time_convenient": 4,
    "Ease_of_Online_booking": 3,
    "Gate_location": 3,
    "Food_and_drink": 4,
    "Online_boarding": 4,
    "Seat_comfort": 4,
    "Inflight_entertainment": 3,
    "On_board_service": 4,
    "Leg_room_service": 3,
    "Baggage_handling": 4,
    "Checkin_service": 4,
    "Inflight_service": 4,
    "Cleanliness": 4,
    "Departure_Delay_in_Minutes": 10.0,
    "Arrival_Delay_in_Minutes": 5.0,
}


class AirlineApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(airline_app.app)

    def test_root_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("prediction-form", response.text)
        self.assertIn("Predict Satisfaction", response.text)
        self.assertIn('fetch("/predict"', response.text)
        self.assertIn('name="Gender"', response.text)
        self.assertIn("/docs", response.text)

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_predict_rejects_empty_payload(self):
        response = self.client.post("/predict", json={})

        self.assertEqual(response.status_code, 422)

    def test_predict_returns_valid_prediction(self):
        response = self.client.post("/predict", json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {"prediction", "result", "confidence"})
        self.assertIn(body["prediction"], {0, 1})
        self.assertIn(body["result"], {"Satisfied", "Neutral or Dissatisfied"})
        self.assertGreaterEqual(body["confidence"], 0)
        self.assertLessEqual(body["confidence"], 1)


if __name__ == "__main__":
    unittest.main()
