import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure package src is importable
sys.path.append(os.path.abspath("."))

# Set model path env before importing app
os.environ["MODEL_PATH"] = "models/fraud_model_pipeline.joblib"

from api.app import app

class TestFraudAPI(unittest.TestCase):
    def test_health_endpoint(self):
        """Verifies that GET /health returns 200 and indicates model is loaded."""
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "healthy")
            self.assertTrue(data["model_loaded"])
        
    def test_predict_single_endpoint(self):
        """Verifies that POST /predict correctly classifies a sample transaction."""
        with TestClient(app) as client:
            # Legitimate transaction input (centered around 0)
            legit_tx = {
                "Time": 100.0,
                "Amount": 15.0,
                **{f"V{i}": 0.0 for i in range(1, 29)}
            }
            
            response = client.post("/predict", json=legit_tx)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("is_fraud", data)
            self.assertIn("probability", data)
            self.assertFalse(data["is_fraud"]) # Expect legitimate (Class 0)
            
            # Fraudulent transaction input (V10, V12, V14, V17 shifted low)
            fraud_tx = {
                "Time": 200.0,
                "Amount": 2500.0,
                **{f"V{i}": 0.0 for i in range(1, 29)}
            }
            fraud_tx["V10"] = -8.0
            fraud_tx["V12"] = -8.0
            fraud_tx["V14"] = -8.0
            fraud_tx["V17"] = -8.0
            
            response = client.post("/predict", json=fraud_tx)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["is_fraud"]) # Expect fraud (Class 1)
            self.assertGreater(data["probability"], 0.5)

    def test_predict_batch_endpoint(self):
        """Verifies that POST /predict/batch processes a batch of transactions."""
        with TestClient(app) as client:
            batch_tx = [
                # Legitimate
                {
                    "Time": 500.0, "Amount": 10.0,
                    **{f"V{i}": 0.0 for i in range(1, 29)}
                },
                # Fraudulent
                {
                    "Time": 600.0, "Amount": 5000.0,
                    **{f"V{i}": 0.0 for i in range(1, 29)}
                }
            ]
            batch_tx[1]["V10"] = -8.0
            batch_tx[1]["V12"] = -8.0
            batch_tx[1]["V14"] = -8.0
            batch_tx[1]["V17"] = -8.0
            
            response = client.post("/predict/batch", json=batch_tx)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("predictions", data)
            self.assertEqual(len(data["predictions"]), 2)
            self.assertFalse(data["predictions"][0]["is_fraud"])
            self.assertTrue(data["predictions"][1]["is_fraud"])

if __name__ == "__main__":
    unittest.main()
