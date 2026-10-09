import httpx
import time

def run_client(base_url="http://127.0.0.1:8000"):
    print(f"Connecting to AIFORGE Server at {base_url}...")
    
    # 1. Check health
    resp = httpx.get(f"{base_url}/health")
    print(f"Health: {resp.json()}")
    
    # 2. Wait for readiness
    for _ in range(5):
        resp = httpx.get(f"{base_url}/ready")
        if resp.status_code == 200:
            print("Server is READY.")
            break
        print("Waiting for server to load model...")
        time.sleep(1)
    else:
        print("Server not ready. Exiting.")
        return
        
    # 3. Get Model Metadata
    resp = httpx.get(f"{base_url}/v1/model")
    print(f"Model Metadata: {resp.json()}")
    
    # 4. Predict Single Example
    # Replace with valid data for your model
    single_payload = {"data": [[0.0] * 64]} 
    resp = httpx.post(f"{base_url}/v1/predict", json=single_payload)
    if resp.status_code == 200:
        print(f"Single Prediction: {resp.json()}")
    else:
        print(f"Prediction Error: {resp.text}")
        
    # 5. Predict Batch
    batch_payload = {"data": [[0.0] * 64, [1.0] * 64]} 
    resp = httpx.post(f"{base_url}/v1/predict", json=batch_payload)
    if resp.status_code == 200:
        print(f"Batch Prediction: {resp.json()}")
        
    # 6. Metrics
    resp = httpx.get(f"{base_url}/metrics")
    print(f"Metrics: {resp.json()}")

if __name__ == "__main__":
    run_client()
