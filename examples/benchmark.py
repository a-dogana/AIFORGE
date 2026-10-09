import time
import httpx
import numpy as np

def benchmark():
    base_url = "http://127.0.0.1:8000"
    
    print("Waiting for server...")
    for _ in range(10):
        try:
            if httpx.get(f"{base_url}/ready").status_code == 200:
                break
        except:
            pass
        time.sleep(1)
        
    print("Benchmarking Direct HTTP Request...")
    # Student MLP Digits model has input 64
    batch = np.random.randn(32, 64).tolist()
    
    # Warmup
    for _ in range(5):
        httpx.post(f"{base_url}/v1/predict", json={"data": batch})
        
    latencies = []
    for _ in range(100):
        start = time.time()
        resp = httpx.post(f"{base_url}/v1/predict", json={"data": batch})
        assert resp.status_code == 200
        latencies.append((time.time() - start) * 1000)
        
    print(f"HTTP Latency (N=100, Batch=32):")
    print(f"Median: {np.median(latencies):.2f} ms")
    print(f"P95:    {np.percentile(latencies, 95):.2f} ms")

if __name__ == "__main__":
    benchmark()
