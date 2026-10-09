import time
import httpx
import numpy as np
import sys

def benchmark_llm():
    base_url = "http://127.0.0.1:8001"
    
    print("Waiting for LLM server...")
    for _ in range(15):
        try:
            if httpx.get(f"{base_url}/ready").status_code == 200:
                break
        except:
            pass
        time.sleep(1)
        
    print("Benchmarking Direct HTTP Generation...")
    
    payload = {
        "prompt": "def benchmark():",
        "max_new_tokens": 20,
        "temperature": 0.0
    }
    
    # Warmup
    httpx.post(f"{base_url}/v1/generate", json=payload)
        
    latencies = []
    for _ in range(10):
        start = time.time()
        resp = httpx.post(f"{base_url}/v1/generate", json=payload)
        assert resp.status_code == 200
        latencies.append((time.time() - start) * 1000)
        
    print(f"HTTP Latency (N=10, Tokens=20):")
    print(f"Median: {np.median(latencies):.2f} ms")
    print(f"P95:    {np.percentile(latencies, 95):.2f} ms")

if __name__ == "__main__":
    benchmark_llm()
