import httpx
import time

def run_client(base_url="http://127.0.0.1:8000"):
    print(f"Connecting to LLM Server at {base_url}...")
    
    # 1. Check health
    resp = httpx.get(f"{base_url}/health")
    print(f"Health: {resp.json()}")
    
    # 2. Wait for readiness
    for _ in range(5):
        try:
            resp = httpx.get(f"{base_url}/ready")
            if resp.status_code == 200:
                print(f"Server is READY. Model ID: {resp.json()['model_id']}")
                break
        except Exception:
            pass
        print("Waiting for server to load LLM...")
        time.sleep(1)
    else:
        print("Server not ready. Exiting.")
        return
        
    # 3. Generate Text
    prompt = "def calculate_area(radius):"
    print(f"\nSending Prompt: '{prompt}'")
    
    payload = {
        "prompt": prompt,
        "max_new_tokens": 30,
        "temperature": 0.7
    }
    
    resp = httpx.post(f"{base_url}/v1/generate", json=payload)
    if resp.status_code == 200:
        print(f"\nGenerated Response:\n{resp.json()['text']}")
    else:
        print(f"Error: {resp.text}")

if __name__ == "__main__":
    run_client()
