import os

# Allow the FastAPI TestClient host ("testserver") through the dashboard's local-only guard.
os.environ.setdefault("AIFORGE_TESTING", "1")
