import os
import requests
import urllib.parse

LOCAL_DOCS_DIR = "/app/docs"
CONTAINER_DOCS_DIR = "/app/docs"
API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/ingest")

def ingest_all():
    if not os.path.exists(LOCAL_DOCS_DIR):
        print(f"❌ Error: Directory '{LOCAL_DOCS_DIR}' not found.")
        return
    files = [f for f in os.listdir(LOCAL_DOCS_DIR) if f.endswith(".txt")]
    
    if not files:
        print("⚠️ No .txt files found in ./docs")
        return
    print(f"🚀 Found {len(files)} files. Starting ingestion...\n")
    for filename in files:
        file_path_inside_container = f"{CONTAINER_DOCS_DIR}/{filename}"
        print(f"Processing: {filename}...")
        
        try:
            # Send POST request to your API
            response = requests.post(API_URL, params={"file_path": file_path_inside_container})
            
            if response.status_code == 200:
                print(f"   ✅ Success: {response.json().get('message')}")
            else:
                print(f"   ❌ Failed: {response.text}")
                
        except Exception as e:
            print(f"   🔥 Error: Could not connect to API. Is it running? {e}")

if __name__ == "__main__":
    ingest_all()