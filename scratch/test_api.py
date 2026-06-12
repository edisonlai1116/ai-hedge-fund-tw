import requests
import json

url = "http://localhost:8000/sentiment/podcast-scan"
payload = {
    "feed_url": "https://feeds.soundon.fm/podcasts/954689a5-3096-43a4-a80b-7810b219cef3.xml",
    "source_name": "Gooaye"
}

print("Sending POST request to /sentiment/podcast-scan...")
try:
    response = requests.post(url, json=payload, timeout=20)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
except Exception as e:
    print(f"HTTP Request failed: {e}")
