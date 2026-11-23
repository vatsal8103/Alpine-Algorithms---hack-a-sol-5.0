import json
import urllib.request
import time

def verify():
    time.sleep(1.0)
    print("--- 1. Testing /vlm/status ---")
    st = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/vlm/status").read().decode())
    print("VLM Status:", json.dumps(st, indent=2))

    print("\n--- 2. Testing /vlm/free_vram ---")
    req = urllib.request.Request("http://127.0.0.1:8000/vlm/free_vram", data=b"{}", headers={"Content-Type": "application/json"})
    free_res = json.loads(urllib.request.urlopen(req).read().decode())
    print("Free VRAM Result:", json.dumps(free_res, indent=2))

    print("\n--- 3. Testing /cases/case-4e54299f/vlm_clarify ---")
    req_vlm = urllib.request.Request("http://127.0.0.1:8000/cases/case-4e54299f/vlm_clarify?track_id=1", data=b"{}", headers={"Content-Type": "application/json"})
    clarify_res = json.loads(urllib.request.urlopen(req_vlm).read().decode())
    print("Clarify Result:", json.dumps(clarify_res, indent=2))

if __name__ == "__main__":
    verify()
