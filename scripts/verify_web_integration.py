import json
import urllib.request

def main():
    print("========================================")
    print("TRACE-X Web Dashboard Verification")
    print("========================================")

    # 1. HTML Verification
    html = urllib.request.urlopen("http://127.0.0.1:8000/").read().decode("utf-8")
    checks = {
        "test1.mp4 Preset Option": "test1.mp4" in html,
        "Recent Cases Selector": "recentCasesSelect" in html,
        "VLM Button (btnVlmClarify)": "btnVlmClarify" in html,
        "Auto-Restore Function": "tryRestoreLastCase" in html,
        "Qwen2-VL Result Container": "vlmClarifyResult" in html,
    }
    for k, v in checks.items():
        print(f"[{'✓' if v else '✗'}] {k}")

    # 2. Latest & Recent Cases API
    latest = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/cases/latest").read().decode())
    print(f"[✓] Latest Case: {latest['case_id']} ({latest['clip_name']} - Status: {latest['status']})")

    recent = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/cases").read().decode())
    print(f"[✓] Recent Cases Available: {len(recent)}")

    # 3. VLM Clarify API on case-4e54299f (test1.mp4)
    req = urllib.request.Request(
        "http://127.0.0.1:8000/cases/case-4e54299f/vlm_clarify?track_id=1",
        data=b"{}",
        headers={"Content-Type": "application/json"}
    )
    vlm_res = json.loads(urllib.request.urlopen(req).read().decode())
    print(f"[✓] VLM Clarify API Endpoint: {vlm_res.get('status')}")
    print(f"    - Model: {vlm_res.get('vlm_model')}")
    print(f"    - Crop: {vlm_res.get('plate_crop')}")
    print(f"    - VLM Reading: '{vlm_res.get('vlm_reading')}'")
    print("========================================")

if __name__ == "__main__":
    main()
