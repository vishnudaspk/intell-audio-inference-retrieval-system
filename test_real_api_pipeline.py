import requests
from collections import Counter

audio_id = "34266ef2-87de-4b39-8535-00ca3c931fe7"
print(f"Triggering POST /api/v1/assets/{audio_id}/process ...")
res = requests.post(f"http://127.0.0.1:8000/api/v1/assets/{audio_id}/process")
print(f"Reprocess Response Status: {res.status_code}")

if res.status_code != 200:
    print("Reprocess failed:", res.text)
else:
    print("Fetching GET /api/v1/assets/{audio_id}/segments ...")
    seg_res = requests.get(f"http://127.0.0.1:8000/api/v1/assets/{audio_id}/segments")
    data = seg_res.json()
    segs = data.get("segments", [])
    print(f"Total API segments: {len(segs)}")
    spks = [s.get("speaker_label") for s in segs]
    print(f"Unique speaker labels: {sorted(list(set(spks)))}")
    print(f"Speaker distribution: {dict(Counter(spks))}")
    print("\nAPI Returned Segments:")
    for i, s in enumerate(segs):
        st = s.get("start_sec", 0.0)
        et = s.get("end_sec", 0.0)
        lbl = s.get("speaker_label", "None")
        txt = s.get("text", "")
        print(f"  [{i:2d}] [{st:5.2f}s - {et:5.2f}s] {lbl:10s} | {txt}")
