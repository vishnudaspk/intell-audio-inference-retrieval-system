"""
Direct FastAPI TestClient End-to-End Tracer
Traces Audio -> Worker -> Diarization -> Assembly -> DB -> API -> Response
"""

from collections import Counter
from fastapi.testclient import TestClient
from backend.api import app, repo, worker
from database.sqlite_db import SQLiteRepository

client = TestClient(app)

def trace_asset(audio_id: str):
    print("=" * 80)
    print(f"TRACING ASSET: {audio_id}")
    print("=" * 80)

    asset = repo.get_audio_asset(audio_id)
    if not asset:
        print("Asset not found!")
        return

    # 1. Inspect existing DB segments before reprocess
    old_db_segs = repo.get_audio_segments(audio_id)
    old_spks = [s.speaker_label for s in old_db_segs]
    print(f"[STAGE 0: PRE-EXISTING DB STATE]")
    print(f"  Old segment count : {len(old_db_segs)}")
    print(f"  Old speaker labels: {dict(Counter(old_spks))}")

    # 2. Trigger Reprocess through FastAPI API
    print(f"\n[STAGE 1: TRIGGERING POST /api/v1/assets/{audio_id}/process]")
    resp = client.post(f"/api/v1/assets/{audio_id}/process")
    print(f"  API Response Code : {resp.status_code}")
    if resp.status_code != 200:
        print(f"  Error Response    : {resp.text}")
        return
    job_data = resp.json().get("job", {})
    timings = job_data.get("timings", {})
    diag = timings.get("speaker_diarization_diagnostics", {})
    print(f"  Job Timings Diag  : {diag}")

    # 3. Direct DB Inspection immediately after reprocess
    new_db_segs = repo.get_audio_segments(audio_id)
    new_spks = [s.speaker_label for s in new_db_segs]
    print(f"\n[STAGE 2: POST-REPROCESS SQLITE DATABASE]")
    print(f"  Persisted segments count : {len(new_db_segs)}")
    print(f"  Persisted unique speakers: {sorted(list(set(new_spks)))}")
    print(f"  Persisted distribution   : {dict(Counter(new_spks))}")

    # 4. Fetch via GET /api/v1/assets/{audio_id}/segments
    print(f"\n[STAGE 3: GET /api/v1/assets/{audio_id}/segments (UI ENDPOINT)]")
    seg_resp = client.get(f"/api/v1/assets/{audio_id}/segments")
    print(f"  GET Status Code          : {seg_resp.status_code}")
    seg_data = seg_resp.json().get("segments", [])
    api_spks = [s.get("speaker_label") for s in seg_data]
    print(f"  API returned segments    : {len(seg_data)}")
    print(f"  API unique speaker labels: {sorted(list(set(api_spks)))}")
    print(f"  API speaker distribution : {dict(Counter(api_spks))}")

    print("\n[STAGE 4: SEGMENT-BY-SEGMENT VERIFICATION]")
    for i, s in enumerate(seg_data):
        print(f"  [{i:2d}] [{s.get('start_sec', 0.0):5.2f}s - {s.get('end_sec', 0.0):5.2f}s] {s.get('speaker_label', 'None'):10s} | {s.get('text', '')}")

if __name__ == "__main__":
    trace_asset("34266ef2-87de-4b39-8535-00ca3c931fe7")
    trace_asset("747c2e4b-db8e-45ae-b64d-4259fd419abc")
