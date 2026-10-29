"""Read-only, prefix-scoped MLQ observer for microscopy and prescription jobs."""

import json
from pathlib import Path
import subprocess
import time


PREFIXES = ("sepclip_microscopy_", "sepclip_prescription_")
LIVE_STATES = {"queued", "starting", "running"}
HEARTBEAT = Path("benchmarks/sepclip_weight_microscopy/supervision_heartbeat.json")


def main():
    known = {}
    for path in (
        Path("benchmarks/sepclip_weight_microscopy/jobs.json"),
        Path("benchmarks/sepclip_prescription_v14/jobs.json"),
    ):
        for job in json.loads(path.read_text())["jobs"]:
            if not job["name"].startswith(PREFIXES):
                raise ValueError(f"Unexpected manifest job {job['id']}")
            known[job["id"]] = job
    observer = subprocess.Popen(["mlq", "subscribe"], stdout=subprocess.PIPE, text=True)
    try:
        for line in observer.stdout:
            if not line.lstrip().startswith("{"):
                continue
            snapshot = json.loads(line)
            current = {j["id"]: j for j in snapshot["jobs"] if j["name"].startswith(PREFIXES)}
            # Completed jobs disappear from subscriptions; resolve through MLQ itself.
            for job_id, previous in known.items():
                if job_id not in current and previous["state"] in LIVE_STATES:
                    current[job_id] = json.loads(subprocess.check_output(
                        ["mlq", "show", str(job_id), "--json"], text=True))
            changes = []
            for job_id, job in current.items():
                if job["maxParallelRuns"] != 1:
                    raise ValueError(f"Unexpected concurrency limit for {job_id}")
                previous = known.get(job_id)
                if previous is None or previous["state"] != job["state"]:
                    changes.append({key: job.get(key) for key in (
                        "id", "name", "state", "attemptCount", "maxParallelRuns")})
                known[job_id] = job
            heartbeat = json.loads(HEARTBEAT.read_text())
            heartbeat.update(
                updated_at=time.time(),
                status="Active read-only supervision of both owned prefixes",
                owned_pending=[{key: j.get(key) for key in (
                    "id", "name", "state", "attemptCount", "maxParallelRuns")}
                    for j in known.values() if j["state"] in LIVE_STATES],
                active_job_ids=[j["id"] for j in snapshot["jobs"] if j["state"] == "running"],
                reservation_job_id=(snapshot.get("reservation") or {}).get("protectedJob"),
                observed_prefixes=list(PREFIXES),
                unrelated_jobs_controlled=[],
            )
            HEARTBEAT.write_text(json.dumps(heartbeat, indent=2) + "\n")
            if changes:
                print(json.dumps({"captured_at": time.time(), "changes": changes}), flush=True)
    finally:
        observer.terminate()
        observer.wait(timeout=10)


if __name__ == "__main__":
    main()
