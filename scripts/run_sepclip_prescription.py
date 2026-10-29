"""Foreground MLQ runner for the isolated sepclip prescription experiments."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "benchmarks/sepclip_prescription_v14"
PREFIX = "sepclip_prescription_"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--ref", default="sepclip_mechanisms_rollout_clock_full")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--contracts-xml", type=Path)
    parser.add_argument("--expected-tests", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not args.name.startswith(PREFIX) or not command:
        parser.error(f"Require a {PREFIX} name and a foreground command")
    if not args.training and (args.contracts_xml is None or not args.expected_tests):
        parser.error("Contract jobs require their JUnit path and expected test count")
    if not os.environ.get("XDG_RUNTIME_DIR"):
        raise RuntimeError("Submit --inherit-env XDG_RUNTIME_DIR")

    pins_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    if hashlib.sha256(pins_path.read_bytes()).hexdigest() != args.manifest_sha256:
        raise RuntimeError("Frozen experiment source manifest changed after submission")
    pins = json.loads(pins_path.read_text())
    for relative, expected in pins["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Experiment source changed after review: {relative}")

    sources = [Path(__file__), *[ROOT / path for path in args.source]]
    sources.extend(sorted((ROOT / "cleanrl/shared").glob("*.py")))
    sources.extend(
        [ROOT / "cleanrl/shared/host_kernel.c", ROOT / "cleanrl/shared/mujoco_batch.c",
         ROOT / "scripts/autocull.py", ROOT / "scripts/_runs.py"]
    )
    manifest = {
        "name": args.name,
        "started_at_ns": time.time_ns(),
        "command": command,
        "frozen_source_manifest_sha256": args.manifest_sha256,
        "declared_max_parallel_runs": 1,
        "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "numpy", "scipy", "gymnasium", "mujoco", "tyro")},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources if path.exists()},
        "environment": {name: os.environ.get(name) for name in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CLEANRL_ENV_SPIN", "TMPDIR")},
        "autocull": {"reference": args.ref, "only": PREFIX, "minimum_steps": "default"}
                    if args.training else None,
    }
    gym_files = importlib.metadata.distribution("gymnasium")
    env_paths = [gym_files.locate_file(relative) for relative in (
        "gymnasium/envs/mujoco/assets/half_cheetah.xml",
        "gymnasium/envs/mujoco/half_cheetah_v4.py",
        "gymnasium/wrappers/time_limit.py",
    )]
    manifest["environment_source_sha256"] = {
        str(env_path): hashlib.sha256(env_path.read_bytes()).hexdigest()
        for env_path in env_paths if env_path.exists()
    }
    directory = ARTIFACTS / "execution_sources"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{args.name}_{manifest['started_at_ns']}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    cull = None
    result = None
    try:
        if not args.training:
            xml_path = args.contracts_xml if args.contracts_xml.is_absolute() else ROOT / args.contracts_xml
            if xml_path.exists():
                archived = xml_path.with_name(f"{xml_path.stem}_{manifest['started_at_ns']}.previous.xml")
                xml_path.replace(archived)
                manifest["previous_contract_report"] = str(archived)
        if args.training:
            cull = subprocess.Popen(
                [str(ROOT / ".venv/bin/python"), "-u", "scripts/autocull.py",
                 "--ref", args.ref, "--only", PREFIX, "--enforce", "all", "--yes",
                 "--watch", "600"], cwd=ROOT
            )
        result = subprocess.run(command, cwd=ROOT).returncode
        if not args.training:
            xml_path = args.contracts_xml if args.contracts_xml.is_absolute() else ROOT / args.contracts_xml
            if xml_path.exists():
                root = ET.parse(xml_path).getroot()
                cases = list(root.iter("testcase"))
                counts = {label: sum(case.find(label) is not None for case in cases)
                          for label in ("failure", "error", "skipped")}
                manifest["contracts"] = dict(tests=len(cases), expected=args.expected_tests, **counts)
                if len(cases) != args.expected_tests or any(counts.values()):
                    result = result or 1
            else:
                manifest["contracts"] = {"error": "No JUnit report produced"}
                result = result or 1
            from torch.utils.tensorboard import SummaryWriter

            log = ROOT / "runs" / f"HalfCheetah-v4__{args.name}__1__{int(time.time())}"
            with SummaryWriter(str(log)) as writer:
                writer.add_text("protocol", "CUDA contract validation; no shortened training run")
                writer.add_text("execution_manifest", str(path))
                writer.add_scalar("queue/exit_code", result, 0)
    finally:
        if cull is not None:
            cull.terminate()
            try:
                cull.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cull.kill()
                cull.wait()
        manifest.update(exit_code=result, finished_at_ns=time.time_ns())
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
