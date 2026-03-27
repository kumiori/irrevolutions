#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "output" / "demo-script-matrix.json"

SCRIPT_MATRIX = [
    {
        "name": "demo_bifurcation",
        "path": "demo/demo_bifurcation.py",
        "kind": "application",
        "notes": "Bifurcation demo on the traction bar.",
    },
    {
        "name": "demo_biharmonic",
        "path": "demo/demo_biharmonic.py",
        "kind": "application",
        "notes": "Biharmonic demo.",
    },
    {
        "name": "demo_contact",
        "path": "demo/demo_contact.py",
        "kind": "application",
        "notes": "Contact demo.",
    },
    {
        "name": "demo_elasticity",
        "path": "demo/demo_elasticity.py",
        "kind": "application",
        "notes": "Pure elasticity demo.",
    },
    {
        "name": "demo_strong_aniso",
        "path": "demo/demo_strong_aniso.py",
        "kind": "application",
        "notes": "Strong anisotropy demo.",
    },
    {
        "name": "demo_traction",
        "path": "demo/demo_traction.py",
        "kind": "application",
        "notes": "Traction bar demo.",
    },
    {
        "name": "demo_vector_info",
        "path": "demo/demo_vector_info.py",
        "kind": "application",
        "notes": "Vector information demo.",
    },
    {
        "name": "demo_vi",
        "path": "demo/demo_vi.py",
        "kind": "application",
        "notes": "Variational inequality demo.",
    },
]


def selected_scripts(only, exclude):
    scripts = []
    only_set = set(only or [])
    exclude_set = set(exclude or [])
    for entry in SCRIPT_MATRIX:
        if only_set and entry["name"] not in only_set:
            continue
        if entry["name"] in exclude_set:
            continue
        scripts.append(entry)
    return scripts


def build_env():
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("XDG_RUNTIME_DIR", "/tmp")
    return env


def run_script(entry, timeout):
    start = time.time()
    command = [sys.executable, entry["path"]]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=build_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        status = "ok" if completed.returncode == 0 else "failed"
        return {
            "name": entry["name"],
            "path": entry["path"],
            "kind": entry["kind"],
            "status": status,
            "returncode": completed.returncode,
            "duration_seconds": round(time.time() - start, 3),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": entry["name"],
            "path": entry["path"],
            "kind": entry["kind"],
            "status": "timeout",
            "returncode": None,
            "duration_seconds": round(time.time() - start, 3),
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
        }


def tail_lines(text, line_count):
    if not text:
        return ""
    lines = text.rstrip().splitlines()
    return "\n".join(lines[-line_count:])


def main():
    parser = argparse.ArgumentParser(description="Run demo scripts one by one.")
    parser.add_argument("--only", action="append", help="Demo name to run. Repeatable.")
    parser.add_argument("--exclude", action="append", help="Demo name to skip. Repeatable.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-demo timeout in seconds.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON file to write results to.",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Print stdout/stderr tails for failed demos during the run.",
    )
    parser.add_argument(
        "--failure-lines",
        type=int,
        default=40,
        help="Number of stdout/stderr tail lines to print for failures.",
    )
    parser.add_argument("--list", action="store_true", help="List the manifest and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    args = parser.parse_args()

    scripts = selected_scripts(args.only, args.exclude)

    if args.list:
        for entry in scripts:
            print(f"{entry['name']}: {entry['path']} [{entry['kind']}]")
        return 0

    if args.dry_run:
        for entry in scripts:
            print(f"{sys.executable} {entry['path']}")
        return 0

    results = []
    for entry in scripts:
        print(f"[run] {entry['name']} -> {entry['path']}")
        result = run_script(entry, args.timeout)
        print(
            f"[{result['status']}] {entry['name']} "
            f"(rc={result['returncode']}, {result['duration_seconds']}s)"
        )
        if args.show_output and result["status"] != "ok":
            stdout_tail = tail_lines(result["stdout_tail"], args.failure_lines)
            stderr_tail = tail_lines(result["stderr_tail"], args.failure_lines)
            if stdout_tail:
                print(f"--- stdout tail: {entry['name']} ---")
                print(stdout_tail)
            if stderr_tail:
                print(f"--- stderr tail: {entry['name']} ---")
                print(stderr_tail)
        results.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo_root": str(REPO_ROOT),
        "generated_at_epoch": time.time(),
        "results": results,
        "summary": {
            "ok": sum(result["status"] == "ok" for result in results),
            "failed": sum(result["status"] == "failed" for result in results),
            "timeout": sum(result["status"] == "timeout" for result in results),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))
    summary = payload["summary"]
    print(
        f"summary: ok={summary['ok']} failed={summary['failed']} "
        f"timeout={summary['timeout']} output={output_path}"
    )
    return 0 if summary["failed"] == 0 and summary["timeout"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
