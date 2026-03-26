#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PRACTICE_DIR = REPO_ROOT / "practice"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "practice-script-matrix.json"

SCRIPT_MATRIX = [
    {
        "name": "default",
        "path": "practice/default.py",
        "kind": "application",
        "notes": "Default traction-bar workflow.",
    },
    {
        "name": "discrete_atk",
        "path": "practice/discrete_atk.py",
        "kind": "application",
        "notes": "Discrete spring model.",
    },
    {
        "name": "discrete_atk_homogeneous",
        "path": "practice/discrete_atk_homogeneous.py",
        "kind": "application",
        "notes": "Homogeneous discrete spring model.",
    },
    {
        "name": "enpassant",
        "path": "practice/enpassant.py",
        "kind": "application",
        "notes": "Legacy exploratory crack evolution script.",
    },
    {
        "name": "multiaxial_disc",
        "path": "practice/multiaxial-disc.py",
        "kind": "application",
        "notes": "Multiaxial disc nucleation test.",
    },
    {
        "name": "pacman_cone",
        "path": "practice/pacman-cone.py",
        "kind": "application",
        "notes": "Pacman notch with cone stability.",
    },
    {
        "name": "pacman_hybrid",
        "path": "practice/pacman_hybrid.py",
        "kind": "application",
        "notes": "Pacman notch with hybrid first-order solve.",
    },
    {
        "name": "thinfilm_bar",
        "path": "practice/thinfilm-bar.py",
        "kind": "application",
        "notes": "Thin-film bar workflow.",
    },
    {
        "name": "traction_AT1_cone",
        "path": "practice/traction-AT1_cone.py",
        "kind": "application",
        "notes": "AT1 traction-bar with cone stability.",
    },
    {
        "name": "traction_AT1_first_order",
        "path": "practice/traction-AT1_first_order.py",
        "kind": "application",
        "notes": "AT1 traction-bar first-order workflow.",
    },
    {
        "name": "traction_AT2_cone",
        "path": "practice/traction-AT2_cone.py",
        "kind": "application",
        "notes": "AT2 traction-bar with cone stability.",
    },
    {
        "name": "traction_ATJJ",
        "path": "practice/traction-ATJJ.py",
        "kind": "application",
        "notes": "ATJJ traction-bar workflow.",
    },
    {
        "name": "traction_bar_clean",
        "path": "practice/traction-bar-clean.py",
        "kind": "application",
        "notes": "Clean traction-bar workflow.",
    },
    {
        "name": "traction_cone",
        "path": "practice/traction-cone.py",
        "kind": "application",
        "notes": "Traction-bar with cone stability.",
    },
    {
        "name": "traction_parametric",
        "path": "practice/traction-parametric.py",
        "kind": "application",
        "notes": "Parametric traction-bar workflow.",
    },
    {
        "name": "unstabinst",
        "path": "practice/unstabinst.py",
        "kind": "application",
        "notes": "Legacy unstable-instability script.",
    },
    {
        "name": "parametric_traction_bar_r",
        "path": "practice/parametric-traction-bar-r.py",
        "kind": "launcher",
        "notes": "Sweep launcher for traction-parametric refinement.",
    },
    {
        "name": "parametric_traction_bar_s",
        "path": "practice/parametric-traction-bar-s.py",
        "kind": "launcher",
        "notes": "Sweep launcher for traction-parametric scaling.",
    },
]


def selected_scripts(only, exclude, include_launchers):
    scripts = []
    only_set = set(only or [])
    exclude_set = set(exclude or [])
    for entry in SCRIPT_MATRIX:
        if entry["kind"] == "launcher" and not include_launchers:
            continue
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
    parser = argparse.ArgumentParser(description="Run migrated practice scripts one by one.")
    parser.add_argument("--only", action="append", help="Script name to run. Repeatable.")
    parser.add_argument("--exclude", action="append", help="Script name to skip. Repeatable.")
    parser.add_argument(
        "--include-launchers",
        action="store_true",
        help="Include sweep launchers such as parametric-traction-bar-r.py.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-script timeout in seconds.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON file to write results to.",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Print stdout/stderr tails for failed scripts during the run.",
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

    scripts = selected_scripts(args.only, args.exclude, args.include_launchers)

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
        "scripts": scripts,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2))

    ok = sum(result["status"] == "ok" for result in results)
    failed = sum(result["status"] == "failed" for result in results)
    timed_out = sum(result["status"] == "timeout" for result in results)
    print(
        f"summary: ok={ok} failed={failed} timeout={timed_out} "
        f"output={output_path}"
    )
    return 0 if failed == 0 and timed_out == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
