# -*- coding: utf-8 -*-
"""Unified QA runner for in-image self-check, warmup and release gating.

Examples (inside the container):
    # fast unit tests only (safe before engine start)
    python -m app.qa.run_qa --unit

    # warmup + functional smoke against this container's own service
    python -m app.qa.run_qa --smoke --url http://127.0.0.1:8000

    # concurrency load sweep
    python -m app.qa.run_qa --load --url http://127.0.0.1:8000 --total 32

    # full release gate: unit + smoke + load (with min speedup check)
    python -m app.qa.run_qa --all --url http://127.0.0.1:8000 --min-speedup 1.5
"""

import argparse
import os
import sys


def _default_key() -> str | None:
    return os.getenv("API_KEY") or None


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3-ASR in-image QA suite")
    ap.add_argument("--unit", action="store_true", help="run unit checks")
    ap.add_argument("--smoke", action="store_true", help="run smoke checks")
    ap.add_argument("--load", action="store_true", help="run load sweep")
    ap.add_argument("--all", action="store_true", help="unit + smoke + load")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="service base URL")
    ap.add_argument("--key", default=None, help="API key (defaults to env API_KEY)")
    ap.add_argument("--concurrency", default="1,4,8,16,32,64")
    ap.add_argument("--total", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--min-speedup", type=float, default=None)
    args = ap.parse_args()

    if not (args.unit or args.smoke or args.load or args.all):
        ap.print_help()
        return 2

    do_unit = args.unit or args.all
    do_smoke = args.smoke or args.all
    do_load = args.load or args.all

    key = args.key if args.key is not None else _default_key()
    failed = []

    if do_unit:
        from app.qa.unit_checks import run as run_unit

        print("=== UNIT CHECKS ===")
        if not run_unit(verbosity=1):
            failed.append("unit")

    if do_smoke:
        from app.qa.smoke_checks import run as run_smoke

        print("=== SMOKE CHECKS ===")
        if not run_smoke(args.url, key, args.timeout):
            failed.append("smoke")

    if do_load:
        from app.qa.load_checks import run as run_load

        print("=== LOAD SWEEP ===")
        if not run_load(
            args.url, key, args.concurrency, args.total, args.timeout,
            min_speedup=args.min_speedup,
        ):
            failed.append("load")

    if failed:
        print(f"[qa] FAILED: {failed}")
        return 1
    print("[qa] ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
