#!/usr/bin/env python3
"""
Run MLP benchmarks (PRIM or Memclave) for predefined N values (m=n=N),
extract DPU Kernel Time (ms), and upsert into a CSV:

Columns: System, <N1>, <N2>, ...
Rows: CPU, PIM-insecure, Memclave

Behavior:
- mode=prim: updates rows CPU (CPU Version Time) and PIM-insecure (DPU Kernel Time)
  from ./bin/mlp_host output (run from PRIM's MLP directory by default).
- mode=memclave: updates row Memclave (DPU Kernel Time)
  from ./ime-mlp-example output (run from Memclave build directory by default).

Rounding: 2 decimals.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


DEFAULT_NS = [1024, 2048, 4096, 8192, 16384]
DEFAULT_CSV = "mlp_results.csv"

ROW_CPU = "CPU"
ROW_PIM = "PIM-insecure"
ROW_MEMCLAVE = "Memclave"

RE_CPU_TIME = re.compile(r"CPU Version Time \(ms\):\s*([0-9]*\.?[0-9]+)")
RE_DPU_KERNEL = re.compile(r"DPU Kernel Time \(ms\):\s*([0-9]*\.?[0-9]+)")

# Some outputs have missing whitespace between fields; this regex is tolerant enough:
# e.g., "... DPU Kernel Time (ms): 500.930000Inter-DPU Time ..."
# because it stops at first non-number char.
RE_DPU_KERNEL_TOL = re.compile(r"DPU Kernel Time \(ms\):\s*([0-9]*\.?[0-9]+)")


def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    out = proc.stdout or ""
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n--- Output ---\n{out}"
        )
    return out


def parse_times(output: str) -> Tuple[Optional[float], Optional[float]]:
    cpu = None
    dpu = None

    m = RE_CPU_TIME.search(output)
    if m:
        cpu = float(m.group(1))

    m = RE_DPU_KERNEL_TOL.search(output)
    if m:
        dpu = float(m.group(1))

    return cpu, dpu


def round2(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return round(x + 1e-12, 2)  # tiny epsilon to avoid 1.005 -> 1.00 surprises


def read_csv(path: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """
    Returns (header, table) where:
      header = ["System", "1024", ...]
      table[row_name][col_name] = value (string)
    If file doesn't exist, returns empty table with just ["System"] header.
    """
    if not os.path.exists(path):
        return ["System"], {}

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return ["System"], {}
        header = list(reader.fieldnames)
        table: Dict[str, Dict[str, str]] = {}
        for row in reader:
            sysname = (row.get("System") or "").strip()
            if not sysname:
                continue
            table[sysname] = {k: (row.get(k) or "").strip() for k in header if k != "System"}
        return header, table


def write_csv(path: str, header: List[str], table: Dict[str, Dict[str, str]]) -> None:
    # Stable row order
    row_order = [ROW_CPU, ROW_PIM, ROW_MEMCLAVE]
    existing = [r for r in row_order if r in table]
    extras = sorted([r for r in table.keys() if r not in row_order])
    final_rows = existing + extras + [r for r in row_order if r not in existing and r in table]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in final_rows:
            row = {"System": r}
            for c in header[1:]:
                row[c] = table.get(r, {}).get(c, "")
            writer.writerow(row)


def ensure_header_has_cols(header: List[str], ns: List[int]) -> List[str]:
    # Ensure "System" first and each N exists as a string column
    cols = ["System"]
    seen = set()
    # keep any existing numeric cols in their order
    for h in header:
        if h == "System":
            continue
        if h not in seen:
            cols.append(h)
            seen.add(h)
    for n in ns:
        s = str(n)
        if s not in seen:
            cols.append(s)
            seen.add(s)
    return cols


def upsert_value(
    table: Dict[str, Dict[str, str]],
    row_name: str,
    col_name: str,
    value: Optional[float],
) -> None:
    if row_name not in table:
        table[row_name] = {}
    if value is None:
        return
    table[row_name][col_name] = f"{value:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prim", "memclave"], required=True,
                    help="Where to run: prim (./bin/mlp_host) or memclave (./ime-mlp-example)")
    ap.add_argument("--ns", default=",".join(map(str, DEFAULT_NS)),
                    help="Comma-separated list of N values (m=n=N). Default: 1024,2048,4096,8192,16384")
    ap.add_argument("--csv", default=DEFAULT_CSV, help=f"CSV path. Default: {DEFAULT_CSV}")
    ap.add_argument("--cwd", default=".", help="Working directory to run commands from (default: current dir)")
    ap.add_argument("--prim-bin", default="./bin/mlp_host",
                    help="Path to PRIM mlp_host binary (relative to --cwd). Default: ./bin/mlp_host")
    ap.add_argument("--memclave-bin", default="./ime-mlp-example",
                    help="Path to Memclave ime-mlp-example binary (relative to --cwd). Default: ./ime-mlp-example")
    ap.add_argument("--dry-run", action="store_true", help="Print commands but do not execute or write CSV")
    args = ap.parse_args()

    ns = [int(x.strip()) for x in args.ns.split(",") if x.strip()]
    if not ns:
        print("No N values provided.", file=sys.stderr)
        return 2

    header, table = read_csv(args.csv)
    header = ensure_header_has_cols(header, ns)

    # Run benchmarks
    for n in ns:
        col = str(n)
        if args.mode == "prim":
            cmd = [args.prim_bin, "-m", str(n), "-n", str(n)]
        else:
            cmd = [args.memclave_bin, "-m", str(n), "-n", str(n)]

        print(f"==> Running N={n}: {' '.join(cmd)} (cwd={args.cwd})")
        if args.dry_run:
            continue

        out = run_cmd(cmd, cwd=args.cwd)
        cpu_time, dpu_kernel = parse_times(out)

        cpu_time = round2(cpu_time)
        dpu_kernel = round2(dpu_kernel)

        if dpu_kernel is None:
            print(f"[WARN] Could not parse DPU Kernel Time for N={n}. Output was:\n{out}", file=sys.stderr)
            continue

        if args.mode == "prim":
            # CPU row uses CPU Version Time
            if cpu_time is None:
                print(f"[WARN] Could not parse CPU Version Time for N={n}. Output was:\n{out}", file=sys.stderr)
            else:
                upsert_value(table, ROW_CPU, col, cpu_time)
            # PIM-insecure row uses DPU Kernel Time
            upsert_value(table, ROW_PIM, col, dpu_kernel)
        else:
            # Memclave row uses DPU Kernel Time
            upsert_value(table, ROW_MEMCLAVE, col, dpu_kernel)

    if args.dry_run:
        print("[DRY-RUN] Skipping CSV write.")
        return 0

    # Ensure rows exist if partially filled
    for r in [ROW_CPU, ROW_PIM, ROW_MEMCLAVE]:
        table.setdefault(r, {})

    write_csv(args.csv, header, table)
    print(f"Updated: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
