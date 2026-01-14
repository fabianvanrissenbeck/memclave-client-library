#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


DEFAULT_CSV = "bfs_results.csv"

ROW_CPU = "CPU"
ROW_PIM = "PIM-insecure"
ROW_MEMCLAVE = "Memclave"

RE_DPU_KERNEL = re.compile(r"DPU Kernel Time:\s*([0-9]*\.?[0-9]+)\s*ms")
RE_CPU_TIME = re.compile(r"CPU Version Time:\s*([0-9]*\.?[0-9]+)\s*ms")


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
    m = RE_DPU_KERNEL.search(output)
    if m:
        dpu = float(m.group(1))
    return cpu, dpu


def round2(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return round(x + 1e-12, 2)


def read_csv(path: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
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
    row_order = [ROW_CPU, ROW_PIM, ROW_MEMCLAVE]
    final_rows = [r for r in row_order if r in table] + sorted([r for r in table if r not in row_order])

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in final_rows:
            row = {"System": r}
            for c in header[1:]:
                row[c] = table.get(r, {}).get(c, "")
            writer.writerow(row)


def ensure_header_has_cols(header: List[str], cols: List[str]) -> List[str]:
    out = ["System"]
    seen = set()
    for h in header:
        if h == "System":
            continue
        if h not in seen:
            out.append(h)
            seen.add(h)
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def upsert_value(table: Dict[str, Dict[str, str]], row: str, col: str, val: Optional[float]) -> None:
    if val is None:
        return
    table.setdefault(row, {})
    table[row][col] = f"{val:.2f}"


def autodetect_prefixes(mode: str) -> List[str]:
    # Ordered guesses, relative to cwd you run from
    if mode == "memclave":
        return ["../examples/BFS/data", "./examples/BFS/data", "./data", "."]
    else:
        return ["./data", "../data", ".", "../examples/BFS/data"]


def resolve_graph_paths(
    graphs: List[str],
    cwd: str,
    graph_prefix: Optional[str],
    mode: str,
) -> List[str]:
    resolved: List[str] = []

    prefixes: List[str] = []
    if graph_prefix:
        prefixes = [graph_prefix]
    else:
        prefixes = autodetect_prefixes(mode)

    for g in graphs:
        g = g.strip()
        if not g:
            continue

        # If user already provided a path, resolve against cwd and validate
        if os.path.sep in g or g.startswith("."):
            cand = os.path.normpath(os.path.join(cwd, g)) if not os.path.isabs(g) else g
            if not os.path.exists(cand):
                raise FileNotFoundError(f"Graph not found: {g} (resolved to {cand})")
            resolved.append(os.path.relpath(cand, start=cwd))
            continue

        # Filename only: try prefixes
        found = None
        for p in prefixes:
            cand = os.path.normpath(os.path.join(cwd, p, g))
            if os.path.exists(cand):
                found = cand
                break
        if not found:
            tried = [os.path.normpath(os.path.join(cwd, p, g)) for p in prefixes]
            raise FileNotFoundError(
                f"Graph file '{g}' not found. Tried:\n  " + "\n  ".join(tried)
            )

        resolved.append(os.path.relpath(found, start=cwd))

    return resolved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prim", "memclave"], required=True)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--prim-bin", default="./bin/host_code")
    ap.add_argument("--memclave-bin", default="./ime-bfs-example")
    ap.add_argument(
        "--graphs",
        default="LiveJournal1,loc-gowalla,roadNet-PA",
        help="Comma-separated graph filenames or paths.",
    )
    ap.add_argument(
        "--graph-prefix",
        default=None,
        help="Directory containing graph files (e.g., ./data or ../examples/BFS/data). "
             "If omitted, script auto-detects.",
    )
    ap.add_argument("--colnames", choices=["basename", "path"], default="basename")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cwd = args.cwd
    raw_graphs = [x.strip() for x in args.graphs.split(",") if x.strip()]
    if not raw_graphs:
        print("No graphs provided.", file=sys.stderr)
        return 2

    try:
        graph_paths = resolve_graph_paths(raw_graphs, cwd=cwd, graph_prefix=args.graph_prefix, mode=args.mode)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        print("\nTip: pass --graph-prefix ../examples/BFS/data (memclave) or --graph-prefix ./data (prim).",
              file=sys.stderr)
        return 2

    cols = [os.path.basename(p) if args.colnames == "basename" else p for p in graph_paths]

    header, table = read_csv(args.csv)
    header = ensure_header_has_cols(header, cols)

    for graph_path, col in zip(graph_paths, cols):
        if args.mode == "prim":
            cmd = [args.prim_bin, "-f", graph_path]
        else:
            cmd = [args.memclave_bin, "-f", graph_path]

        print(f"==> Running graph={graph_path}: {' '.join(cmd)})")
        if args.dry_run:
            continue

        out = run_cmd(cmd, cwd=cwd)
        cpu_time, dpu_kernel = parse_times(out)

        cpu_time = round2(cpu_time)
        dpu_kernel = round2(dpu_kernel)

        if dpu_kernel is None:
            print(f"[WARN] Could not parse DPU Kernel Time for graph={graph_path}.", file=sys.stderr)
            continue

        if args.mode == "prim":
            if cpu_time is None:
                print(f"[WARN] Could not parse CPU Version Time for graph={graph_path}.", file=sys.stderr)
            else:
                upsert_value(table, ROW_CPU, col, cpu_time)
            upsert_value(table, ROW_PIM, col, dpu_kernel)
        else:
            upsert_value(table, ROW_MEMCLAVE, col, dpu_kernel)

    if args.dry_run:
        print("[DRY-RUN] Skipping CSV write.")
        return 0

    for r in [ROW_CPU, ROW_PIM, ROW_MEMCLAVE]:
        table.setdefault(r, {})

    write_csv(args.csv, header, table)
    print(f"Updated: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

