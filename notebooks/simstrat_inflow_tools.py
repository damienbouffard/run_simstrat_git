#!/usr/bin/env python3
"""
Shared utilties to create inflow files for the Simstrat 1D lake model

Key behaviors:
- File top lines:
  * Mode 1 (manual):  "t (1.column)\tz (1.row)\tQ [m2/s]"
  * Mode 2 (density): "t [d]\tz (1. row)\tQ [m3/s] / [m2/s]"
- Line 2: "Nd Ns" → number of deep columns (fixed to bottom) and surface columns (move with level)
- Line 3: "-1 <deep depths...> <surface depths...>"
- Lines 4+: time [d] in col 1; then values per column (units depend on mode and column type).

Manual syntax (mode 1 and surface/manual in mode 2):
- Values must be per-meter [m2/s]. The model integrates over depth.
- A manual interval [a,b] m below surface is written using four columns (no binning):
  [a, 0], [a, Q/(b-a)], [b, Q/(b-a)], [b, 0]. Depths are written negative.

Density-driven syntax (mode 2, deep plunging inflows):
- Each column is one inflow with constant discharge in [m3/s] at its input depth (negative).

Depth conventions:
- Input depths in the text are given as positive meters below the surface.
- In files, depths are written as negative numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import re


# ---------------- Data structures ----------------

@dataclass
class Column:
    placement: str   # 'deep' or 'surface'
    mode: str        # 'manual' or 'density' (density for plunging inflows in mode 2)
    depth_m: float   # negative value (below surface)
    value_unit: str  # 'm3/s' or 'm2/s'
    value_const: float  # constant value for all times


# ---------------- Parsing ----------------
# Mode 1: only manual inflows
# Lines use S (surface-linked) or D (deep/bottom-linked), with interval a-b m and discharge Q [m3/s]
# Example:
#   1) S, depth: 5-15 m, discharge: 80 m3/s
#   2) D, depth: 25-30 m, discharge: 12.5 m3/s

_pat_mode1 = re.compile(
    r"(?i)^[0-9]+\)\s*([SD])\s*,\s*depth:\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*[-–]\s*([0-9]+(?:\.[0-9]+)?)\s*m\s*,\s*"
    r"discharge:\s*([0-9]+(?:\.[0-9]+)?)\s*m\s*3\s*/\s*s\s*$"
)

def parse_mode1_text(text: str) -> List[tuple]:
    """
    Parse mode-1 manual inflow definitions.
    Returns a list of tuples: (placement, a, b, Q), where
      placement ∈ {'deep','surface'}, a,b > 0 (m below surface), Q in m3/s.
    """
    entries: List[tuple] = []
    for ln in (ln.strip() for ln in text.splitlines() if ln.strip()):
        m = _pat_mode1.match(ln)
        if not m:
            continue
        tag = m.group(1).upper()
        a = float(m.group(2))
        b = float(m.group(3))
        Q = float(m.group(4))
        if b <= a:
            raise ValueError(f"Manual interval has non-positive thickness: '{a}-{b} m'.")
        placement = 'surface' if tag == 'S' else 'deep'
        entries.append((placement, a, b, Q))
    if not entries:
        raise ValueError("No valid mode-1 lines found.")
    return entries


# Mode 2: plunging deep (P) and surface-linked manual (S) ONLY
# Examples:
#   1) P, depth: 20 m, discharge: 100 m3/s
#   2) S, depth: 5-15 m, discharge: 100 m3/s

_pat_mode2_P = re.compile(
    r"(?i)^[0-9]+\)\s*P\s*,\s*depth:\s*([0-9]+(?:\.[0-9]+)?)\s*m\s*,\s*discharge:\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*m\s*3\s*/\s*s\s*$"
)

# NOTE: accepts only 'S' (surface-linked manual). 'M' is intentionally NOT supported.
_pat_mode2_S = re.compile(
    r"(?i)^[0-9]+\)\s*S\s*,\s*depth:\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*[-–]\s*([0-9]+(?:\.[0-9]+)?)\s*m\s*,\s*"
    r"discharge:\s*([0-9]+(?:\.[0-9]+)?)\s*m\s*3\s*/\s*s\s*$"
)

def parse_mode2_text(text: str) -> Tuple[List[tuple], List[tuple]]:
    """
    Parse mode-2 definitions.

    Returns (plunging, manual_surface) where
      plunging        = list of (depth, Q) for 'P' entries (deep density-driven)
      manual_surface  = list of (a, b, Q) for 'S' entries (surface manual intervals)

    Lines beginning with 'M' for manual are NOT accepted here by design.
    """
    P: List[tuple] = []
    manual_surface: List[tuple] = []

    for ln in (ln.strip() for ln in text.splitlines() if ln.strip()):
        mp = _pat_mode2_P.match(ln)
        ms = _pat_mode2_S.match(ln)

        if mp:
            depth = float(mp.group(1))
            Q = float(mp.group(2))
            P.append((depth, Q))
        elif ms:
            a = float(ms.group(1))
            b = float(ms.group(2))
            Q = float(ms.group(3))
            if b <= a:
                raise ValueError(f"Manual interval has non-positive thickness: '{a}-{b} m'.")
            manual_surface.append((a, b, Q))
        else:
            # If you prefer a hard error on any non-matching line, replace `continue` with:
            # raise ValueError(f"Invalid Mode-2 line (only P/S allowed): '{ln}'")
            continue

    if not (P or manual_surface):
        raise ValueError("No valid mode-2 lines found (expect 'P' or 'S').")
    return P, manual_surface


# ---------------- Column builders ----------------

def build_columns_mode1(manual_entries: List[tuple]) -> List[Column]:
    """Build columns for mode 1 (manual only). Each entry is (placement, a, b, Q)."""
    cols: List[Column] = []
    for placement, a, b, Q in manual_entries:
        thickness = b - a
        q_per_m = Q / thickness
        a_neg = -abs(a)
        b_neg = -abs(b)
        # Four columns at a and b with 0 → q_per_m → q_per_m → 0
        for depth, val in [(a_neg, 0.0), (a_neg, q_per_m), (b_neg, q_per_m), (b_neg, 0.0)]:
            cols.append(Column(
                placement=placement,
                mode='manual',
                depth_m=depth,
                value_unit='m2/s',
                value_const=val
            ))
    return cols


def build_columns_mode2(plunging: List[tuple], manual_surface: List[tuple]) -> List[Column]:
    """
    Build columns for mode 2.
    - plunging: list of (depth, Q) → deep density-driven columns in m3/s
    - manual_surface: list of (a, b, Q) → surface manual intervals in m2/s, four-step columns
    """
    cols: List[Column] = []
    # Deep density-driven
    for depth, Q in plunging:
        cols.append(Column(
            placement='deep',
            mode='density',
            depth_m=-abs(depth),
            value_unit='m3/s',
            value_const=Q
        ))
    # Surface manual intervals
    for a, b, Q in manual_surface:
        thickness = b - a
        q_per_m = Q / thickness
        a_neg = -abs(a)
        b_neg = -abs(b)
        for depth, val in [(a_neg, 0.0), (a_neg, q_per_m), (b_neg, q_per_m), (b_neg, 0.0)]:
            cols.append(Column(
                placement='surface',
                mode='manual',
                depth_m=depth,
                value_unit='m2/s',
                value_const=val
            ))
    return cols


# ---------------- File writer & utilities ----------------

def split_deep_surface_columns(cols: List[Column]) -> Tuple[List[Column], List[Column]]:
    deep = [c for c in cols if c.placement == 'deep']
    surf = [c for c in cols if c.placement == 'surface']
    return deep, surf


def generate_times(start_day: float, end_day: float, dt_day: float) -> List[float]:
    if dt_day <= 0:
        raise ValueError("dt_day must be > 0.")
    if end_day < start_day:
        raise ValueError("end_day must be >= start_day.")
    t = start_day
    times = []
    # inclusive
    while t <= end_day + 1e-12:
        times.append(round(t, 10))
        t += dt_day
    return times


def write_inflow_file(path: str, mode: int, cols: List[Column],
                      start_day: float, end_day: float, dt_day: float) -> None:
    """Write inflow file with proper header/units for the given mode (1 or 2)."""
    if mode not in (1, 2):
        raise ValueError("mode must be 1 or 2")
    deep_cols, surf_cols = split_deep_surface_columns(cols)
    Nd, Ns = len(deep_cols), len(surf_cols)
    times = generate_times(start_day, end_day, dt_day)

    with open(path, 'w', encoding='utf-8') as f:
        # Header line
        if mode == 1:
            f.write("t (1.column)\tz (1.row)\tQ [m2/s]\n")
        else:
            f.write("t [d]\tz (1. row)\tQ [m3/s] / [m2/s]\n")
        # Counts
        f.write(f"{Nd} {Ns}\n")
        # Depths line
        f.write("-1")
        for c in deep_cols:
            f.write(f"\t{c.depth_m:.6f}")
        for c in surf_cols:
            f.write(f"\t{c.depth_m:.6f}")
        f.write("\n")
        # Time rows
        for t in times:
            f.write(f"{t:.6f}")
            if mode == 1:
                # All columns are manual (m2/s)
                for c in deep_cols + surf_cols:
                    f.write(f"\t{c.value_const:.6f}")
            else:
                # Mode 2: deep first (m3/s for density), then surface manual (m2/s)
                for c in deep_cols:
                    f.write(f"\t{c.value_const:.6f}")
                for c in surf_cols:
                    f.write(f"\t{c.value_const:.6f}")
            f.write("\n")


def preview_file(path: str, n: int = 16) -> None:
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            print(line.rstrip())
            if i + 1 >= n:
                break


