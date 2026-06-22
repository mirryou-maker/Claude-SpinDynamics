"""Unified results store for the cross-solver benchmark campaign.

Every runner (Claude-SD, mumax3, mumax+, MuMax-CO) appends records here with one
common schema, so make_report.py can render all tables/figures from a single
file: benchmarks/results/all_solvers.json

Record schema (one dict per measurement):
  scenario   : str   e.g. "A1_sp4", "B2_large3d", "C1_sot_thermal", "D-sp2"
  solver     : str   "claude-sd" | "mumax3" | "mumax+" | "mumax-co"
  build      : str   "cuFFT_f64" | "cuFFT_f32" | "VkFFT_f32" | "f32"  (f32 for mumax family)
  integrator : str   "RK4" | "RK45-DP" | "Heun" | "relax"
  dim        : str   "2D" | "3D" | "macrospin" | "quasistatic"
  cells      : int
  T_K        : float
  metric     : str   "throughput" | "accuracy" | "switching" | "remanence" | "thermal"
  # timing (optional, present for throughput/wall):
  ms_step    : float | None
  ms_eval    : float | None     # ms_step / evals_per_step
  wall_ms    : float | None
  repeats    : int              # number of measured repeats
  ms_step_iqr: float | None     # inter-quartile range across repeats
  # observable (optional, present for accuracy):
  observable : str | None       # e.g. "mx_1ns", "t_switch_ps", "H_sw_mT", "remanence_111"
  value      : float | None
  error_vs_ref    : float | None   # vs µMAG reference
  error_vs_csf64  : float | None   # vs Claude-SD f64 (high-precision anchor)
  # provenance:
  hw         : str              # GPU + clocks
  notes      : str
"""
import json
import os
import pathlib

HERE = pathlib.Path(__file__).parent
STORE = HERE / "all_solvers.json"

EVALS = {"RK4": 4, "RK45-DP": 6, "DOPRI5": 6, "Heun": 2, "Euler": 1, "relax": 1}


def _load():
    if STORE.exists():
        try:
            return json.loads(STORE.read_text())
        except Exception:
            return []
    return []


def append(record: dict):
    """Append one record (dict matching the schema) to all_solvers.json."""
    data = _load()
    data.append(record)
    STORE.write_text(json.dumps(data, indent=2))
    return record


def append_many(records):
    data = _load()
    data.extend(records)
    STORE.write_text(json.dumps(data, indent=2))


def make_record(scenario, solver, build, integrator, dim, cells, **kw):
    """Construct a schema-complete record with sensible defaults."""
    evals = EVALS.get(integrator, 1)
    ms_step = kw.get("ms_step")
    rec = {
        "scenario": scenario, "solver": solver, "build": build,
        "integrator": integrator, "dim": dim, "cells": int(cells),
        "T_K": float(kw.get("T_K", 0.0)),
        "metric": kw.get("metric", "throughput"),
        "ms_step": ms_step,
        "ms_eval": (ms_step / evals) if ms_step is not None else kw.get("ms_eval"),
        "wall_ms": kw.get("wall_ms"),
        "repeats": int(kw.get("repeats", 1)),
        "ms_step_iqr": kw.get("ms_step_iqr"),
        "observable": kw.get("observable"),
        "value": kw.get("value"),
        "error_vs_ref": kw.get("error_vs_ref"),
        "error_vs_csf64": kw.get("error_vs_csf64"),
        "hw": kw.get("hw", "RTX 5060 Ti, CUDA 13.2"),
        "notes": kw.get("notes", ""),
    }
    return rec


if __name__ == "__main__":
    # Smoke test
    r = make_record("A1_sp4", "claude-sd", "cuFFT_f32", "RK45-DP", "2D", 10000,
                    ms_step=0.156, metric="throughput", repeats=3, ms_step_iqr=0.004)
    print(json.dumps(r, indent=2))
    print("EVALS map:", EVALS)
