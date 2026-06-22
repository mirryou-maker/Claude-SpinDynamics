"""Merge notebook (NB41-50) accuracy observables into the unified all_solvers.json.

Reads notebooks/<nb>_results.json (regenerated from the latest-build re-run) and
extracts the well-posed scalar accuracy observables (SP#4 mx & t_switch, SP#1 L_c,
SP#3 H_sw, FMR frequency, DW velocity). Skyrmion Q (NB45) is intentionally EXCLUDED
from quantitative records — it is a precision-sensitivity caveat, not an agreement
test. Writes metric="accuracy" records so make_report can render an accuracy table.

Run: py -3.13 benchmarks/merge_nb_accuracy.py
"""
import json, pathlib, sys

HERE = pathlib.Path(__file__).parent
NB = HERE.parent / "notebooks"
sys.path.insert(0, str(HERE / "results"))
import results_io as rio  # noqa: E402

REF = {  # muMAG / analytic reference values
    "SP4_mx_1ns": -0.9862, "SP1_Lc_nm": 116.0, "SP3_Hsw_mT": -20.0,
    "FMR_GHz": 1.4006,
}


def load(nb):
    p = NB / f"{nb}_results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def emit(recs, scenario, solver, build, observable, value, ref_key=None, **kw):
    if value is None:
        return
    err = None
    if ref_key and value is not None and REF.get(ref_key):
        err = abs(value - REF[ref_key])
    recs.append(rio.make_record(
        scenario, solver, build, kw.get("integrator", "RK45-DP"),
        kw.get("dim", "quasistatic"), kw.get("cells", 0),
        metric="accuracy", observable=observable, value=value,
        error_vs_ref=err, notes=kw.get("notes", "from notebook re-run (latest build)")))


def cs_solver(b):
    return ("claude-sd", b)


if __name__ == "__main__":
    recs = []

    # NB41 SP#4 — mx(1ns)
    d = load("41")
    if d:
        for c in d.get("cs", []):
            emit(recs, "SP4_accuracy", "claude-sd", c["build"], "mx_1ns",
                 c.get("mx_1ns"), "SP4_mx_1ns", dim="2D", cells=10000)
        mp = d.get("mumaxplus") or {}
        if mp.get("mx_1ns") is not None:
            emit(recs, "SP4_accuracy", "mumax+", "f32", "mx_1ns", mp.get("mx_1ns"),
                 "SP4_mx_1ns", dim="2D", cells=10000)

    # NB46 SP#1 — L_c
    d = load("46")
    if d:
        for c in d.get("cs", []):
            emit(recs, "SP1_Lc", "claude-sd", c["build"], "L_c_nm", c.get("Lc_nm"),
                 "SP1_Lc_nm", integrator="relax")
        mp = d.get("mumaxplus") or {}
        if mp.get("Lc_nm") is not None:
            emit(recs, "SP1_Lc", "mumax+", "f32", "L_c_nm", mp.get("Lc_nm"),
                 "SP1_Lc_nm", integrator="relax")

    # NB47 SP#3 — H_sw (minimize)
    d = load("47")
    if d:
        for c in d.get("cs", []):
            emit(recs, "SP3_Hsw", "claude-sd", c["build"], "H_sw_mT", c.get("H_sw_mT"),
                 "SP3_Hsw_mT", integrator="relax")
        mp = d.get("mumaxplus") or {}
        if mp.get("H_sw_mT") is not None:
            emit(recs, "SP3_Hsw", "mumax+", "f32", "H_sw_mT", mp.get("H_sw_mT"),
                 "SP3_Hsw_mT", integrator="relax")

    # NB48 FMR — peak frequency
    d = load("48")
    if d:
        for c in d.get("cs", []):
            v = c.get("f_peak_GHz", c.get("f_peak"))
            emit(recs, "FMR_freq", "claude-sd", c["build"], "f_FMR_GHz", v,
                 "FMR_GHz", dim="macrospin", integrator="RK4")
        mp = d.get("mumaxplus") or {}
        v = mp.get("f_peak_GHz", mp.get("f_peak"))
        if v is not None:
            emit(recs, "FMR_freq", "mumax+", "f32", "f_FMR_GHz", v, "FMR_GHz",
                 dim="macrospin", integrator="RK45-DP")

    # mumax3 accuracy values (the NBs parse mumax3 TIMING; physics observables
    # are sparse in the json). Add the reliably-available ones.
    d48 = load("48")
    if d48 and d48.get("mumax3_fpeak_GHz"):
        emit(recs, "FMR_freq", "mumax3", "f32", "f_FMR_GHz",
             d48["mumax3_fpeak_GHz"], "FMR_GHz", dim="macrospin", integrator="RK45-DP")
    # SP#3 H_sw: mumax3 minimize() = -13.3 mT (NB47 console / BENCHMARK_REPORT.md)
    emit(recs, "SP3_Hsw", "mumax3", "f32", "H_sw_mT", -13.3, "SP3_Hsw_mT",
         integrator="relax", notes="mumax3 minimize(); from NB47 re-run")

    # NB44 DW velocity at J = 2e12 A/m^2 (dynamics accuracy)
    d = load("44")
    if d:
        def vat(c):
            Js = c.get("J_vals", []); vs = c.get("v_list", [])
            for j, v in zip(Js, vs):
                if abs(j - 2e12) < 1e10:
                    return v
            return None
        for c in d.get("cs", []):
            emit(recs, "DW_v_2e12", "claude-sd", c["build"], "v_DW_mps", vat(c),
                 dim="2D", cells=8000, integrator="Heun")
        mp = d.get("mumaxplus") or {}
        if mp.get("v_list"):
            # mumax+ result lacks J_vals here; assume same J grid as CS
            cs0 = (d.get("cs") or [{}])[0]
            mp2 = dict(mp); mp2["J_vals"] = cs0.get("J_vals", [])
            emit(recs, "DW_v_2e12", "mumax+", "f32", "v_DW_mps", vat(mp2),
                 dim="2D", cells=8000, integrator="RK45-DP")

    # Idempotent: drop any pre-existing accuracy records before re-inserting.
    import json as _json
    existing = _json.loads(rio.STORE.read_text()) if rio.STORE.exists() else []
    kept = [r for r in existing if r.get("metric") != "accuracy"]
    rio.STORE.write_text(_json.dumps(kept, indent=2))
    rio.append_many(recs)
    print(f"merged {len(recs)} accuracy records into {rio.STORE} "
          f"(purged {len(existing)-len(kept)} old accuracy records)")
    for r in recs:
        e = f" err={r['error_vs_ref']:.4f}" if r["error_vs_ref"] is not None else ""
        print(f"  {r['scenario']:<14} {r['solver']:<10} {r['build']:<10} "
              f"{r['observable']:<10} = {r['value']}{e}")
