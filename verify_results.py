"""Audit saved data, model selection, certificates, and source provenance."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def main():
    root=Path(__file__).resolve().parent
    out=root/"results"
    meta=json.loads((out/"metadata.json").read_text())
    cfg=meta["config"]
    b=pd.read_csv(out/"benchmark.csv")
    sweep=pd.read_csv(out/"lambda_sweep.csv")
    selected=pd.read_csv(out/"selected_models.csv")
    refs=pd.read_csv(out/"reference_checks.csv")
    steps=pd.read_csv(out/"step_sizes.csv")
    expected=len(cfg["seeds"])*len(cfg["scenarios"])
    assert len(sweep)==expected*len(cfg["lambda_ratios"])
    assert len(selected)==expected
    assert len(b)==2*len(cfg["seeds"])*len(cfg["benchmark_rhos"])
    assert len(refs)==expected+len(b)//2
    for df in [b,sweep,selected,steps]:
        numeric=df.select_dtypes(include="number").to_numpy()
        assert np.isfinite(numeric).all()
        assert df.relative_kkt.max()<=cfg["tol"]*(1+1e-10)
        assert df.relative_gap.max()<=cfg["tol"]*(1+1e-10)
        if "converged" in df:
            assert df.converged.all()
    assert selected.ista_relative_kkt.max()<=cfg["tol"]
    assert selected.ista_relative_gap.max()<=cfg["tol"]
    assert refs.relative_kkt.max()<=1e-8
    assert refs.relative_gap.max()<=1e-8
    for row in selected.itertuples():
        path=sweep[(sweep.scenario==row.scenario)&(sweep.seed==row.seed)]
        winner=path.loc[path.validation_mse.idxmin()]
        assert row.lambda_ratio==winner.lambda_ratio
    for path in ["lasso.py","data.py","run_experiments.py","experiment_config.json"]:
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==meta["source_sha256"][path]
    trace_audit={}
    for path in sorted((out/"traces").glob("*.csv")):
        h=pd.read_csv(path)
        assert np.isfinite(h.select_dtypes(include="number").to_numpy()).all()
        increments=np.diff(h.objective)
        rises=int(np.sum(increments>1e-12))
        if path.name.startswith(("ISTA","step")):
            assert rises==0
        trace_audit[path.name]=dict(iterations=int(h.iteration.iloc[-1]),
                                   objective_increases_above_1e_minus12=rises)
    d=np.load(out/"base_problem.npz")
    sol=np.load(out/"base_solutions.npz")
    pred=d["A_test"]@sol["x_selected"]+d["response_mean"]
    mse=float(np.mean((pred-d["y_test"])**2))
    row=selected[(selected.scenario=="base")&(selected.seed==cfg["seeds"][0])].iloc[0]
    assert np.isclose(mse,row.test_mse,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(d["A"].mean(axis=0),0,atol=1e-13)
    np.testing.assert_allclose(d["A"].std(axis=0),1,atol=1e-13)
    checks=dict(status="PASS",candidate_fits=len(sweep),selected_models=len(selected),
                benchmark_rows=len(b),reference_problems=len(refs),
                all_values_finite=True,all_stopping_criteria_met=True,
                selection_reproduced_from_validation_only=True,
                source_hashes_match=True,base_snapshot_mse=mse,
                max_checked_objective_difference=float(max(b.objective_difference.abs().max(),
                    selected.objective_difference.abs().max(),selected.ista_objective_difference.abs().max())),
                traces=trace_audit)
    (out/"validation.json").write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
