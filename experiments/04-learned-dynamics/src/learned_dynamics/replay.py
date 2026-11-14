"""Explicit checkpoint replay, separate from the saved-record semantic audit."""

from pathlib import Path

from forward_dynamics.dataset import load_prediction_cases
from forward_dynamics.serialization import require_equal
from learned_dynamics.checkpoint import load_checkpoint
from learned_dynamics.evaluation import EvaluationCandidate, evaluate
from learned_dynamics.evidence import load_evidence
from learned_dynamics.predictor import LearnedPredictor, SUPPORTED_CONTEXT
from learned_dynamics.training_evidence import load_run_manifest


def replay_checkpoints(input_dir: Path, device: str = "cpu") -> dict:
    evidence = load_evidence(input_dir)
    directory = Path(input_dir)
    roster_dir = (directory / evidence["training_roster_dir"]).resolve()
    live = load_run_manifest(roster_dir)
    snapshot = evidence["roster"]
    require_equal({key: live[key] for key in snapshot}, snapshot, "replay training roster")
    dataset = (directory / evidence["dataset_dir"]).resolve()
    all_cases = load_prediction_cases(dataset, evidence["dataset_version"])
    wanted = {(row["split"], row["episode_id"], row["step_index"])
              for row in evidence["case_ids"]}
    cases = tuple(case for case in all_cases
                  if (case.split, case.episode_id, case.step_index) in wanted)
    planned = {row["run_id"]: row["run_spec"] for row in snapshot["planned_runs"]}
    stored = tuple(row for row in evidence["records"]
                   if row["candidate"]["kind"] == "learned")
    replayed = []
    for result in snapshot["results"]:
        if result["status"] != "success":
            continue
        run_spec = planned[result["run_id"]]
        bundle = load_checkpoint(roster_dir / result["checkpoint_dir"], device=device)
        predictor = LearnedPredictor(bundle.model, bundle.normalizer, SUPPORTED_CONTEXT,
                                     result["run_id"], device)
        candidate = EvaluationCandidate.learned(
            predictor_id=result["run_id"], predictor=predictor, run_spec=run_spec,
            checkpoint_dir=result["checkpoint_dir"], checkpoint_sha256=result["checkpoint_sha256"],
            selected_epoch=result["selected_epoch"],
            selected_validation_mse=result["selected_validation_mse"],
            training_config_sha256=snapshot["training_config_sha256"])
        replayed.extend(evaluate(cases, (candidate,), (1, 5, 10, 25, 50, 100),
                                 run_spec["membership"]))
    require_equal(replayed, list(stored), "checkpoint replay predictions")
    return {"schema_version": 1, "replayed_candidates": len({r["predictor_id"] for r in replayed}),
            "replayed_records": len(replayed), "status": "verified"}
