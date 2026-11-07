"""Narrative conventions and limitations derived from audited evidence."""

from forward_dynamics.evidence import EvaluationEvidence
from forward_dynamics.report_tables import table_rows


def report_markdown(evidence: EvaluationEvidence) -> str:
    payload = evidence.evaluation
    provenance = payload["provenance"]
    runtime = provenance["runtime"]
    parameters = payload["plant_parameters"]
    seeds = sorted({row["seed"] for row in payload["episode_outcomes"]})
    lines = ["# Forward dynamics comparison", "",
        f"Scope: `{payload['scope']}`. Evaluator commit: `{provenance['evaluator_commit']}`.",
        f"Working tree dirty: `{provenance['working_tree_dirty']}`. Dataset source: `{provenance['dataset_source_commit']}`; builder: `{provenance['dataset_builder_commit']}`.",
        f"Runtime: Python {runtime['python']}, MuJoCo {runtime['mujoco']}, {runtime['os']} {runtime['architecture']}.",
        f"Dataset: `{payload['protocol']['dataset_version']}`; seeds: `{seeds}` (fixed before comparison). Model SHA-256: `{parameters['environment_version']}`.",
        f"Known effective inertia: {parameters['inertia_kg_m2']} kg m²; damping: {parameters['damping_nm_s_per_rad']} N m s/rad. Analytical RK4 steps: `{payload['protocol']['analytical_timesteps_s']}` s.",
        "", "This report is derived from independently audited saved predictions. The input bundle's `evaluation.json` retains input SHA-256 checksums, runtime, adapter settings and the frozen protocol; `manifest.json` retains output checksums.",
        "", "## Protocol and interpretation", "",
        "All six adapters use the same validated cases and eligible windows. Predictors receive only state, applied torque and context. No fitting, normalization or selection using test outcomes occurs. The control interval is 0.010 s; MuJoCo 3.16.0 uses five 0.002 s RK4 steps per interval.",
        "Position residual = predicted minus expected in radians, directly without wrapping for this limited hinge. Velocity residual uses rad/s. MAE, RMSE and maximum absolute errors are separate for each component; split summaries weight transitions or windows equally. CSV episode rows retain dispersion across episodes, plus controller/scenario/split summaries.",
        "The analytical RK4 baselines use known parameters from the checksummed plant, independently bound to a committed parameter contract and cross-checked against compiled extraction during generation. They omit MuJoCo soft joint-limit and contact forces. Finer integration steps address ODE integration error, not omitted physics; these coefficients are neither learned nor fitted.",
        "Rollouts recursively feed predicted states under the original recorded applied actions; intermediate observations never reset predictions. Complete windows can end on a terminal transition but cannot cross one. Short failed episodes and unavailable starts are retained; empty CSV metrics mean no eligible windows. Overlapping windows are correlated and are not independent trials.",
        f"MuJoCo agreement is a simulator consistency check with a {payload['protocol']['simulator_absolute_tolerance']} absolute tolerance in the recorded environment. This one-seed, narrow scenario dataset does not establish generalization, learned-model performance, control quality, cross-platform fidelity or physical-hardware validity.",
        "", "## Episode outcomes", "", "| Split | Episode | Transitions | Terminated | Truncated | Reason |", "|---|---|---:|---|---|---|"]
    for row in payload["episode_outcomes"]:
        lines.append(f"| {row['split']} | {row['episode_id']} | {row['transition_count']} | {row['terminated']} | {row['truncated']} | {row['termination_reason']} |")
    lines += ["", "## Split comparison", "", "All component metrics and coverage counts are in [one_step.csv](one_step.csv) and [rollouts.csv](rollouts.csv). Plots use a linear y-axis so exact zero reference errors remain visible. One-step labels show counts; rollout legends list counts in horizon order.", "",
        "| Adapter | Split | Horizon steps | Samples | Position RMSE (rad) | Velocity RMSE (rad/s) |", "|---|---|---:|---:|---:|---:|"]
    for row in table_rows(evidence, "rollouts"):
        if row["group"] == "split":
            lines.append(f"| {row['predictor_id']} | {row['split']} | {row['horizon_steps']} | {row['sample_count']} | {row['position_rmse_rad']:.8g} | {row['velocity_rmse_rad_s']:.8g} |")
    for name in ("one_step_position", "one_step_velocity", "rollout_position", "rollout_velocity"):
        lines += ["", f"![{name.replace('_', ' ')}]({name}.png)"]
    lines += ["", "## Reproduction", "", "```sh", provenance["command"], "uv run python experiments/03-forward-dynamics/scripts/generate_report.py --input <output-dir> --output <output-dir>/report", "```", ""]
    return "\n".join(lines)
