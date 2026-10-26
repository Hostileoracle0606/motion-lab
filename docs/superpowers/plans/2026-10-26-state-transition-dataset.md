# Experiment 02 State-Transition Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a validated, scenario-split, deterministically replayed JSONL transition dataset from the committed Experiment 01 evidence.

**Architecture:** A new `state_transition_dataset` package consumes Experiment 01's validated `EvaluationEvidence`, converts each durable transition into a finite schema-v1 row, verifies saved actions by replaying them in MuJoCo, writes three deterministic JSONL partitions and a checksummed manifest, then validates the result as an independent consumer. Experiment 01 owns source trajectories; Experiment 02 owns dataset rows, split policy, replay evidence, and the dataset card.

**Tech Stack:** Python 3.12, standard-library JSON/TOML/hashlib, MuJoCo 3.16.0, uv, pytest

**Spec:** `docs/superpowers/specs/2026-10-26-state-transition-dataset-design.md`

## Global Constraints

- Preserve `(state_t, applied_torque_t) -> state_{t+1}` exactly.
- Use dataset schema version `1` and dataset version `experiment-02-transition-v1`.
- Split only by complete scenario; never split adjacent rows or paired controller episodes.
- Materialize finite scalar rows only; reject unsupported non-finite source states.
- Add no third-party dependency and make no learned-model or physical-hardware claim.
- Refuse dirty-tree publication by default; generated provenance identifies the clean builder commit.

## Review Focus

- A source episode with an unassigned or multiply assigned scenario must fail before any files are published; Task 3 tests both cases.
- A duplicate episode or transition appearing in two partitions must be rejected as leakage; Task 2 tests it.
- A tampered JSONL partition with a correct-looking manifest count but wrong bytes must fail checksum validation; Task 2 tests it.
- An early-terminated episode must retain one final boundary and remain valid; Task 2 tests it.
- Replay must compare the saved applied action—not raw controller output—and must fail outside tolerance; Task 4 tests both behaviors.

---

### Task 1: Package and versioned configuration

**Files:**
- Modify: `pyproject.toml`
- Create: `experiments/02-state-transition-dataset/configs/dataset-v1.toml`
- Create: `experiments/02-state-transition-dataset/src/state_transition_dataset/__init__.py`
- Create: `experiments/02-state-transition-dataset/src/state_transition_dataset/config.py`
- Create: `experiments/02-state-transition-dataset/tests/test_config.py`

**Interfaces:**
- Produces `DatasetConfig`, `SplitName`, and `load_dataset_config(path: Path) -> DatasetConfig`.
- `DatasetConfig` exposes resolved source evaluation/config paths, dataset version, replay tolerance, and a scenario-to-split mapping.

- [ ] Write failing tests for schema version `1`, dataset version, relative path resolution, the exact four-scenario assignment, and rejection of duplicate/missing scenario assignments or non-positive replay tolerance.
- [ ] Run `uv run pytest -q experiments/02-state-transition-dataset/tests/test_config.py`; expect import failure.
- [ ] Add both packages and both experiment test directories to `pyproject.toml`; implement the frozen config dataclasses and TOML loader.
- [ ] Run the focused tests and `uv run python -c "import feedback_control, state_transition_dataset"`; expect success.
- [ ] Commit with `feat: define Experiment 02 dataset configuration`.

### Task 2: Dataset schema, serialization, and validation

**Files:**
- Create: `experiments/02-state-transition-dataset/src/state_transition_dataset/schema.py`
- Create: `experiments/02-state-transition-dataset/src/state_transition_dataset/validation.py`
- Create: `experiments/02-state-transition-dataset/tests/test_validation.py`

**Interfaces:**
- Produces frozen `TransitionRow`, `EpisodeManifest`, `DatasetMetadata`, `PartitionManifest`, `DatasetManifest`, and `ValidatedDataset` dataclasses.
- Produces `write_dataset(output_dir: Path, metadata: DatasetMetadata, rows_by_split: Mapping[SplitName, tuple[TransitionRow, ...]]) -> DatasetManifest` and `load_and_validate_dataset(output_dir: Path, expected_dataset_version: str) -> ValidatedDataset`. The writer computes partition checksums/counts before serializing the final manifest.

- [ ] Write failing tests for deterministic round-trip serialization, exact fields, non-finite values, time/index continuity, state continuity, action limits, one final boundary, partition checksum tampering, undeclared membership, duplicate keys, and episode/scenario leakage.
- [ ] Run the focused file; expect missing-module failure.
- [ ] Implement schema-v1 JSON mapping and deterministic JSONL/manifest writers using sorted keys and `allow_nan=False`.
- [ ] Implement independent loading and validation of files, rows, episode summaries, partition checksums/counts, and leakage rules.
- [ ] Run focused tests; expect all pass.
- [ ] Commit with `feat: validate versioned transition datasets`.

### Task 3: Build rows from Experiment 01 evidence

**Files:**
- Create: `experiments/02-state-transition-dataset/src/state_transition_dataset/builder.py`
- Create: `experiments/02-state-transition-dataset/tests/test_builder.py`

**Interfaces:**
- Consumes `load_evaluation_evidence`, `DatasetConfig`, and Task 2 schema types.
- Produces frozen `RowBuild(rows_by_split, episodes, source_evaluation_sha256, source_code_version, model_path)` and `build_rows(config: DatasetConfig) -> RowBuild`. `episodes` retains the validated `EpisodeEvidence` needed by replay; Task 3 does not construct the final `EpisodeManifest` prematurely.

- [ ] Write failing tests that map complete scenarios to the declared splits, keep P/PD pairs together, preserve all temporal/action/provenance fields, retain early termination, and reject unassigned, multiply assigned, or non-finite source episodes.
- [ ] Run focused tests; expect missing builder failure.
- [ ] Implement evidence loading, row conversion, deterministic ordering, and per-episode manifest creation without rerunning MuJoCo.
- [ ] Run focused tests and the full Experiment 02 test directory; expect pass.
- [ ] Commit with `feat: materialize scenario-split transition rows`.

### Task 4: Deterministic applied-action replay

**Files:**
- Create: `experiments/02-state-transition-dataset/src/state_transition_dataset/replay.py`
- Create: `experiments/02-state-transition-dataset/tests/test_replay.py`
- Modify: `experiments/02-state-transition-dataset/src/state_transition_dataset/builder.py`
- Modify: `experiments/02-state-transition-dataset/src/state_transition_dataset/schema.py`

**Interfaces:**
- Produces `ReplayResult(max_position_error_rad: float, max_velocity_error_rad_s: float)` and `verify_action_replay(trajectory: TrajectoryRecord, model_path: Path, absolute_tolerance: float) -> ReplayResult`.
- Produces `build_episode_manifests(build: RowBuild, absolute_tolerance: float) -> tuple[EpisodeManifest, ...]`, invoking replay once per source episode and filling the replay fields already defined by Task 2.

- [ ] Write failing tests for exact deterministic replay, use of applied rather than raw torque, model checksum mismatch, invalid tolerance, and failure when a saved next state exceeds tolerance.
- [ ] Run focused tests; expect missing replay failure.
- [ ] Implement model/keyframe reset, recorded initial state, applied-action stepping, finite next-state comparison, and maximum-error reporting.
- [ ] Integrate replay results into episode manifests and dataset validation.
- [ ] Run focused tests and all Experiment 02 tests; expect pass.
- [ ] Commit with `feat: verify deterministic dataset action replay`.

### Task 5: CLI, dataset card, publication, and status

**Files:**
- Create: `experiments/02-state-transition-dataset/scripts/build_dataset.py`
- Create: `experiments/02-state-transition-dataset/tests/test_cli.py`
- Create: `experiments/02-state-transition-dataset/README.md`
- Create: `experiments/02-state-transition-dataset/DATASET_CARD.md`
- Add: `experiments/02-state-transition-dataset/data/dataset-v1/{manifest.json,train.jsonl,validation.jsonl,test.jsonl}`
- Modify: `README.md`
- Modify: `docs/roadmap.md`

**Interfaces:**
- Produces `build_dataset(config_path: Path, output_dir: Path, *, allow_dirty: bool = False) -> ValidatedDataset` and a CLI with `--config`, `--output`, and `--allow-dirty`.

- [ ] Write failing CLI tests for default paths, dirty-tree refusal, clean provenance, byte-identical repeated output, and validation after generation.
- [ ] Run focused tests; expect missing script/orchestrator failure.
- [ ] Implement orchestration: capture full Git SHA/dirty state/command, call Tasks 3–4, compute partition checksums, write, then independently reload/validate.
- [ ] Write the experiment README and dataset card with collection policy, exact row schema, units/coordinate frame, action timing, split rationale, coverage/counts, provenance, replay tolerance/results, reproduction commands, failures, limitations, and no model-performance claim.
- [ ] Commit implementation and documentation before publication; run `uv sync && uv run pytest -q` from the clean commit.
- [ ] Generate the default dataset twice and compare SHA-256 manifests byte-for-byte; commit the four generated files unchanged.
- [ ] Reproduce from a separate clean worktree at the manifest's builder commit and compare all four files byte-for-byte.
- [ ] Update root README and roadmap status only after reproduction succeeds; validate all local links and run the full suite.
- [ ] Commit with `docs: publish Experiment 02 transition dataset`.

