# Experiment 01 Completion Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Reproduce Experiment 01 from a clean checkout, publish its complete deterministic evidence set, and update completion status only after every gate passes.

**Architecture:** Treat the clean plan-only commit created in Task 1 as the evaluated source version. Generate the already-defined evaluation and report artifacts from that clean tree, verify exact repeatability, commit the artifacts unchanged, then update documentation status. No simulation, controller, metric, evaluator, or reporting behavior changes are in scope.

**Tech Stack:** Python 3.11, uv, pytest, MuJoCo, Matplotlib, Git, GitHub CLI

**Spec:** `docs/superpowers/specs/2026-10-24-experiment-01-completion-design.md`

## Global constraints

- Do not change executable source, tests, controller gains, evaluation scenarios, or reporting logic.
- Do not claim physical-robot evidence, timing performance, optimality, or production readiness.
- Generated evidence must retain clean-tree provenance for the evaluated source commit.
- Status changes and issue closures happen only after independent clean reproduction succeeds.

### Task 1: Prove clean deterministic reproduction

**Files:**
- Generate: `experiments/01-feedback-control/results/evaluation-v1/**`
- Record privately: SHA-256 manifests under this plan's ignored SDD workspace

**Interfaces:** Produces the exact evidence files and manifest consumed by Task 2.

**Step 1: Verify the publication gate is initially red**

Run: `git ls-files 'experiments/01-feedback-control/results/evaluation-v1/**'`

Expected: no output, proving the evidence is not yet published.

**Step 2: Commit the plan and design so the source tree is clean**

Run:

```bash
git add docs/superpowers/specs/2026-10-24-experiment-01-completion-design.md \
  docs/superpowers/plans/2026-10-24-experiment-01-completion.md
git commit -m "docs: plan Experiment 01 completion gate"
```

Expected: documentation commit succeeds and the tracked worktree is clean.

**Step 3: Install and verify the clean source tree**

Run: `uv sync && uv run pytest -q`

Expected: dependency sync succeeds and all tests pass.

**Step 4: Generate the evaluation and report using the documented commands**

Run:

```bash
uv run python experiments/01-feedback-control/scripts/run_evaluation.py
uv run python experiments/01-feedback-control/scripts/generate_report.py \
  --input experiments/01-feedback-control/results/evaluation-v1 \
  --config experiments/01-feedback-control/configs/evaluation-v1.toml \
  --output experiments/01-feedback-control/results/evaluation-v1/report
```

Expected: four scenarios, eight successful episodes, aggregate evaluation JSON, and a flat report directory containing one Markdown report, three CSVs, and five PNGs.

**Step 5: Verify same-checkout repeatability**

Create a sorted SHA-256 manifest in the ignored SDD workspace, rerun both commands, and compare the new manifest with the first.

Expected: manifests are identical.

**Task completion test:** `uv run pytest -q`

### Task 2: Publish and independently reproduce the evidence

**Files:**
- Modify: `.gitignore`
- Add: `experiments/01-feedback-control/results/evaluation-v1/evaluation.json`
- Add: `experiments/01-feedback-control/results/evaluation-v1/trajectories/*.json`
- Add: `experiments/01-feedback-control/results/evaluation-v1/report/report.md`
- Add: `experiments/01-feedback-control/results/evaluation-v1/report/*.csv`
- Add: `experiments/01-feedback-control/results/evaluation-v1/report/*.png`

**Interfaces:** Consumes Task 1's exact evidence files and manifest; produces the tracked evidence checked by Task 3.

**Step 1: Make the selected evidence set trackable**

Remove only the `.gitignore` rule for `experiments/*/results/evaluation-v1/`. Keep ad hoc trajectory and demo outputs ignored.

Expected: `git status --short` shows exactly the plan/design, `.gitignore`, and the selected evaluation evidence.

**Step 2: Validate the evidence contract**

Check that the evaluation and report identify the source commit, clean tree, environment, config checksum, scenario IDs, seeds, failures, limitations, and unknowns. Validate every relative Markdown link in the report.

Expected: all required fields are present, the four declared P-controller failures and their reasons remain explicit, all four PD episodes pass, and all local links resolve.

**Step 3: Commit the evidence unchanged**

Run:

```bash
git add .gitignore experiments/01-feedback-control/results/evaluation-v1
git commit -m "docs: publish Experiment 01 evaluation evidence"
```

Expected: evidence commit succeeds and the worktree is clean.

**Step 4: Reproduce from an independent clean checkout**

Create a temporary detached worktree at the recorded source commit, run `uv sync`, the full test suite, evaluation, and report commands, then compare its sorted SHA-256 manifest with Task 1's published manifest.

Expected: tests pass and every generated file is byte-identical. Remove the temporary worktree after comparison.

**Task completion test:** `uv run pytest -q`

### Task 3: Mark Experiment 01 complete and verify the repository contract

**Files:**
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `experiments/01-feedback-control/README.md`

**Interfaces:** Consumes Task 2's tracked, independently reproduced evidence and exposes its status and commands to readers.

**Step 1: Verify the status gate is initially red**

Run a search for the current `In progress` and unchecked completion-gate text in the three status surfaces.

Expected: the old status and two unchecked checklist items are present.

**Step 2: Update only evidence-supported status language**

Mark Experiment 01 complete, link the published report/evaluation evidence, check the remaining gate items, and retain the simulation-only limitations.

Expected: all three status surfaces agree and no claim exceeds the one-link simulation scope.

**Step 3: Verify links, evidence, and tests**

Run the full test suite, validate repository-local Markdown links in the modified documentation and published report, confirm all expected evidence files are tracked, and confirm `git diff --check` is clean.

Expected: all tests and checks pass.

**Step 4: Commit the completion status**

Run:

```bash
git add README.md docs/roadmap.md experiments/01-feedback-control/README.md
git commit -m "docs: mark Experiment 01 complete"
```

Expected: completion commit succeeds and the worktree is clean.

**Task completion test:** `uv run pytest -q`

## Review focus

- Confirm the committed evidence exactly matches a clean run at its recorded source commit.
- Confirm every report source link resolves to a committed trajectory.
- Confirm the documentation distinguishes evaluated source commit from evidence/status commits.
- Confirm status wording does not imply physical-robot, real-time, production, or optimality evidence.
- Confirm Issue #7's dependencies are satisfied before any issue is closed.

