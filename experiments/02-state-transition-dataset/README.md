# Experiment 02 — Versioned State-Transition Dataset

## Status

> **Complete — the schema-v1 scenario-split dataset was reproduced byte-for-byte from its clean builder commit.**

Experiment 02 turns the validated Experiment 01 trajectories into a directly
consumable dynamics dataset. It preserves the transition contract

```text
(state_t, applied_torque_t) -> state_{t+1}
```

and splits complete scenarios rather than individual rows. The published
artifact is a data-engineering result; it does not train or evaluate a learned
model.

See the [dataset card](DATASET_CARD.md) for the exact schema, collection policy,
coverage, split rationale, provenance, replay evidence, and limitations.

## Published artifact

The versioned configuration is
[`configs/dataset-v1.toml`](configs/dataset-v1.toml). The generated artifact is
under [`data/dataset-v1/`](data/dataset-v1/):

- `manifest.json` records checksums, counts, source and builder provenance, and
  episode-level replay results;
- `train.jsonl` contains both wide-displacement scenarios;
- `validation.jsonl` contains the negative-to-positive narrow scenario;
- `test.jsonl` contains the positive-to-negative narrow scenario.

The split keeps P and PD episodes for the same scenario together and prevents
adjacent transitions from crossing partitions.

## Build and validate

From the repository root:

```bash
uv sync
uv run pytest -q
uv run python experiments/02-state-transition-dataset/scripts/build_dataset.py
```

The builder refuses a dirty working tree by default. `--allow-dirty` exists for
local development and records that state in the manifest; it must not be used
to publish clean provenance. Custom locations are available through `--config`
and `--output`.

The build does four things in order:

1. independently loads and validates the committed Experiment 01 evaluation;
2. materializes finite schema-v1 rows with scenario-level membership;
3. replays every saved applied-action sequence in the checksummed MuJoCo model;
4. writes deterministic partitions, then reloads and validates them as a
   consumer.

## Reproduce the committed bytes

The generated manifest identifies the clean builder commit. Because the
generated files are committed afterward, byte reproduction uses a separate
worktree at that recorded commit:

```bash
set -euo pipefail
published_checkout="$(pwd -P)"
builder_commit="$(jq -r '.metadata.builder_code_version' \
  experiments/02-state-transition-dataset/data/dataset-v1/manifest.json)"
reproduction_parent="$(mktemp -d)"
reproduction_checkout="$reproduction_parent/motion-lab-builder"

git worktree add --detach "$reproduction_checkout" "$builder_commit"
cd "$reproduction_checkout"

uv sync
uv run pytest -q
uv run python experiments/02-state-transition-dataset/scripts/build_dataset.py

diff -qr \
  "$published_checkout/experiments/02-state-transition-dataset/data/dataset-v1" \
  "$reproduction_checkout/experiments/02-state-transition-dataset/data/dataset-v1"

cd "$published_checkout"
git worktree remove --force "$reproduction_checkout"
rmdir "$reproduction_parent"
```

A successful `diff -qr` exits with no output. The replay statement is limited
to the recorded deterministic simulator, model, dependency versions, and
numerical tolerance. It is not evidence about a physical robot.
