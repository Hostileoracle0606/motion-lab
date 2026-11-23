# Experiment 04 independent reproduction

The [published study](https://github.com/Hostileoracle0606/motion-lab/releases/tag/experiment-04-study-v1)
is the reference for this same-host reproduction. Its checksum and download
instructions are in the [experiment README](README.md#download-and-audit-the-published-study).
The record below separates retraining, replay of the released checkpoints, and
regeneration of reports from saved predictions.

## Source and runtime

| Role | Commit |
| --- | --- |
| Original training and evaluation | `6844dd259e31a03ff171589be2001b4bdcb1b4b6` |
| Published checkpoint replay and report generator | `8fb57a75590724e19a5986c8855088242276393f` |
| Study release linked from main | `cc3d014706ef1bf3330bd1879c064b315dee97da` |

The run used an Apple M4 Mac (Mac16,12), arm64 Darwin 25.4.0, 10 logical CPU
cores, Python 3.12.15, PyTorch 2.15.0, NumPy 2.5.4 and MuJoCo 3.16.0. Use
`uv sync --locked` in each checkout. The original-source suite passed 743
tests with `uv run --frozen pytest -q` using a fresh pytest temporary
directory. Ordinary package imports and CLI help for training, studies,
evaluation, replay, reporting and benchmarking passed. The 747-test result
belongs to the later publication branch.

## Reproduce the three gates

1. Retrieve the published evidence in a checkout pinned to the report-generator
   commit and verify its archive and member hashes. From that checkout, run
   the published checkpoint replay and regenerate its report:

   ```bash
   experiment=experiments/04-learned-dynamics
   uv run --frozen python "$experiment/scripts/replay_checkpoints.py" \
     --input "$experiment/results/evaluation-v1" --device recorded
   uv run --frozen python "$experiment/scripts/generate_report.py" \
     --input "$experiment/results/study-v1" --output "$experiment/results/report-rebuilt"
   ```

2. Create a separate, clean checkout at the original training/evaluation
   commit. Run the frozen validation phase and then its final test. The
   paths are coupled to the versioned configuration. Use fresh result
   directories and supply observed launch conditions for your own run.

   ```bash
   experiment=experiments/04-learned-dynamics
   uv run --frozen python "$experiment/scripts/run_studies.py" \
     --config "$experiment/configs/studies-v1.toml" \
     --output "$experiment/results/study-v1" --phase validation \
     --power-state "<observed power>" --competing-load "<observed load>"
   uv run --frozen python "$experiment/scripts/run_studies.py" \
     --config "$experiment/configs/studies-v1.toml" \
     --output "$experiment/results/study-v1" --phase final_test
   ```

3. From the report-generator checkout, use its same `generate_report.py`
   command with the independently trained checkout's `results/study-v1`
   as input and a fresh output directory. Keep the training, evaluation and
   benchmark directories beside that study directory.

## Comparison results

- **Retraining.** Both rosters succeeded on all 45 declared runs, including
  the 15 diagnostic runs. Matched validation MAE deltas were zero for both
  position and velocity under the declared bounds of `1e-4 rad` and
  `1e-3 rad/s`. Training losses, validation MSE, and optimizer schedules
  matched over 13,500 paired epochs. All selected epochs, weight archive
  hashes, and normalizer hashes matched.
- **Checkpoint replay.** The recorded-device CLI verified 30 candidates and
  144,000 learned forecast attempts, including unavailable starts, within the
  declared per-output bounds of `1e-6 rad` and `1e-5 rad/s`.
- **Final test.** The independently generated evaluation directory matched all
  five published files byte-for-byte, including 230,400 prediction records,
  288 top-level summaries and the manifest. The independent comparison found
  zero learned endpoint and summary metric deltas. The records include
  132,700 successful learned attempts, 11,300 unavailable learned attempts,
  79,620 successful baseline attempts and 6,780 unavailable baseline attempts.
- **Report generation.** The rebuild from published records matched all 13
  output files other than the regenerated provenance. The report from new
  training matched eight files byte-for-byte, including all five prediction or
  coverage CSVs, plot data and both PNGs. The plotted error values matched
  exactly. The 13,500 learning-curve rows and 45 study-summary rows differed
  only in elapsed training time. The other changed report outputs were the
  timing CSVs and the timing section of the Markdown. Provenance and manifest
  checksums record the new run and report location.

The [machine-readable comparison](reproduction-v1.json) records the checked
counts, outcome deltas and file digests. Compare the SHA-256 of a regenerated
`evaluation-v1/records.jsonl` and `evaluation-v1/manifest.json` to the
values in that file. The release provides the full comparison inputs.

## Evidence boundary

Both training manifests record the clean original source commit. The
evaluation manifests preserve a dirty-worktree flag because generated
untracked outputs were present. The tracked source and configuration remained
unchanged. Independently generated benchmark files retain 24 workloads and
2,400 timing samples. The reproduction's timing metadata records launch
conditions, so the power and load at benchmark execution are unverified.
Timing equality is outside the numerical reproduction gates.

The held-out test covers two simulated episodes in one narrow direction and
displacement scenario. The training dataset includes joint-limit failures.
Training-seed comparisons and overlapping forecast windows share the same
simulated data. Broader plant or sensor shifts, physical hardware transfer and
closed-loop planning are separate experimental questions.
