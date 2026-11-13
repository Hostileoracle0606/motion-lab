# Stage 4 progress and decision record — 2026-11-13

## Scope and source of authority

The approved [design](../superpowers/specs/2026-11-10-learned-dynamics-design.md)
and [implementation plan](../superpowers/plans/2026-11-11-learned-dynamics.md)
remain the protocol. This record makes the current delivery and process
decisions durable. The user requested progress stewardship, resolution of
open questions and PRs, and documentation of decisions; the recurring
check-in was changed to every two hours, with this check requested immediately.

Implementation at reviewed source `f99d383b8aa1e2ecccc439208180094b57d394e0`
covers tasks 1–4 (#24–#26). Training, measured accuracy, timing comparisons,
and Stage 4 reproduction remain subsequent work.

## Delivery decisions

1. **Preserve ancestry while integrating in dependency order.**
   [PR #32](https://github.com/Hostileoracle0606/motion-lab/pull/32) carries
   approved protocol commit `faccef8`,
   [PR #33](https://github.com/Hostileoracle0606/motion-lab/pull/33) carries
   the data foundation through `9abab2c`, and
   [PR #34](https://github.com/Hostileoracle0606/motion-lab/pull/34) carries
   the predictor/checkpoints through `f99d383`. Merge commits preserve these
   source identities for future provenance checks. Retarget each dependent PR
   to `main` only after its parent is integrated. Gate integration on review
   and a successful repository suite; absence of configured CI is not a test result.
2. **Continue #27 in an isolated branch while integration is verified.**
   Its approved scope is strict training configuration, reproducible seeded
   orchestration, validation-only selection, failure evidence and publication.
   Short configured smoke runs belong here; the complete 45-run study remains
   with #29. Parent branches stay unchanged during the parallel work.
3. **Keep consequential decisions in tracked documentation.** Local
   `artifacts/` verification records are ignored by Git. Preserve their useful
   conclusions here and in PR descriptions; local files alone are insufficient
   as a durable published decision history.
4. **Defer a speculative type-module split.** Standards review flagged a
   possible divergent-change smell in `learned_dynamics/types.py`: episode
   checks, normalizer records, windows and model specification share a module.
   It found no hard standards violation. Keep the existing public interface
   for this integration; reconsider cohesion if #27 introduces a concrete
   ownership or maintenance problem.
5. **Resolve the old #11 documentation mismatch.** Its remaining issue was a
   focused command naming absent `test_types.py`; equivalent shared validation
   already lives in `test_baselines.py`. Align the historical plan and issue
   command with the implemented coverage, then close #11 after verification.
6. **Resume interrupted work from its saved state.** The first #27 turn
   stopped at an account usage limit before creating its implementation
   worktree. After the reported retry time passed, the existing chat was
   resumed with instructions to inspect saved progress and avoid duplicate
   work. No account purchase or credit-reset action was taken.

## Technical decision index

These implemented choices refine or realize the approved protocol; details
and public interfaces are in the [Experiment 04 README](../../experiments/04-learned-dynamics/README.md).

| Decision | Reason and consequence |
| --- | --- |
| Retain independently validated scenario partitions and terminal transitions | Preserve the immutable dataset and failure evidence; no repartitioning. |
| Accept only training views for fitting | Prevent accidental validation/test preprocessing leakage. |
| Fit float64 population statistics on 940 ordered training rows | Fixed numeric algorithm and provenance; below `1e-12`, use scale 1.0 with a constant-feature flag. |
| Persist ordered episode/step identities with hashes and statistics | Later checkpoints and audits must retain which rows were fitted. |
| Construct episode-local windows and account for unavailable future before missing history | Retain terminal endpoints and consistent eligibility counts: train 940/936/924 for canonical/history/five-step membership. |
| Use a seeded float32 residual tanh MLP and physical-unit inference adapter | Implement the approved delta-state model; explicit history inputs preserve stateless inference. |
| Use explicit MPS or CPU selection without silent fallback | MPS remains the selected primary device; CPU supports comparison and diagnosis. |
| Pin PyTorch 2.15.0, retaining previous package versions | Real CPU/MPS operation was checked against the resolved environment. |
| Use fixed JSON/NPZ inference-only checkpoint bundles | Exact fields, tensor names/shapes/dtypes and hashes can be audited without pickle. Resume is unsupported; restart from scratch. |
| Bound JSON files to 2 MiB and tensor archives to 16 MiB | Check resource limits before hashing/loading; larger architectures require an explicit schema/budget decision. The study-wide 500 MiB budget remains separate. |
| Bind loading to the original validated dataset and independently refit its train normalizer | Verify hashes, fitted identities and numeric statistics. The repository and dataset must remain available. |
| Treat source/config/seed metadata as declared provenance | Hashes and a resolvable commit do not authenticate how weights were trained; later training evidence and reproduction must substantiate those claims. |
| Validate staging before exclusively reserving a fresh output | Refuse overwrite and protected inputs. Copying is not crash-atomic: reject incomplete bundles and retry in a new directory. |

## Verification record

The originating #25 record reports 579 passing repository tests and 63 new
focused tests. The #26 record reports 120 focused and 636 repository tests
passing at `f99d383`, including actual MPS forward/backward/Adam and prediction
checks on macOS 26.4 arm64, Python 3.12.15 and torch 2.15.0. These are prior
implementation-run results, not independent reproduction of a learned study.

The current integration audit reviews `84d28e2...f99d383` on two axes:

- **Standards:** no hard violations; one nonblocking module-cohesion judgement
  recorded above.
- **Spec:** no findings against approved tasks 1–4 and issues #24–#26.

A fresh `uv run --frozen pytest -q` at `f99d383` passed **636 tests in
378.99 s**, exit 0. Ordinary package imports and both explicit CPU and MPS
smoke commands passed; each fitted 940 training rows and emitted finite
synthetic predictions. The corrected #11 focused command passed **102 tests
in 2.59 s**. `git diff --check` passed. Commands ran in the clean #26 worktree;
this is integration verification, not a fresh-environment reproduction.

Merge identities are recorded in the linked PR discussions and Git history.
These checks establish implementation behavior, not learned accuracy, GPU
speedup, physical fidelity or Stage 4 completion.
