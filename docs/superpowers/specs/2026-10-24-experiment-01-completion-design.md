# Experiment 01 Completion Design

**Issue:** [#7 — Reproduce Experiment 01 from a clean checkout and mark it complete](https://github.com/Hostileoracle0606/motion-lab/issues/7)

## Intent

Move Experiment 01 from "implementation available" to "complete" only after a clean checkout reproduces its tests, evaluation records, report tables, and plots. Publish the evidence needed for another reader to audit the claim from the repository.

## Completion boundary

The completion claim covers one simulated single-link MuJoCo system, the committed P/PD controllers, four committed evaluation scenarios, and the report generated from those records. It does not claim physical-robot validation, real-time performance, production readiness, or controller optimality.

## Selected approach

Commit the complete `evaluation-v1` evidence set: the aggregate evaluation record, all eight source trajectory records, the Markdown report, three CSV tables, and five PNG plots. The set is small enough for this repository and preserves the report's source links and checksums.

Two alternatives were rejected:

1. Publishing only the report and summary would make the linked episode records unavailable and prevent an independent audit of the reported metrics.
2. Publishing to an external artifact store would add infrastructure and retention assumptions that this repository does not currently define.

## Evidence flow

1. Start from the exact clean source commit named in the generated provenance.
2. Install the locked environment and run the full test suite.
3. Generate evaluation records and the report using the documented commands.
4. Hash every generated file, rerun both commands, and require byte-identical hashes.
5. Commit those generated files unchanged.
6. Reproduce the same files again from a separate clean worktree at the recorded source commit and compare them byte-for-byte with the committed evidence.
7. Only then update status text and close the fulfilled Experiment 01 issues.

The evidence commit necessarily comes after the source commit it evaluates. Therefore `code_version` identifies the evaluated source tree, not the later commit that merely adds generated evidence and completion wording.

## Failure handling

- Any failing test blocks publication.
- Any missing artifact, broken local link, or hash mismatch blocks completion.
- Any generated provenance that reports a dirty tree blocks completion.
- A numerical or visual mismatch is treated as a reproduction failure, even if it appears small, because this evaluation contract is deterministic and currently uses zero tolerance.

## Verification surfaces

- `uv run pytest -q`
- documented evaluation and report commands
- byte-for-byte SHA-256 manifests across repeated and independent clean runs
- Markdown local-link validation
- repository status and issue dependency checks

