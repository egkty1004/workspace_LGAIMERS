# workspace_LGAIMERS operating policy

This repository supports the LG Aimers 9 pitch-control probability-prediction competition.

## Sources of truth

- The GitHub repository is the technical source of truth.
- The actual default branch is `master`.
- For nontrivial changes, do not work directly on `master`. Use a scoped task branch such as `exp/...`, `fix/...`, or `chore/...` unless the user explicitly requests otherwise.
- `AI_CONTEXT.md` contains stable technical context.
- `PROJECT_STATUS.md` contains mutable current project state.
- The current core implementation lives primarily under `repro_979/`.
- `baseline/` and `experiments/` contain official-baseline and older/historical work.
- Notion, when available, is a human-readable mirror rather than a technical SSOT.

## Git and artifact safety

- Never blindly run `git add -A`. Inspect `git status` and the intended diff before staging.
- Never commit or push without explicit user approval.
- Never commit raw data, model binaries, caches, large outputs, or submission zip artifacts unless explicitly required and verified.
- Existing tracked `.omo/evidence` files are historical evidence; do not assume that all of `.omo/` is untracked.
- Preserve unrelated user changes and do not rewrite history or delete artifacts without explicit authorization.

## Competition and ML safety

- Use official competition data only. The final solution must not depend on external data or external APIs.
- Treat each test row independently: do not use information from other test rows or apply test-distribution-based correction.
- Probability quality, measured through Brier Score/Brier Skill Score, is the primary evaluation concern.
- Leakage is a blocker. This includes target, temporal, split, preprocessing, terminal-label, and test-distribution leakage.
- Never change a validation protocol, scoring form, fold role, calibration rule, or promotion gate silently. Document and obtain agreement for the change.

## Execution-cost policy

- **CHEAP**: static inspection, syntax/type checks, and small unit tests.
- **MEDIUM**: smoke tests, small fixtures, and tiny-subset runs.
- **EXPENSIVE**: full CV, multi-seed training, 10,000-resample bootstrap, full deployment training, and full-size package inference/validation.
- EXPENSIVE commands require explicit user approval before execution.
- Do not run expensive legacy scripts merely because they exist. Confirm that the runner belongs to the active policy first.

## GPU and environment policy

- Never assume GPU 0 is free; inspect GPU availability before GPU work.
- School-server GPU performance is development evidence, not a substitute for the competition evaluation environment.
- Prefer the progression: smoke test, limited validation, then an explicitly approved full run.
- Final submission checks should reproduce the official CPU, memory, GPU, Python, offline, row-count, and time constraints as closely as practical.

## Working discipline

- For read/review tasks, do not modify files or external systems.
- For implementation tasks, modify only the authorized scope and verify proportionally to risk and cost.
- Before selecting or running an experiment, read `AI_CONTEXT.md`, `PROJECT_STATUS.md`, and the active policy/state files referenced there.

## Delegation policy

- The parent/orchestrator owns analysis, planning, task decomposition, project and ML policy interpretation, and final integration.
- Use the named `implementation_worker` only for concrete, bounded implementation and proportionate testing within the user-authorized scope.
- The implementation worker must not choose or change experiment, validation, scoring, calibration, recovery, promotion, leakage, or champion policy.
- The parent must review the worker's changes and test evidence before accepting or integrating them.
- Delegation does not weaken Git safety, leakage controls, execution-cost classifications, GPU policy, or user-approval requirements.
