# Isambard Cloud Compute Access and Usage

| Purpose | Path |
| --- | --- |
| Training and generation Python | `/projects/u6oz/yuhe/jlens_belief-v1/.venv/bin/python` |
| Evaluation and operations Python | `/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python` |
| Hugging Face cache | `/projects/u6oz/yuhe/hf` |
| Shared submission lock | `/projects/u6oz/.jlens-subliminal-submit.lock` |
| Submission ledger | `/projects/u6oz/.jlens-subliminal-submissions-UID.json`, with the user's numeric UID |
| Existing compiler-cache archives | `/projects/u6oz/yuhe/compiler-cache-archives-20260918` |

Use the explicit prepared interpreter required by the stage. Training and evaluation use separate dependency environments; do not substitute the system Python or upgrade the shared installed environment.

Existing launchers use the prefetched model cache and offline execution:

```bash
export HF_HOME=/projects/u6oz/yuhe/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
```

Use each campaign's established launcher for its remaining environment settings and private working directory. Compiler scratch can be node-local when the launcher already supports it. Scientific data, configurations, checkpoints, scores, provenance and completion receipts must remain in persistent project storage.

## GPU accounting and submission

The **64-GPU aggregate limit supersedes historical 32-GPU instructions**. The shared guard conservatively counts all current-user jobs across accounts, including pending dependencies and unresolved submission reservations. Reconcile live scheduler state and the ledger before every submission.

Submit through the existing shared locked guard, `scripts/isambard/submit.py`, using the guard checkout named in the campaign's `HANDOFF.md`. Campaign-specific submitters must use the same lock and ledger. Do not bypass the guard with an independent submission path.

The code-teacher campaign records this guard checkout:

```text
/projects/u6oz/yuhe/jlens-single-claim-benchmarks-20260911-r1-ops-tqa-cap64-20260915
```

Prefer independent, useful jobs that are ready to run. Allocate additional GPUs to a job only if its implementation can use them. There is no explicit GPU-hour budget, but experiments should be staged and results inspected before expansion. Do not create unnecessary replicas simply to occupy 64 GPUs.

Before launching:

1. Read the campaign's current handoff, approved protocol and frozen configuration.
2. Inspect running jobs, pending requests, unresolved reservations and storage headroom.
3. Run the campaign's required preflight and matching GPU diagnostics.
4. Preview the submission and confirm its model, paths, dependencies and resource request.
5. Submit through the guard, then verify the recorded Slurm IDs and actual job startup.

Always pass **absolute paths** to `--campaign`, `--jobs`, data and output directories. Submission scripts save their arguments in `run.sh`, and workers run from a private working directory; relative campaign paths have caused startup failures.

For campaigns that provide `submit_jobs.py`, the following is a template. Replace the directory with the existing authorized campaign and use `--only JOB_ID ...` when submitting a bounded subset:

```bash
CAMPAIGN_DIR='/projects/u6oz/yuhe/REPLACE_WITH_AUTHORIZED_CAMPAIGN'
OPS_PYTHON='/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python'

# Preview; this does not submit jobs.
"$OPS_PYTHON" "$CAMPAIGN_DIR/submit_jobs.py" \
  --campaign "$CAMPAIGN_DIR/campaign.json" \
  --jobs "$CAMPAIGN_DIR/jobs.json"

# Submit the approved selection after preflight and capacity checks.
"$OPS_PYTHON" "$CAMPAIGN_DIR/submit_jobs.py" \
  --campaign "$CAMPAIGN_DIR/campaign.json" \
  --jobs "$CAMPAIGN_DIR/jobs.json" \
  --apply
```

The generic `scripts/isambard/submit.py` uses `--submit`, whereas the campaign wrapper above uses `--apply`. Check the selected entry point's help and campaign instructions.

## Status and storage checks

Run these read-only commands on Isambard:

```bash
# Current user's queued and running jobs.
squeue --me --format='%.18i %.35j %.12T %.10M %R'

# Project byte and file quota.
lfs quota -p 1483806624 /projects/u6oz

# Replace JOB_ID with the numeric Slurm ID.
scontrol show job JOB_ID
sacct --allocations --jobs JOB_ID \
  --format=JobIDRaw,State,ExitCode,Elapsed,Start,End
```

Use the guard's `inspect_capacity()` for aggregate capacity, rather than estimating capacity from the number of rows in `squeue`. A read-only capacity snapshot does not reserve GPUs; the submitter must recheck under the shared lock.

For the guard checkout recorded above:

```bash
PYTHONPATH=/projects/u6oz/yuhe/jlens-single-claim-benchmarks-20260911-r1-ops-tqa-cap64-20260915 \
/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -c \
'import json; from scripts.isambard.submit import inspect_capacity; print(json.dumps(inspect_capacity(), indent=2))'
```

The project has historically approached its file-count quota of roughly 51.2 million files. Check both file and byte headroom before checkpoint-writing jobs. Follow the campaign's preflight threshold; the code-teacher full-campaign preflight uses at least 10,000 free file slots. Do not treat a past quota measurement as current availability.

If storage recovery is needed, preserve models, datasets, checkpoints, results and running-worker files. Any archival of inactive compiler caches must first verify that the associated jobs have ended, then verify archived file contents and metadata before removing originals. Retain the archive and its verification receipt.

## Checkpointing, failures and resumption

- Save progress frequently and use durable stage-completion receipts and resumable training checkpoints.
- Preserve running workers, paused campaigns, failed attempts, logs and historical results. Use fresh output directories when a changed configuration would invalidate existing provenance.
- Reuse verified completed work after reconnecting. Inspect both Slurm state and persistent completion evidence before retrying a stage.
- If submission was interrupted before a job ID was recorded, reconcile the saved command, scheduler and ledger. Do not erase an uncertain reservation or submit again blindly.
- A failed diagnostic is separate from a scientific result. Investigate it before launching dependent work.
- Do not interpret every Slurm `FAILED` state as resumable. Use the campaign's interruption classification and checkpoint validation.
- Completed jobs may disappear from `scontrol`. Consult `sacct` and saved receipts. For the code-teacher submitter, the approved dependency repair omits an already-satisfied scheduler prerequisite only after verifying its completion, campaign and manifest hashes; execution-time checks remain required.

Keep the local `runs/<campaign>/HANDOFF.md` and execution record synchronized with the remote campaign directory. Record submitted job IDs, configuration and source provenance, completion evidence, outstanding work and exact resume commands.

## Approval and scientific integrity

Before augmenting or fixing an existing pipeline, investigate read-only and present the observed problem, evidence, exact proposed change, expected effect and alternatives, including retaining the original. Obtain explicit approval before applying an unapproved pipeline change. General authorization to run experiments does not authorize changing their scientific settings. Do not request approval again for a specific change already approved.

Keep operational recovery within the campaign's recorded authorizations. Preserve the original implementation and evidence when an approved change is applied, and never edit the shared installed environment.

Use the campaign's pinned model revisions, matching teacher/student base checkpoint and required attention checks. Keep smoke-test evidence separate from scientific results. Report behavioral outcomes and JLens/logit-lens outcomes separately; successful execution alone does not establish subliminal transfer.

## Source records

This export was compiled from `CLAUDE.md`, the user-supplied `AGENTS.md` instructions, `scripts/isambard/submit.py`, and the code-teacher campaign's launcher, submitter and `HANDOFF.md`. Historical README examples containing a 32-GPU ceiling or a Qwen-only restriction are superseded by the later approved project instructions. Consult the relevant campaign's current handoff for its live scope and resume state.

## Operational notes verified on 2026-09-29

- Stage large model downloads on compute nodes. The login service assigns hosts per connection and limits each session to one CPU core, 4 GiB memory, and 500 tasks. Save host, PID, Slurm ID, command, and log location in each transfer receipt. [Login guidance](https://docs.isambard.ac.uk/user-documentation/guides/login/), [resource limits](https://docs.isambard.ac.uk/user-documentation/faqs/#my-application-is-showing-resource-temporarily-unavailable-errors-on-login-nodes).
- Hugging Face authentication uses the account's rate limit instead of the cluster's shared anonymous IP limit. Use scoped credentials without logging them; do not assume every slow transfer is rate limiting. [Official guidance](https://docs.isambard.ac.uk/user-documentation/faqs/#why-am-i-getting-rate-limited-by-huggingface).
- CPU download allocations and GPU inference allocations are separate. The existing shared guard conservatively charges a zero-GPU CPU job as four GPUs; campaign submitters must preserve that accounting at submission time.
- In this campaign, `sbatch --test-only` predicted a long wait, but actual jobs started within seconds. Report submitted job states and observed throughput; treat queue projections as estimates.
