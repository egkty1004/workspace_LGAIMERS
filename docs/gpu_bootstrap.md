# School-GPU development bootstrap

This workflow creates the **aimers9-dev** development environment and the
local, Git-ignored symlink bridges required by the current recovery code.
Passing it means **school-GPU development readiness only**. It does not
establish official submission compatibility, offline installability, resource
compliance, runtime compliance, package parity, or submission readiness.

The bootstrap does not run Git commands, select a GPU, load CSV rows, copy or
extract assets, or handle credentials.

## External prerequisites

- A Conda-compatible **conda** command on PATH with the **libmamba** solver
  installed and supported by `conda env create --solver libmamba`.
- Initial network access for package installation.
- An NVIDIA driver compatible with the CUDA 12.8 Torch build.
- Official competition data extracted outside Git. The canonical default is
  **<repo>/데이터/open/**, with these files directly inside it:
  - train.csv
  - test.csv
  - sample_submission.csv
- The authorized, already-extracted v93 directory outside Git.
- Optionally, the original authorized v93 archive. Supplying it enables archive
  SHA-256 verification; extracted assets alone cannot prove archive provenance.

Do not place raw data, model weights, share archives, caches, credentials, or
authentication files under version control.

## Read-only check

Use the canonical data default:

~~~bash
python3 scripts/bootstrap_gpu.py check \
  --v93-dir "/absolute/path/to/v93_extract_verify"
~~~

Or supply a different official-data location and optional archive explicitly:

~~~bash
python3 scripts/bootstrap_gpu.py check \
  --data-dir "/absolute/path/to/open" \
  --v93-dir "/absolute/path/to/v93_extract_verify" \
  --v93-archive "/absolute/path/to/submit_v93_r0476.zip" \
  --env-name aimers9-dev
~~~

**check** reads filenames, CSV headers, authoritative v93 artifact identities,
the optional archive, Conda metadata, package versions, and nvidia-smi output.
It does not create an environment or filesystem links. On a fresh account it
normally reports **NOT READY** until **apply** has completed.

## Explicit apply

~~~bash
python3 scripts/bootstrap_gpu.py apply \
  --data-dir "/absolute/path/to/open" \
  --v93-dir "/absolute/path/to/v93_extract_verify" \
  --v93-archive "/absolute/path/to/submit_v93_r0476.zip" \
  --env-name aimers9-dev
~~~

Omit **--v93-archive** when only extracted assets are available. In that case
the tool reports structural development readiness and explicitly reports that
archive provenance is not proven.

**apply** validates every external source and every destination before creating
links. If **aimers9-dev** is absent, it creates Python 3.11.15, installs the
pinned direct development dependencies, and installs Torch. Fresh environment
creation explicitly uses the server-verified solver command:

~~~text
conda env create --solver libmamba --name aimers9-dev \
  --file environment/aimers9-dev.yml
~~~

If libmamba is missing or unsupported, bootstrap fails at the named environment
creation stage; it does not silently retry with the classic solver. Package and
environment subprocesses retain the 600-second fail-closed timeout.

Torch uses this explicit source:

~~~text
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  torch==2.7.1+cu128
~~~

The post-install audit requires all of:

- **torch.__version__ == "2.7.1+cu128"**
- **torch.version.cuda == "12.8"**
- **torch.cuda.is_available() is True**

XGBoost 3.2.0 is installed for development only; it is not a submission
dependency.

## Repository-local bridges

Only these data links are created:

~~~text
repro_979/open/data/train.csv
repro_979/open/data/test.csv
repro_979/open/data/sample_submission.csv
~~~

No **trackman_history.csv** bridge is created. The v93 extracted directory is
linked at:

~~~text
repro_979/cache/v93_extract_verify
~~~

Tracked Task-3 baseline evidence authoritatively enumerates the 51 identities
under **model/**. The bootstrap validates that exact identity set instead of
counting arbitrary files:

- 43 explicitly named model-weight files
- four prep files: **mlp_prep.pkl**, **ftt_prep.pkl**, **armb_prep.pkl**, and
  **catboost_prep.pkl**
- four metadata files: **mlp_meta.json**, **ftt_meta.json**,
  **armb_meta.json**, and **train_meta.json**

Prep and metadata are validated separately from model weights. Extra or renamed
files cannot satisfy the contract. Structural identity validation does not
prove byte parity; supplying the authorized archive proves its recorded
SHA-256, while full real-asset parity remains a separate validation concern.

## Safety and idempotency

- Correct links and an exact existing environment are accepted on rerun.
- Missing links are created only in **apply** mode.
- Wrong or broken links, ordinary files at destinations, symlinked destination
  parents, and mismatched environments cause a safe failure.
- There is no force mode, destructive repair, automatic extraction, or asset
  deletion. A partially completed safe state can be inspected and rerun.
- GPU inventory, memory, utilization, and compute processes are reported.
  Missing or failed nvidia-smi inspection, no usable reported device, or
  unavailable Torch CUDA makes the result **NOT READY**. The tool never sets
  CUDA_VISIBLE_DEVICES, chooses a device, or allocates a CUDA tensor during
  normal checks.
- CSV validation reads headers only. It never computes train/test statistics or
  compares test rows.

Before later GPU work, inspect the reported occupancy and explicitly choose an
available device. Do not assume GPU 0 is free.
