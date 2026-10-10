# Reproduction guide

## 1. Environment

The measured campaigns were executed with Python 3.10.20 on Windows 11. Install the pinned Python packages with:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r environment/requirements.txt
```

Linux and macOS can be used for evidence verification and figure regeneration. The original file-backed performance numbers should be interpreted with the recorded Windows/NTFS environment in `environment/`.

## 2. Frozen evidence verification

```bash
python scripts/verify_artifact.py
```

The command checks the repository manifest, required frozen summaries, paper-level headline values, and hardware metadata.

`src/repressdb/` is the organized reference implementation. The eight
entry scripts used in the original experiments are preserved byte for byte
in `experiments/original/`; their recorded SHA-256 values are in
`manifests/original_experiment_scripts.json`. The paper's frozen summaries
are in `results/frozen_summaries/`.

## 3. Reference implementation smoke test

```bash
python scripts/smoke_test.py
```

The smoke test creates a temporary single-active store, writes a synthetic float32 chunk, serves queries through the active view, performs a representation migration, closes the store, reopens it, runs recovery, and verifies query equality.

## 4. Paper assets

```bash
python scripts/regenerate_paper_assets.py
```

This regenerates the numerical macros, generated tables, and figures from `results/frozen_summaries/`. The manuscript itself is not required for evidence verification.

## 5. Public datasets

The study uses PEMS04, PEMS08, and Building Data Genome Project 2. They are not redistributed. Acquisition notes are in `docs/DATASETS.md`.

## 6. Raw result release

The complete raw result archives, including raw latency arrays, are packaged
for distribution as the `v1.0.0` GitHub Release asset. GitHub provides the source
archive for the tag separately. Public source datasets are not redistributed.
See `release_assets/README.md` for the four result archives and checksums.

## 7. Original experiment provenance

The exact script names, line counts, and SHA-256 hashes recorded by the measured environment are stored in `manifests/original_experiment_scripts.json`. The corresponding entry-script snapshots are in `experiments/original/`. Large execution outputs belong in the versioned release asset.
