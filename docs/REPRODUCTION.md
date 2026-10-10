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

Large raw latency arrays and complete execution outputs should be distributed as the `v1.0.0` GitHub release asset. See `release_assets/README.md`.

## 7. Original experiment provenance

The exact script names, line counts, and SHA-256 hashes recorded by the measured environment are stored in `manifests/original_experiment_scripts.json`. The compact repository keeps the frozen evidence and reference architecture implementation under version control; large execution outputs belong in the versioned release asset.
