# RePressDB Artifact

This repository accompanies the manuscript:

**RePressDB: A Dependable Self-Adaptive Storage Architecture for Nonstationary Time-Series Workloads**

Authors: Zhichao Zhang, Xiaolu Peng, and Daoqi Sun.

## Contents

- `src/repressdb/`: reference implementation of the storage architecture described in the paper.
- `results/frozen_summaries/`: frozen CSV/JSON evidence used to reconstruct the reported paper-level statistics.
- `paper/`: manuscript source, supplementary material, final figures, and figure/table generation scripts.
- `environment/`: software and hardware metadata for the measured campaigns.
- `scripts/`: verification, hardware capture, and smoke-test utilities.
- `docs/`: reproduction notes, dataset provenance, artifact map, and release instructions.
- `manifests/`: checksums and source-script provenance.

The public datasets used by the study are not redistributed here. Dataset acquisition and placement are described in `docs/DATASETS.md`.

## Quick verification

Create an environment and install the pinned dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install -r environment/requirements.txt
```

Verify the repository and frozen evidence:

```bash
python scripts/verify_artifact.py
```

Run the architecture smoke test:

```bash
python scripts/smoke_test.py
```

Regenerate paper-level numerical macros and figures from the frozen summaries:

```bash
python scripts/regenerate_paper_assets.py
```

Detailed instructions are in `docs/REPRODUCTION.md`.

## Experimental environment

The paired file-backed campaign was run on Windows 11 using an Intel Core i7-14700HX (20 physical cores, 28 logical processors), 16 GB RAM, and a YMTC YMSS2ED08D25MC NVMe SSD. SQLite used WAL mode with `synchronous=FULL`. Exact metadata is preserved in `environment/hardware_exact.json` and `environment/environment_report.json`.

## Releases

Large raw result archives should be attached to the GitHub `v1.0.0` release rather than committed to the Git history. `release_assets/README.md` gives the naming and checksum procedure.

## License

Source code is released under the MIT License. Frozen result tables and documentation are released under CC BY 4.0; see `LICENSE_RESULTS`.
