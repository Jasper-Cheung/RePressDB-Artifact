# RePressDB JSS Revision Artifact

This repository supports the major revision of **RePressDB: A Dependable Self-Adaptive Storage Architecture for Nonstationary Time-Series Workloads**.

It contains the experiment code, frozen summary results, environment records, trace and selection manifests, and the scripts used to generate the revised tables and numerical TeX macros. Raw public datasets are not redistributed.

## Artifact levels

**Audit the paper numbers.** The files under `results/frozen_summaries/` and `paper_assets/` are sufficient to regenerate the main revision tables and check the reported ratios.

**Inspect paired end-to-end evidence.** `results/paired_e2e_core/` contains all worker summaries and the result manifest. Raw latency arrays are distributed in the versioned release asset because they exceed the normal Git repository file limit.

**Rerun experiments.** Obtain PEMS04, PEMS08, and BDG2 from their cited public sources, preserve the frozen data-selection manifests, install the pinned environment, and run the one-entry scripts in `src/`.

## Main entry points

- `src/RUN_THIS_ONLY_REPRESSDB_JSS_MAJOR_REVISION_EXPERIMENTS_V1_5_1.py`
- `src/RUN_THIS_ONLY_REPRESSDB_J1R_LOW_MEMORY_PAIRED_V1_7_0.py`
- `paper_assets/make_revision_assets.py`
- `scripts/verify_artifact.py`

## Environment

The formal runs used Python 3.10.20, NumPy 2.2.6, pandas 2.3.3, LZ4 4.4.5, and zstandard 0.25.0. The captured operating-system and package reports are included. Run `environment/capture_hardware.bat` on the original experiment host to record the exact CPU marketing name and the project-drive model before final submission.

## Large files

Do not commit the full raw-result ZIP to the Git history. Attach `RePressDB_JSS_Artifact_Full_Release_V1.0.0.zip` to a GitHub Release and archive the same file in Zenodo. Checksums are listed in `release_assets/RELEASE_ASSET_SHA256.txt`.

## Licenses

Code is released under the MIT License. RePressDB-generated summaries, manifests, and result records are released under CC BY 4.0. Source datasets retain their original terms.
