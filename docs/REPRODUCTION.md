# Reproduction guide

## Check the frozen artifact

```bash
python scripts/verify_artifact.py
```

This verifies repository hashes and checks that the frozen v1.7.0 assessment reports all ten acceptance conditions as true.

## Rebuild paper assets

Run `paper_assets/make_revision_assets.py` from the manuscript directory after placing the frozen summary files in the expected `revision_data/` path.

## Rerun the major-revision experiments

Use the same Python environment and obtain the public datasets separately. Start with:

```text
src/RUN_THIS_ONLY_REPRESSDB_JSS_MAJOR_REVISION_EXPERIMENTS_V1_5_1.py
```

For the paired low-memory end-to-end campaign, place the scripts in one project directory and run only:

```text
src/RUN_THIS_ONLY_REPRESSDB_J1R_LOW_MEMORY_PAIRED_V1_7_0.py
```

The heavy worker is launched automatically. Completed cells are resumable.
