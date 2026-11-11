# RePressDB

Risk-aware layout adaptation for immutable compressed time-series chunks.

Authors: Zhichao Zhang, Daoqi Sun, Xiaolu Peng.

## Contents

| Path | Contents |
| --- | --- |
| `paper/current/` | Current manuscript, figures, bibliography, and PDF |
| `experiments/current/method_v22/` | Final controller, durable executor, frozen protocol, and results |
| `experiments/current/source/` | Original experiment modules and calibration inputs |
| `experiments/current/review_analysis_v23/` | Decision, migration-episode, and stable-window analyses |
| `experiments/current/figure_revision_v26/` | Figure-generation script and plotted values |
| `src/repressdb/` | Reference storage implementation |

The current comparison contains 144 executions, six policies, 24 paired instances, and 23,040,000 checked events. Continuous-mixture and persistent-shift results are reported separately. Calibration and controller initialization are accounted for separately from service time.

## Environment

Measurements used Python 3.10.20, NumPy 2.2.6, Pandas 2.3.3, LZ4 4.4.5, Zstandard 0.25.0, and psutil 7.2.2 on Windows/NTFS. Install the recorded dependencies:

```bash
pip install -r experiments/current/requirements.txt
```

## Verification

Download `RePressDB-evidence-2026-11-11.zip` from the **artifact-2026-11-11** release and install the event arrays:

```bash
python scripts/install_measurements.py RePressDB-evidence-2026-11-11.zip
python experiments/current/method_v22/check_method.py
python experiments/current/method_v22/verify_v2.py
python experiments/current/review_analysis_v23/verify_review.py
```

The installer checks the archive checksum. Verification covers source hashes, calibration records, per-event component sums, migration records, payload budgets, and recovery outcomes. The release retains the complete measurements and development records. Git tracks the source, summaries, derived evaluation inputs, and analysis tables.

## Manuscript and figures

The files in `paper/current/` match the supplied source ZIP byte for byte. The PDF is compiled from those unchanged inputs. Compilation checks are in `paper/current/build_verification.json`.

```bash
cd paper/current
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Figures 2--4 can be regenerated with `experiments/current/figure_revision_v26/render_figures.py`. Plotting uses the dependencies in `experiments/current/requirements-figures.txt` and installed Times New Roman fonts. Figure 1 is supplied as a separate image.

Original experiment snapshots remain in their existing directories. Dataset information is in `docs/DATASETS.md`. Source code uses the MIT License; result tables and documentation use CC BY 4.0.
