# Dataset provenance

The artifact does not redistribute third-party datasets.

## PEMS04 and PEMS08

Traffic measurements are used for representation calibration, development-only policy configuration, and held-out evaluation. Use the versions cited in the manuscript. Preserve the original sensor/time/channel ordering before applying the frozen selection procedure described in the supplementary material.

## Building Data Genome Project 2

Building-meter series are used for cross-domain holdout, continuous replay, the physical corpus, and logical-scale experiments. Use the public Building Data Genome Project 2 release cited in the manuscript.

## Integrity practice

Keep downloaded datasets outside the Git repository. Record the local source path and source checksum before generating derived chunks. The published artifact should contain only derived manifests, frozen summaries, and scripts, not the original third-party data.
