# Dataset access

The artifact does not include raw PEMS04, PEMS08, or Building Data Genome Project 2 data.

To reproduce the experiments:

1. Obtain each dataset from the source cited in the manuscript.
2. Keep the original files unchanged.
3. Use the frozen selection and trace manifests included with the result package.
4. Record local paths in a private run configuration. Do not commit local absolute paths or raw datasets.

PEMS04 is the development domain. PEMS08 and the BDG2 holdouts must not be used for parameter retuning.
