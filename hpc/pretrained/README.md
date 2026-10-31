# Pretrained weights

`avianz_hpc train` starts RegNetY-008 from BirdCLEF weights (`model_fold0.pth`,
68 MB) when it can find them, in this order:

1. `--pretrained PATH`
2. `$AVIANZ_HPC_PRETRAINED` (set in `hpc/config/project.env`)
3. `hpc/pretrained/model_fold0.pth` (this folder; `*.pth` is git-ignored)

The copy in `model-testing/model-testing/BirdClefModels/model_fold0.pth` is a
**Git LFS pointer** (133 bytes), not the weights. Fetch the real file from the
model-testing repository (`git lfs pull`) or ask the team, then copy it to
`/nfs/scratch/$USER/pretrained/` on Rāpoi. Pointer stubs are detected and
rejected rather than silently loaded; without real weights the backbone trains
from random initialisation, which needs far more data.
