# Code provenance

This repository contains the CIFAR-10 and CIFAR-100 code from the
[supplementary material](https://openreview.net/attachment?id=OB02U98b9E&name=supplementary_material)
for our NeurIPS 2026 paper.

The source attachment is 45,402 bytes, with SHA-256:

```text
27039c59e9ff91ccb7c98954b5eee4e69d13e0d4706cbc92f8daa7f2617af2d9
```

Its internal source directory is
`FedVSSAM_NeurIPS2026_code/FedVSSAM_NeurIPS2026_code/`.

For this release, we removed code and options for other datasets and added
installation instructions and command examples. The CIFAR samplers, local
updates, models, optimizers, initialization and default parameters follow the
submitted implementation. The dependency list and MIT license file are also
retained. `constraints-tested.txt` records the versions used for our CPU checks.

The package includes source code and documentation; model weights and experiment
logs are not bundled. See [implementation notes](REPRODUCIBILITY.md)
for the data layout and defaults, and [validation](VALIDATION.md) for the
comparison with the submitted code.
