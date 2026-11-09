# Validation

We compared this CIFAR release with the [submitted code](SOURCE.md) on
2026-11-10.

| Check | Result |
|---|---|
| Local updates, aggregation, evaluation and optimizer files | Byte for byte identical to the submitted attachment. |
| Six CIFAR sampling functions | Same source text, allowing only newline encoding normalization. |
| Retained model definitions | Same source/AST as the submitted classes. |
| CIFAR training entry point | Complete dataset-specialized AST matches the original after removing only excluded dataset imports, branches and initialization-path keys. |
| CLI options | All 61 retained defaults, types and actions match; only dataset choices are restricted to CIFAR-10/100. |
| Initialization and defaults | Original optional weight lookup, fixed local mixing default 0.1, hardcoded global step 4, RNG handling, algorithm order, LR application and evaluation defaults are retained. |
| Original utilities and dependencies | Plotting, threshold-table utility and requirements file are byte for byte identical. |
| Paired execution | Five offline test settings, 13 methods and two rounds per method on each version: partitions, client schedules, optimizer step counts, model hashes, losses, curves, summaries and execution order match exactly. |
| Model definitions | ResNet-18 and WRN-28-4 initialization and forward-output hashes match; outputs have 10/100 classes respectively. |

The five execution settings cover both CIFAR Dirichlet paths, both
pathological paths, and CIFAR-10 with an available initial weight file. We
used the same synthetic inputs and small network in both versions to compare
training control flow. ResNet-18 and WRN-28-4 were checked separately with
their actual architectures.

The checks ran on Windows with Python 3.13.5, PyTorch 2.9.1+cpu,
torchvision 0.24.1+cpu and the versions in `constraints-tested.txt`.
Dependency consistency and Python syntax checks also passed.

These checks establish implementation equivalence. The full 800-round
experiments were not rerun.
