# Implementation notes

This page documents the defaults and output conventions used by the CIFAR
implementation. See [SOURCE.md](SOURCE.md) for its provenance and
[VALIDATION.md](VALIDATION.md) for verification details.

## Default settings

| Argument / behavior | Default |
|---|---|
| Dataset / model | `cifar10` / `cnn` |
| Communication rounds | `epochs=100` |
| Total / selected clients | `num_users=100`, `user_subset=10` |
| Local updates / local batch | `n_epochs=5`, `local_bs=32` |
| Test batch | `bs=128` |
| Common LR / decay | `lr=0.1`, `gamma_lr=0.99995` |
| Momentum / weight decay | `momentum=0`, `weight_decay=0` |
| Dirichlet alpha | `alpha=0.5` |
| Pathological / IID switches | Both false; the default is Dirichlet |
| Classes per client option | `num_classes_per_user=2` |
| Seed / Monte Carlo count | `seed=1`, `monto_carlo_times=1` |
| Evaluation / summary target | `eval_interval=20`, `target_acc=90` |
| FedVSSAM local / global mixing | `fedvssam_gamma_l=0.1`, `fedvssam_gamma_g=0.6` |
| FedVSSAM perturbation | `fedvssam_rho=0.05` |
| FedVSSAM server step | `global_eta=4`, hardcoded in `main.py` |
| FedSAM perturbation | `sam_rho=0.1` |
| MoFedSAM perturbation / mixing | `rho_mosam=-1` selects 0.01 for CIFAR-100 and 0.1 for CIFAR-10; `gamma_mosam=0.1` |
| FedProx local LR / strength | `fedprox_lr=0.1`, `mu_prox=0.1` |
| FedAdam client LR | `10**(-1.5)` |
| FedAdam server parameters | `beta1=0.9`, `beta2=0.99`, `tau=0.001`, `eta=0.01` |
| FedACG local LR / lookahead / local term | `0.1` / `0.85` / `0.01` |
| FedSMOO / FedDyn / FedCM | `r=0.1`, `beta=10`; `mu=0.01`; `alpha_cm=0.1` |
| Initialization | Load the dataset's initial state dictionary if present; otherwise use random initialization |
| CIFAR data location | `./data/cifar` relative to the working directory; `data_dir` does not redirect these loaders |

Run `python main.py --help` for the full argument list. FedProx, FedAdam and
FedACG have separate client learning-rate options; setting `--lr` alone does
not change those rates.

## Sampling and local updates

The CIFAR Dirichlet samplers reset a category's index counter when its sample
pool is exhausted. Training indices can therefore be reused within or across
clients. This applies to the Dirichlet routines; other partition routines
have their own allocation rules.

`n_epochs` specifies the local mini-batch update count K, rather than complete
passes over a client's dataset. For FedSAM, a loader with L batches performs
K updates when K is divisible by L and can perform K+1 otherwise: its inner
loop checks `count > K` before incrementing. FedVSSAM uses `count >= K` and
stops at K. Some other methods have a 100-pass cap, while FedSimpSAM uses a
single pass. The 100-pass cap does not limit the usual K=10 setting for a
nonempty loader.

## Commands and outputs

`commands.txt` includes a K=40 example from the submitted README and 16
K=10 CIFAR configurations covering the paper's device counts and partition
settings. Each command sets its learning rate, mixing coefficients and
evaluation interval explicitly.

Use a separate `save_dir` for each seed and configuration: filenames do not
include the seed or every parameter. Record the command, environment and
initial weight file with the results. The supplied utilities summarize one
run at a time; across-seed statistics require a separate aggregation step.

Evaluation runs every 20 completed rounds and at the final round by default.
Set `--eval_interval 1` to measure the first threshold crossing at every
round. The CIFAR examples use targets of 75% and 40%; the parser's default
target is 90%.

`summary_*.csv` reports final and best accuracy separately.
`compute_rd_table.py` reads the stored round indices and uses `NR` for both
missing methods and unreached targets; it skips unreadable files.
`plot_all_acc.py` selects curves by their filename prefix and prints best
accuracy. Only recorded evaluation points contribute to these results.

## Environment

The submitted setup recommends Python 3.10. `requirements.txt` lists its eight
dependencies without version pins. For Python 3.10, install compatible
package versions without applying `constraints-tested.txt`.

`constraints-tested.txt` records the Python 3.13 CPU environment used for the
checks in [VALIDATION.md](VALIDATION.md). Its SciPy version requires Python
3.11 or newer.
