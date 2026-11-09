# FedVSSAM (NeurIPS 2026)

Official implementation of **FedVSSAM: Mitigating Flatness Incompatibility in
Sharpness-Aware Federated Learning**, accepted at NeurIPS 2026.

**Bingnan Xiao, Yuan Gao, Bingcong Li, Wei Ni, Xin Wang, and Tony Quek**

[Paper](https://openreview.net/forum?id=OB02U98b9E)

FedVSSAM uses a variance-suppressed global direction to guide local perturbation
search, local training, and server updates. This repository contains the
CIFAR-10 and CIFAR-100 experiments.

## Installation

Use a virtual environment and install the PyTorch/torchvision build for your
CPU or CUDA version following the [PyTorch instructions](https://pytorch.org/get-started/previous-versions/).
Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

For the tested CPU environment (Python 3.13), use:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -c constraints-tested.txt
```

The submitted code recommends Python 3.10. On Python 3.10, install from
`requirements.txt` without `constraints-tested.txt`, which contains newer
package versions for the Python 3.13 environment.

## Data and models

Run commands from the repository root. CIFAR data are downloaded automatically
to `./data/cifar`; the current CIFAR loaders use this path directly rather than
the `--data_dir` argument.

With `--model cnn`, the model is selected by dataset:

| Dataset | Model | Optional initial weights |
|---|---|---|
| CIFAR-10 | ResNet-18 with GroupNorm | `ini_model_weights/cifar100_new.pth` |
| CIFAR-100 | WRN-28-4 | `ini_model_weights/cifar100_wrn.pth` |

The script loads the corresponding weight file when it exists and otherwise
uses random initialization. Initial weights are not included in the repository.

## Training

Run FedVSSAM on CIFAR-10 with 100 clients, 10 clients per round, and Dirichlet
alpha 0.1:

```bash
python main.py --dataset cifar10 --model cnn --algos FedVSSAM --num_users 100 --user_subset 10 --epochs 800 --n_epochs 10 --local_bs 128 --lr 0.1 --gamma_lr 1 --alpha 0.1 --eval_interval 1 --target_acc 75 --seed 1 --gpu 0 --save_dir ./save_cifar10_dir0p1_seed1
```

`--epochs` sets the communication rounds; `--n_epochs` sets the number of local
mini-batch updates. On Windows or Linux, use `--gpu -1` to run on CPU.

Choose methods with `--algos`, for example `--algos FedAvg,FedSAM,FedVSSAM`.
The default runs FedVSSAM and all 12 baselines: FedAvg, FedProx, FedAdam,
SCAFFOLD, FedACG, FedCM, FedDyn, FedSAM, MoFedSAM, FedGAMMA, FedSMOO, and
FedLESAM. `FedMOSAM` is an alias for MoFedSAM.

[commands.txt](commands.txt) contains more CIFAR-10/100 settings, including
100/200 clients, Dirichlet alpha 0.5/0.1, and pathological label splits.
It also includes the K=40 example from the submitted README.

A few parameter conventions:

- FedVSSAM uses `fedvssam_gamma_l=0.1` and `fedvssam_gamma_g=0.6` by default.
  Set the local coefficient explicitly to use another value.
- The FedVSSAM server step is fixed at 4 in `main.py`.
- FedProx, FedAdam, and FedACG have separate client learning-rate arguments.
- Evaluation defaults to every 20 rounds. The example above evaluates every
  round to measure the first crossing of the target accuracy.

Use a separate `--save_dir` for each seed and configuration, since output
filenames do not include the seed. See [implementation notes](docs/REPRODUCIBILITY.md)
for the full defaults and sampling details.

## Results

Training saves per-method accuracy curves (`*_acc_<method>.npy`), a convergence
plot, and CSV/Markdown summaries of final accuracy, best accuracy, and rounds
to the target.

To generate a round-to-threshold table:

```bash
python compute_rd_table.py --save_dir ./save_cifar10_dir0p1_seed1 --out_csv ./save_cifar10_dir0p1_seed1/rd_table.csv --out_md ./save_cifar10_dir0p1_seed1/rd_table.md
```

The default thresholds are 75% for CIFAR-10 and 40% for CIFAR-100. `NR` means
that a method's curve is missing or the target was not reached. The summaries
use recorded evaluation rounds.

To plot the curves for the example above:

```bash
python -c "from plot_all_acc import plot_all_algos; plot_all_algos('./save_cifar10_dir0p1_seed1', 'data_non-iid_set_cifar10_noniid_0.1_c_2_M_100_S_10_lr_0.1_n_ep_10_mtcl_1')"
```

For another run, use the filename prefix before `_acc_<method>`.

## Citation

If you use this code or build on our work, please cite:

```bibtex
@inproceedings{xiao2026fedvssam,
  title     = {{FedVSSAM}: Mitigating Flatness Incompatibility in Sharpness-Aware Federated Learning},
  author    = {Bingnan Xiao and Yuan Gao and Bingcong Li and Wei Ni and Xin Wang and Tony Quek},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=OB02U98b9E}
}
```

## License

This code is released under the [MIT License](LICENSE_Anonymous_Authors).

Source provenance and the checks performed on this release are documented in
[SOURCE.md](docs/SOURCE.md) and [VALIDATION.md](docs/VALIDATION.md).
