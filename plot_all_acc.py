import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_algo_file(path: str):
    """Load one per-algorithm acc file saved by np.save(dict, allow_pickle=True)."""
    d = np.load(path, allow_pickle=True).item()
    # Robust key handling
    algo = d.get("algo", None)
    if algo is None:
        # fallback: parse from filename suffix "_acc_<Algo>.npy"
        base = os.path.basename(path)
        algo = base.split("_acc_")[-1].replace(".npy", "")
    rounds = d.get("rounds", None)
    acc = d.get("acc", None)
    if rounds is None or acc is None:
        raise ValueError(f"Missing 'rounds' or 'acc' in {path}. Keys={list(d.keys())}")
    rounds = np.asarray(rounds).astype(int)
    acc = np.asarray(acc).astype(float)
    return algo, rounds, acc, d


def find_experiment_files(save_dir: str, exp_prefix: str):
    """
    exp_prefix: the prefix before '_acc_' in your filename, e.g.
      'data_non-iid_set_cifar10_noniid_0.5_M_100_S_10_lr_0.1_n_ep_40_mtcl_1'
    """
    pattern = os.path.join(save_dir, f"{exp_prefix}_acc_*.npy")
    files = sorted(glob.glob(pattern))
    return files


def plot_all_algos(save_dir: str, exp_prefix: str, out_png: str = None):
    files = find_experiment_files(save_dir, exp_prefix)
    if not files:
        raise FileNotFoundError(
            f"No files found with pattern: {os.path.join(save_dir, exp_prefix + '_acc_*.npy')}"
        )
    curves = {}  # algo -> (rounds, acc)
    metas = {}

    for fp in files:
        algo, rounds, acc, meta = load_algo_file(fp)
        curves[algo] = (rounds, acc)
        metas[algo] = meta

    # Print max acc per algo
    print(f"\nFound {len(curves)} algorithms for exp_prefix:\n  {exp_prefix}\n")
    max_rows = []
    for algo, (rounds, acc) in sorted(curves.items(), key=lambda x: x[0].lower()):
        idx = int(np.nanargmax(acc))
        max_acc = float(acc[idx])
        max_round = int(rounds[idx])
        max_rows.append((algo, max_acc, max_round))
        print(f"{algo:10s} | max acc = {max_acc:.4f} at round {max_round}")

    # Plot
    plt.figure(figsize=(10, 6))
    for algo, (rounds, acc) in sorted(curves.items(), key=lambda x: x[0].lower()):
        plt.plot(rounds, acc, label=algo)

    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy")
    plt.title(exp_prefix)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if out_png is None:
        out_png = os.path.join(save_dir, f"{exp_prefix}_acc_all_algos.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure to: {out_png}\n")

if __name__ == "__main__":
    # ====== USER SETTINGS ======
    SAVE_DIR = "./save"

    # Put your experiment prefix here (everything before "_acc_<Algo>.npy")
    EXP_PREFIX = "data_pathological_set_cifar10_noniid_0.5_c_3_M_100_S_10_lr_0.1_n_ep_40_mtcl_1"
    plot_all_algos(SAVE_DIR, EXP_PREFIX)
