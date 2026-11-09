import argparse
import csv
import glob
import os
import pickle
import re
from collections import defaultdict

import numpy as np
from numpy.lib import format as np_format

ALGO_ORDER = [
    "FedAvg",
    "FedProx",
    "FedAdam",
    "Scaffold",
    "FedACG",
    "FedCM",
    "FedDyn",
    "FedSAM",
    "FedMOSAM",
    "FedGAMMA",
    "FedSMOO",
    "FedLESAM",
    "FedVSSAM",
]

ALGO_DISPLAY_MAP = {
    "FedMOSAM": "MoFedSAM",
}

SETTING_RE = re.compile(
    r"^data_(?P<partition>non-iid|pathological)_set_"
    r"(?P<dataset>cifar10|cifar100)_noniid_(?P<alpha>[0-9.]+)_c_(?P<c>\d+)_M_(?P<M>\d+)"
    r"_S_(?P<S>\d+)_lr_(?P<lr>[0-9.]+)_n_ep_(?P<n_ep>\d+)_mtcl_(?P<mtcl>\d+)$"
)


class _CompatUnpickler(pickle.Unpickler):
    """Compatibility unpickler for npy files pickled with numpy 2.x path names."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_npy_dict(path):
    """Load one saved accuracy npy file (np.save(dict, allow_pickle=True))."""
    try:
        obj = np.load(path, allow_pickle=True)
        data = obj.item() if isinstance(obj, np.ndarray) and obj.shape == () else obj
        if not isinstance(data, dict):
            raise ValueError(f"Not a dict payload: {path}")
        return data
    except ModuleNotFoundError as e:
        # Fallback for files pickled with numpy 2.x internal path "numpy._core"
        if "numpy._core" not in str(e):
            raise
        with open(path, "rb") as f:
            version = np_format.read_magic(f)
            if version == (1, 0):
                np_format.read_array_header_1_0(f)
            elif version in ((2, 0), (3, 0)):
                np_format.read_array_header_2_0(f)
            else:
                raise ValueError(f"Unsupported npy version {version} in {path}")
            arr = _CompatUnpickler(f).load()
        data = arr.item() if isinstance(arr, np.ndarray) and arr.shape == () else arr
        if not isinstance(data, dict):
            raise ValueError(f"Not a dict payload after fallback load: {path}")
        return data


def first_reach_round(rounds, acc, threshold):
    rounds = np.asarray(rounds).astype(int)
    acc = np.asarray(acc).astype(float)
    hit = np.where(acc >= threshold)[0]
    if hit.size == 0:
        return None
    return int(rounds[hit[0]])


def to_markdown(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(row_vals):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row_vals)) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    out = [fmt_row(headers), sep]
    for r in rows:
        out.append(fmt_row(r))
    return "\n".join(out)


def parse_setting(setting):
    m = SETTING_RE.match(setting)
    if not m:
        return None
    g = m.groupdict()
    return {
        "partition": g["partition"],
        "dataset": g["dataset"],
        "alpha": float(g["alpha"]),
        "c": int(g["c"]),
        "M": int(g["M"]),
        "S": int(g["S"]),
        "lr": float(g["lr"]),
        "n_ep": int(g["n_ep"]),
        "mtcl": int(g["mtcl"]),
    }


def _near(v, target, eps=1e-9):
    return abs(v - target) <= eps


def setting_order_key(setting):
    meta = parse_setting(setting)
    if meta is None:
        return (99, 99, 99.0, 99, setting.lower())

    m_rank = {100: 0, 200: 1}.get(meta["M"], 2)
    scenario_rank = 50
    if meta["dataset"] == "cifar10" and meta["partition"] == "non-iid" and meta["c"] == 2:
        if _near(meta["alpha"], 0.5):
            scenario_rank = 0
        elif _near(meta["alpha"], 0.1):
            scenario_rank = 1
    elif meta["dataset"] == "cifar10" and meta["partition"] == "pathological":
        if meta["c"] == 6:
            scenario_rank = 2
        elif meta["c"] == 3:
            scenario_rank = 3
    elif meta["dataset"] == "cifar100" and meta["partition"] == "non-iid" and meta["c"] == 2:
        if _near(meta["alpha"], 0.5):
            scenario_rank = 4
        elif _near(meta["alpha"], 0.1):
            scenario_rank = 5
    elif meta["dataset"] == "cifar100" and meta["partition"] == "pathological":
        if meta["c"] == 20:
            scenario_rank = 6
        elif meta["c"] == 10:
            scenario_rank = 7

    # Prefer canonical run config first when duplicate scenarios exist.
    lr_delta = abs(meta["lr"] - 0.1)
    nep_delta = abs(meta["n_ep"] - 40)
    return (m_rank, scenario_rank, lr_delta, nep_delta, setting.lower())


def main():
    parser = argparse.ArgumentParser(description="Compute RD table from saved FL accuracy curves.")
    parser.add_argument("--save_dir", type=str, default="./save", help="Directory that stores *_acc_*.npy files.")
    parser.add_argument("--thr_cifar10", type=float, default=75.0, help="Target accuracy for CIFAR-10.")
    parser.add_argument("--thr_cifar100", type=float, default=40.0, help="Target accuracy for CIFAR-100.")
    parser.add_argument("--out_csv", type=str, default="./save/rd_table.csv", help="Output CSV path.")
    parser.add_argument("--out_md", type=str, default="./save/rd_table.md", help="Output Markdown table path.")
    args = parser.parse_args()

    thresholds = {"cifar10": float(args.thr_cifar10), "cifar100": float(args.thr_cifar100)}

    files = sorted(glob.glob(os.path.join(args.save_dir, "*_acc_*.npy")))
    if not files:
        raise FileNotFoundError(f"No *_acc_*.npy files found under: {args.save_dir}")

    # setting -> dataset
    setting_dataset = {}
    # setting -> algo -> rd
    rd_map = defaultdict(dict)
    algos = set()

    used = 0
    for fp in files:
        base = os.path.basename(fp)
        if base.endswith("_acc_curves.npy"):
            continue
        setting = base.split("_acc_")[0]
        if "_c_" not in setting:
            # Ignore legacy curves without explicit c field.
            continue
        try:
            d = _load_npy_dict(fp)
        except Exception:
            # Ignore incompatible/invalid files.
            continue

        if not all(k in d for k in ("dataset", "algo", "rounds", "acc")):
            continue

        dataset = str(d["dataset"]).lower()
        if dataset not in thresholds:
            continue
        threshold = thresholds[dataset]
        algo = str(d["algo"])
        rounds = d["rounds"]
        acc = d["acc"]

        rd = first_reach_round(rounds, acc, threshold)

        setting_dataset[setting] = dataset
        rd_map[setting][algo] = rd
        algos.add(algo)
        used += 1

    if used == 0:
        raise RuntimeError("No valid accuracy files were parsed.")

    ordered_algos = list(ALGO_ORDER)
    ordered_algos.extend(sorted((algos - set(ALGO_ORDER)), key=str.lower))
    settings = sorted(setting_dataset.keys(), key=setting_order_key)

    headers = ["setting", "dataset", "threshold"] + [ALGO_DISPLAY_MAP.get(a, a) for a in ordered_algos]
    rows = []
    for setting in settings:
        dataset = setting_dataset[setting]
        threshold = thresholds[dataset]
        row = [setting, dataset, f"{threshold:g}"]
        for algo in ordered_algos:
            rd = rd_map[setting].get(algo, None)
            row.append("NR" if rd is None else str(rd))
        rows.append(row)

    # Print markdown table
    md_text = to_markdown(headers, rows)
    print(md_text)

    # Save markdown
    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md_text + "\n")

    # Save CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"\nSaved markdown table to: {args.out_md}")
    print(f"Saved csv table to: {args.out_csv}")
    print(f"Parsed files: {used}")

if __name__ == "__main__":
    main()
