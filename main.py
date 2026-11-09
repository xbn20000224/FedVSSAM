"""Run federated learning experiments for FedVSSAM and baseline methods."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pylab import *

import copy
import numpy as np
import os
import gc
import csv
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torchvision

from torch import nn

import random
from random import randint
from utils.sampling import cifar10_iid, cifar10_noniid, cifar100_iid, cifar100_noniid, cifar10_pathological, cifar100_pathological
from utils.options import args_parser
from torch.utils.data import DataLoader
from models.Update import DatasetSplit, LocalUpdate, fedsam_LocalUpdate, fedvssam_LocalUpdate, fedmosam_LocalUpdate, \
    fedsimpsam_LocalUpdate, scaffold_LocalUpdate, fedgamma_LocalUpdate, fedsmoo_LocalUpdate, feddyn_LocalUpdate, \
    fedcm_LocalUpdate, fedlesam_LocalUpdate, fedprox_LocalUpdate, fedacg_LocalUpdate
from models.Nets import MLP, CNNCifar10, ResNet18_CIFAR100
from models.Nets_new import WRN28_4_CIFAR100
from models.Fed import FedAvg
from models.test import test_img


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _state_dict_to_cpu(state_dict):
    """Detach and move model weights to CPU before caching in python lists."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def _release_cached_memory():
    """Force Python/PyTorch to release unreferenced memory between strategies."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_initial_weights_if_available(network, dataset):
    weight_paths = {
        'cifar10': './ini_model_weights/cifar100_new.pth',
        'cifar100': './ini_model_weights/cifar100_wrn.pth',
    }
    path = weight_paths.get(dataset)
    if path and os.path.exists(path):
        loaded_weights = torch.load(path, map_location='cpu')
        network.load_state_dict(loaded_weights)
        print(f'Loaded initial weights from {path}')
    elif path:
        print(f'Initial weights not found at {path}; using random initialization.')

ALGO_TO_STRATEGY = {
    'fedavg': 1,
    'fedsam': 2,
    'fedvssam': 3,
    'mofedsam': 4,
    'fedmosam': 4,
    'fedsimpsam': 5,
    'fedlvssam': 5,
    'fedl-vssam': 5,
    'scaffold': 6,
    'fedgamma': 7,
    'fedsmoo': 8,
    'feddyn': 9,
    'fedcm': 10,
    'fedlesam': 11,
    'fedprox': 12,
    'fedadam': 13,
    'fedacg': 14,
}


def parse_strategy_list(algo_string):
    strategies = []
    for raw_name in algo_string.split(','):
        key = raw_name.strip().lower()
        if not key:
            continue
        if key not in ALGO_TO_STRATEGY:
            raise ValueError(f'Unknown algorithm in --algos: {raw_name}')
        strategy = ALGO_TO_STRATEGY[key]
        if strategy not in strategies:
            strategies.append(strategy)
    return strategies


def summarize_history(history, target_acc):
    if not history:
        return None, None, None
    final_acc = float(history[-1][1])
    best_acc = float(max(acc for _, acc in history))
    rd_target = next((int(round_idx) for round_idx, acc in history if acc >= target_acc), None)
    return final_acc, best_acc, rd_target


def write_summary(rows, args, datatype):
    os.makedirs(args.save_dir, exist_ok=True)
    base = 'summary_data_{}_set_{}_noniid_{}_c_{}_M_{}_S_{}_lr_{}_n_ep_{}'.format(
        datatype, args.dataset, args.alpha, args.num_classes_per_user, args.num_users, args.user_subset, args.lr,
        args.n_epochs
    )
    csv_path = os.path.join(args.save_dir, base + '.csv')
    md_path = os.path.join(args.save_dir, base + '.md')
    fieldnames = [
        'dataset', 'partition', 'alpha', 'classes_per_client', 'algorithm',
        'final_acc', 'best_acc', 'rd_at_target', 'target_acc', 'eval_points'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('| Algorithm | Final Acc. | Best Acc. | Rd@{:.1f}% | Eval Points |\n'.format(args.target_acc))
        f.write('|---|---:|---:|---:|---:|\n')
        for row in rows:
            rd = row['rd_at_target'] if row['rd_at_target'] is not None else '--'
            f.write('| {} | {:.2f} | {:.2f} | {} | {} |\n'.format(
                row['algorithm'], row['final_acc'], row['best_acc'], rd, row['eval_points']
            ))
    return csv_path, md_path


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.dataset = args.dataset.lower()
    args.eval_interval = max(1, int(args.eval_interval))
    if torch.cuda.is_available() and args.gpu != -1:
        args.device = torch.device(f'cuda:{args.gpu}')
    elif torch.backends.mps.is_available():
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")
    setup_seed(args.seed)

    # load dataset and split users

    if args.dataset == 'cifar10':
        trans_cifar_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.RandomApply([transforms.RandomCrop(32, padding=4)], p=0.5)])
        trans_cifar_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar10_iid(dataset_train, args.num_users)
        elif args.pathological:
            dict_users = cifar10_pathological(args, dataset_train, args.num_users)
        else:
            dict_users = cifar10_noniid(args, dataset_train, args.num_users)

    elif args.dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])
        dataset_train = torchvision.datasets.CIFAR100(
            root='./data/cifar',
            train=True,
            download=True,
            transform=transform
        )
        dataset_test = torchvision.datasets.CIFAR100(
            root='./data/cifar',
            train=False,
            download=True,
            transform=transform
        )
        if args.iid:
            dict_users = cifar100_iid(dataset_train, args.num_users)
        elif args.pathological:
            dict_users = cifar100_pathological(args, dataset_train, args.num_users)
        else:
            dict_users = cifar100_noniid(args, dataset_train, args.num_users)



    else:
        exit('Error: unrecognized dataset')
    if args.prepare_only:
        print(f'Dataset prepared: train={len(dataset_train)}, test={len(dataset_test)}')
        exit(0)
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar100':
        net_glob = WRN28_4_CIFAR100(args=args, num_classes=100).to(args.device)

    elif args.model == 'cnn' and args.dataset == 'cifar10':
        net_glob = ResNet18_CIFAR100(args=args, num_classes=10).to(args.device)


    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    if args.iid:
        datatype = 'iid'
    elif args.pathological:
        datatype = 'pathological'
    else:
        datatype = 'non-iid'

    # Strategy1
    fedavg_iter = [int(i) for i in range(args.epochs)]
    fedavg_acc = [np.nan for i in range(args.epochs)]

    # Strategy2
    fedsam_iter = [int(i) for i in range(args.epochs)]
    fedsam_acc = [np.nan for i in range(args.epochs)]

    # Strategy3
    fedvssam_iter = [int(i) for i in range(args.epochs)]
    fedvssam_acc = [np.nan for i in range(args.epochs)]

    # Strategy4
    fedmosam_iter = [int(i) for i in range(args.epochs)]
    fedmosam_acc = [np.nan for i in range(args.epochs)]

    # Strategy5
    fedsimpsam_iter = [int(i) for i in range(args.epochs)]
    fedsimpsam_acc = [np.nan for i in range(args.epochs)]

    # Strategy6
    scaffold_iter = [int(i) for i in range(args.epochs)]
    scaffold_acc = [np.nan for i in range(args.epochs)]

    # Strategy7
    fedgamma_iter = [int(i) for i in range(args.epochs)]
    fedgamma_acc = [np.nan for i in range(args.epochs)]

    # Strategy8
    fedsmoo_iter = [int(i) for i in range(args.epochs)]
    fedsmoo_acc = [np.nan for i in range(args.epochs)]

    # Strategy9
    feddyn_iter = [int(i) for i in range(args.epochs)]
    feddyn_acc = [np.nan for i in range(args.epochs)]

    # Strategy10
    fedcm_iter = [int(i) for i in range(args.epochs)]
    fedcm_acc = [np.nan for i in range(args.epochs)]
    # Strategy11
    fedlesam_iter = [int(i) for i in range(args.epochs)]
    fedlesam_acc = [np.nan for i in range(args.epochs)]

    # Strategy12
    fedprox_iter = [int(i) for i in range(args.epochs)]
    fedprox_acc = [np.nan for i in range(args.epochs)]

    # Strategy13
    fedadam_iter = [int(i) for i in range(args.epochs)]
    fedadam_acc = [np.nan for i in range(args.epochs)]

    # Strategy14
    fedacg_iter = [int(i) for i in range(args.epochs)]
    fedacg_acc = [np.nan for i in range(args.epochs)]

    strategy_curves = {
        1: fedavg_acc,
        2: fedsam_acc,
        3: fedvssam_acc,
        4: fedmosam_acc,
        5: fedsimpsam_acc,
        6: scaffold_acc,
        7: fedgamma_acc,
        8: fedsmoo_acc,
        9: feddyn_acc,
        10: fedcm_acc,
        11: fedlesam_acc,
        12: fedprox_acc,
        13: fedadam_acc,
        14: fedacg_acc,
    }

    def record_accuracy(strategy_id, round_idx, acc_value, history):
        curve = strategy_curves[strategy_id]
        arr_idx = max(0, min(args.epochs - 1, int(round_idx) - 1))
        if np.isnan(curve[arr_idx]):
            curve[arr_idx] = 0.0
        curve[arr_idx] += float(acc_value)
        history.append((int(round_idx), float(acc_value)))
    algo_lis = ['FedAvg', 'FedSAM', 'FedVSSAM', 'MoFedSAM', 'FedSimpSAM', 'SCAFFOLD', 'FedGAMMA', 'FedSMOO', 'FedDyn',
                'FedCM', 'FedLESAM', 'FedProx', 'FedAdam', 'FedACG']
    strategy_lis = parse_strategy_list(args.algos)
    print('Algorithms:', [algo_lis[i - 1] for i in strategy_lis])
    summary_rows = []
    for monto_carlo_time in range(args.monto_carlo_times):

        print('training times:', monto_carlo_time)

        # Generate local update times of devices
        update_lis_iter = []
        for i in range(args.epochs):

            times = [args.n_epochs for i in range(args.num_users)]
            if not args.all_clients:
                index = random.sample(range(0, args.num_users), args.user_subset)  # Uniformly at random Device Sampling
                update_time_lis = [[index[i], times[i]] for i in range(args.user_subset)]
                update_lis_iter.append(update_time_lis)
            else:
                index = random.sample(range(0, args.num_users), args.num_users)  # Uniformly at random Device Sampling
                update_time_lis = [[index[i], times[i]] for i in range(args.num_users)]
                update_lis_iter.append(update_time_lis)

        update_lis_iter_backup = update_lis_iter

        # [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]

        for strategy in strategy_lis:

            print('Strategy:', strategy, algo_lis[strategy - 1])
            # Use strategy-specific local learning rate defaults.
            if strategy == 12:  # FedProx
                local_lr = args.fedprox_lr
                local_lr_gamma = args.fedprox_gamma_lr
            elif strategy == 13:  # FedAdam
                local_lr = args.fedadam_client_lr
                local_lr_gamma = args.fedadam_gamma_lr
            elif strategy == 14:  # FedACG
                local_lr = args.fedacg_lr
                local_lr_gamma = args.fedacg_gamma_lr
            else:
                local_lr = args.lr
                local_lr_gamma = args.gamma_lr

            network = copy.deepcopy(net_glob)
            network.train()

            load_initial_weights_if_available(network, args.dataset)

            # copy weights
            w_glob = network.state_dict()

            if strategy == 3:

                # To get the initial gradient h_glob
                glob_optimizer = torch.optim.SGD(network.parameters(), lr=1)
                loader = DataLoader(dataset_test, batch_size=args.bs)
                images, labels = next(loader.__iter__())
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = network(images)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)

                # Run one backward pass to initialize gradient buffers.
                glob_optimizer.zero_grad()
                loss.backward()

                # copy the gradient to h_glob
                h_glob = []
                for group_idx, group in enumerate(glob_optimizer.param_groups):
                    group_h = []
                    for idx, p in enumerate(group["params"]):
                        p.requires_grad = True
                        if p.grad is not None:
                            group_h.append(p.grad.clone())
                        else:
                            group_h.append(None)
                    h_glob.append(group_h)

                    # initialize h_glob with gradient shape
                h_anchor = []
                for group_idx, group in enumerate(glob_optimizer.param_groups):
                    group_h = []
                    for idx, p in enumerate(group["params"]):
                        p.requires_grad = True
                        if p.grad is not None:
                            group_h.append(torch.zeros_like(p.grad))
                        else:
                            group_h.append(None)
                    h_anchor.append(group_h)

                h_glob = h_anchor

            if strategy == 4:
                # To get the initial gradient h_glob
                glob_optimizer = torch.optim.SGD(network.parameters(), lr=1)
                loader = DataLoader(dataset_test, batch_size=args.bs)
                images, labels = next(loader.__iter__())
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = network(images)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)

                # Run one backward pass to initialize gradient buffers.
                glob_optimizer.zero_grad()
                loss.backward()

                h_glob = []
                for group_idx, group in enumerate(glob_optimizer.param_groups):
                    group_h = []
                    for idx, p in enumerate(group["params"]):
                        p.requires_grad = True
                        if p.grad is not None:
                            group_h.append(torch.zeros_like(p.grad))
                        else:
                            group_h.append(None)
                    h_glob.append(group_h)

            if strategy == 5:

                # To get the initial gradient h_glob
                glob_optimizer = torch.optim.SGD(network.parameters(), lr=1)
                loader = DataLoader(dataset_test, batch_size=args.bs)
                images, labels = next(loader.__iter__())
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = network(images)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)

                # Run one backward pass to initialize gradient buffers.
                glob_optimizer.zero_grad()
                loss.backward()

                # copy the gradient to h_glob
                h_glob = []
                for group_idx, group in enumerate(glob_optimizer.param_groups):
                    group_h = []
                    for idx, p in enumerate(group["params"]):
                        p.requires_grad = True
                        if p.grad is not None:
                            group_h.append(p.grad.clone())
                        else:
                            group_h.append(None)
                    h_glob.append(group_h)

                    # initialize h_glob with gradient shape
                h_anchor = []
                for group_idx, group in enumerate(glob_optimizer.param_groups):
                    group_h = []
                    for idx, p in enumerate(group["params"]):
                        p.requires_grad = True
                        if p.grad is not None:
                            group_h.append(torch.zeros_like(p.grad))
                        else:
                            group_h.append(None)
                    h_anchor.append(group_h)

                h_glob = h_anchor

            # Scaffold
            if strategy == 6:  # initialize c and c_i

                glob_optimizer = torch.optim.SGD(network.parameters(), lr=1)
                loader = DataLoader(dataset_test, batch_size=args.bs)
                images, labels = next(loader.__iter__())
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = network(images)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)

                # Run one backward pass to initialize gradient buffers.
                glob_optimizer.zero_grad()
                loss.backward()

                c = []
                for group_idx, group in enumerate(glob_optimizer.param_groups):
                    group_h = []
                    for idx, p in enumerate(group["params"]):
                        p.requires_grad = True
                        if p.grad is not None:
                            group_h.append(torch.zeros_like(p.grad))
                        else:
                            group_h.append(None)
                    c.append(group_h)
                # Keep per-client control variates on CPU to avoid large GPU residency.
                c_lis = [[[torch.zeros_like(q, device='cpu') for q in p] for p in c] for i in range(args.num_users)]

            # FedGAMMA
            if strategy == 7:

                glob_optimizer = torch.optim.SGD(network.parameters(), lr=1)
                loader = DataLoader(dataset_test, batch_size=args.bs)
                images, labels = next(loader.__iter__())
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = network(images)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)

                # Run one backward pass to initialize gradient buffers.
                glob_optimizer.zero_grad()
                loss.backward()

                c = []
                for group_idx, group in enumerate(glob_optimizer.param_groups):
                    group_h = []
                    for idx, p in enumerate(group["params"]):
                        p.requires_grad = True
                        if p.grad is not None:
                            group_h.append(torch.zeros_like(p.grad))
                        else:
                            group_h.append(None)
                    c.append(group_h)
                c_lis = [[[torch.zeros_like(q, device='cpu') for q in p] for p in c] for i in range(args.num_users)]

            # FedSMOO
            if strategy == 8:  # s, lam, lam_lis, mu_lis

                glob_optimizer = torch.optim.SGD(network.parameters(), lr=1)
                loader = DataLoader(dataset_test, batch_size=args.bs)
                images, labels = next(loader.__iter__())
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = network(images)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)

                # Run one backward pass to initialize gradient buffers.
                glob_optimizer.zero_grad()
                loss.backward()

                s = []
                lam = []
                for group_idx, group in enumerate(glob_optimizer.param_groups):
                    group_h = []
                    for idx, p in enumerate(group["params"]):
                        p.requires_grad = True
                        if p.grad is not None:
                            group_h.append(torch.zeros_like(p.grad))
                        else:
                            group_h.append(None)
                    s.append(group_h)

                lam = copy.deepcopy(s)
                # Store per-client dual variables in fp16 on CPU to reduce RAM.
                lam_lis = [[[torch.zeros_like(q, device='cpu', dtype=torch.float16) for q in p] for p in lam] for i in
                           range(args.num_users)]
                mu_lis = [[[torch.zeros_like(q, device='cpu', dtype=torch.float16) for q in p] for p in lam] for i in
                          range(args.num_users)]
            if strategy == 9:

                glob_optimizer = torch.optim.SGD(network.parameters(), lr=1)
                loader = DataLoader(dataset_test, batch_size=args.bs)
                images, labels = next(loader.__iter__())
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = network(images)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)

                # Run one backward pass to initialize gradient buffers.
                glob_optimizer.zero_grad()
                loss.backward()

                c = []
                for group_idx, group in enumerate(glob_optimizer.param_groups):
                    group_h = []
                    for idx, p in enumerate(group["params"]):
                        p.requires_grad = True
                        if p.grad is not None:
                            group_h.append(torch.zeros_like(p.grad))
                        else:
                            group_h.append(None)
                    c.append(group_h)
                c_lis = [[[torch.zeros_like(q, device='cpu') for q in p] for p in c] for i in range(args.num_users)]

            # FedCM
            if strategy == 10:

                glob_optimizer = torch.optim.SGD(network.parameters(), lr=1)
                loader = DataLoader(dataset_test, batch_size=args.bs)
                images, labels = next(loader.__iter__())
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = network(images)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)

                # Run one backward pass to initialize gradient buffers.
                glob_optimizer.zero_grad()
                loss.backward()

                delta_glob = []
                for group_idx, group in enumerate(glob_optimizer.param_groups):
                    group_h = []
                    for idx, p in enumerate(group["params"]):
                        p.requires_grad = True
                        if p.grad is not None:
                            group_h.append(torch.zeros_like(p.grad))
                        else:
                            group_h.append(None)
                    delta_glob.append(group_h)
                # NOTE:
                # FedCM only needs delta_glob / delta_glob_new in this implementation.
                # A full per-user delta_lis (num_users x model_size) is not used and
                # causes substantial memory growth when running multiple strategies.

            # FedLESAM
            if strategy == 11:

                glob_optimizer = torch.optim.SGD(network.parameters(), lr=1)
                loader = DataLoader(dataset_test, batch_size=args.bs)
                images, labels = next(loader.__iter__())
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = network(images)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)

                # Run one backward pass to initialize gradient buffers.
                glob_optimizer.zero_grad()
                loss.backward()

                g_glob = []
                for group_idx, group in enumerate(glob_optimizer.param_groups):
                    group_h = []
                    for idx, p in enumerate(group["params"]):
                        p.requires_grad = True
                        if p.grad is not None:
                            group_h.append(torch.zeros_like(p.grad))
                        else:
                            group_h.append(None)
                    g_glob.append(group_h)

                w_old = []
                for group_idx, group in enumerate(glob_optimizer.param_groups):
                    group_w = []
                    for idx, p in enumerate(group["params"]):
                        p.requires_grad = True
                        if p.grad is not None:
                            group_w.append(p.data.detach().clone().cpu())
                        else:
                            group_w.append(None)
                    w_old.append(group_w)

                w_t = copy.deepcopy(w_old)

                w_old_list = [copy.deepcopy(w_old) for i in
                              range(args.num_users)]

            if strategy == 13:
                w_glob = _state_dict_to_cpu(network.state_dict())
                m_adam = {k: torch.zeros_like(v) for k, v in w_glob.items()}
                v_adam = {k: torch.zeros_like(v) for k, v in w_glob.items()}

            if strategy == 14:
                w_glob = _state_dict_to_cpu(network.state_dict())
                m_acg = {k: torch.zeros_like(v) for k, v in w_glob.items()}

            # training
            loss_train = []
            cv_loss, cv_acc = [], []
            val_loss_pre, counter = 0, 0
            net_best = None
            best_loss = None
            val_acc_list, net_list = [], []

            # test accuracy list
            test_acc_list = []
            metric_history = []
            for iter in range(args.epochs):
                if strategy == 6:
                    g_i_lis = []
                if strategy in (7, 9):
                    c_i_delta_lis = []
                if strategy == 8:
                    s_i_tilde_lis = []
                loss_locals = []
                w_locals = []
                if strategy == 14:
                    anchor_state = {
                        k: (w_glob[k] + args.fedacg_lambda * m_acg[k]).detach().clone()
                        for k in w_glob.keys()
                    }

                for ele in update_lis_iter[iter]:
                    if strategy == 1:
                        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ele[0]],
                                            update_times=ele[1], iteration=iter, lr=local_lr)

                        w, loss = local.train(net=copy.deepcopy(network).to(args.device))

                        w_locals.append(_state_dict_to_cpu(w))

                        loss_locals.append(copy.deepcopy(loss))
                    if strategy == 2:
                        local = fedsam_LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ele[0]],
                                                   update_times=ele[1], iteration=iter)

                        w, loss = local.train(net=copy.deepcopy(network).to(args.device))

                        w_locals.append(_state_dict_to_cpu(w))

                        loss_locals.append(copy.deepcopy(loss))
                    if strategy == 3:
                        if iter >= 1:
                            local = fedvssam_LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ele[0]],
                                                         update_times=ele[1], h=h_temp_new, iteration=iter, lr=local_lr)

                        else:
                            local = fedvssam_LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ele[0]],
                                                         update_times=ele[1], h=h_glob, iteration=iter, lr=local_lr)

                        w, loss = local.train(net=copy.deepcopy(network).to(args.device))

                        w_locals.append(_state_dict_to_cpu(w))

                        loss_locals.append(copy.deepcopy(loss))
                    if strategy == 4:
                        local = fedmosam_LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ele[0]],
                                                     update_times=ele[1], h=h_glob, iteration=iter, lr=local_lr)

                        w, loss = local.train(net=copy.deepcopy(network).to(args.device))

                        w_locals.append(_state_dict_to_cpu(w))

                        loss_locals.append(copy.deepcopy(loss))
                    if strategy == 5:
                        local = fedsimpsam_LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ele[0]],
                                                       update_times=ele[1], h=h_glob, iteration=iter)

                        w, loss = local.train(net=copy.deepcopy(network).to(args.device))

                        w_locals.append(_state_dict_to_cpu(w))

                        loss_locals.append(copy.deepcopy(loss))
                    if strategy == 6:  # lr_global = 1

                        local = scaffold_LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ele[0]],
                                                     update_times=ele[1], c_i=c_lis[ele[0]], c_glob=c, iteration=iter)

                        w, loss, c_i_update, g_i = local.train(net=copy.deepcopy(network).to(args.device))

                        c_lis[ele[0]] = c_i_update

                        g_i_lis.append(g_i)

                        w_locals.append(_state_dict_to_cpu(w))

                        loss_locals.append(copy.deepcopy(loss))
                    if strategy == 7:
                        local = fedgamma_LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ele[0]],
                                                     update_times=ele[1], c_i=c_lis[ele[0]], c_glob=c, iteration=iter,
                                                     gamma=args.gamma_fedgamma)

                        w, loss, c_i_update, c_i_delta = local.train(net=copy.deepcopy(network).to(args.device))

                        c_lis[ele[0]] = c_i_update

                        c_i_delta_lis.append(c_i_delta)

                        w_locals.append(_state_dict_to_cpu(w))

                        loss_locals.append(copy.deepcopy(loss))

                    if strategy == 8:
                        local = fedsmoo_LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ele[0]],
                                                    update_times=ele[1], iteration=iter, s_global=s,
                                                    lam_i=lam_lis[ele[0]], mu_i=mu_lis[ele[0]], r=args.r,
                                                    beta=args.beta, lr=local_lr)

                        w, loss, s_i_tilde, lam_i, mu_i = local.train(net=copy.deepcopy(network).to(args.device))

                        lam_lis[ele[0]] = lam_i

                        mu_lis[ele[0]] = mu_i

                        s_i_tilde_lis.append(s_i_tilde)

                        w_locals.append(_state_dict_to_cpu(w))

                        loss_locals.append(copy.deepcopy(loss))

                    if strategy == 9:
                        local = feddyn_LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ele[0]],
                                                   update_times=ele[1], c_i=c_lis[ele[0]], c_glob=c, iteration=iter,
                                                   mu=args.mu)

                        w, loss, c_i_update, c_i_delta = local.train(net=copy.deepcopy(network).to(args.device))

                        c_lis[ele[0]] = c_i_update

                        c_i_delta_lis.append(c_i_delta)

                        w_locals.append(_state_dict_to_cpu(w))

                        loss_locals.append(copy.deepcopy(loss))

                    if strategy == 10:

                        if iter >= 1:
                            local = fedcm_LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ele[0]],
                                                      update_times=ele[1], delta=delta_glob_new, iteration=iter,
                                                      alpha=args.alpha_cm)

                        else:
                            local = fedcm_LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ele[0]],
                                                      update_times=ele[1], delta=delta_glob, iteration=iter,
                                                      alpha=args.alpha_cm)

                        w, loss = local.train(net=copy.deepcopy(network).to(args.device))

                        w_locals.append(_state_dict_to_cpu(w))

                        loss_locals.append(copy.deepcopy(loss))
                    if strategy == 11:
                        local = fedlesam_LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ele[0]],
                                                     update_times=ele[1], wi_old=w_old_list[ele[0]], iteration=iter)

                        w, loss = local.train(net=copy.deepcopy(network).to(args.device))

                        w_locals.append(_state_dict_to_cpu(w))

                        w_old_list[ele[0]] = copy.deepcopy(w_t)

                        loss_locals.append(copy.deepcopy(loss))

                    if strategy == 12:
                        prox_anchor = [p.detach().clone().to(args.device) for p in network.parameters()]
                        local = fedprox_LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ele[0]],
                                                    update_times=ele[1], global_params=prox_anchor, iteration=iter,
                                                    lr=local_lr)
                        w, loss = local.train(net=copy.deepcopy(network).to(args.device))
                        w_locals.append(_state_dict_to_cpu(w))
                        loss_locals.append(copy.deepcopy(loss))

                    if strategy == 13:
                        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ele[0]],
                                            update_times=ele[1], iteration=iter, lr=local_lr)
                        w, loss = local.train(net=copy.deepcopy(network).to(args.device))
                        w_locals.append(_state_dict_to_cpu(w))
                        loss_locals.append(copy.deepcopy(loss))

                    if strategy == 14:
                        local = fedacg_LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ele[0]],
                                                   update_times=ele[1], anchor_state=anchor_state, iteration=iter,
                                                   lr=local_lr)
                        w, loss = local.train(net=copy.deepcopy(network).to(args.device))
                        w_locals.append(_state_dict_to_cpu(w))
                        loss_locals.append(copy.deepcopy(loss))
                if strategy == 3:
                    global_gamma = args.fedvssam_gamma_g
                    global_rho = 0.0
                    global_eta = 4


                    # Acquire the updated global gradient g_glob
                    w_temp = FedAvg(w_locals)
                    network_temp = copy.deepcopy(network)
                    network_temp.train()
                    network_temp.load_state_dict(w_temp)

                    optimizer_temp = torch.optim.SGD(network_temp.parameters(), lr=1)

                    # Acquire g_glob
                    g_glob = []
                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        group_diffs = []
                        for idx, p in enumerate(group["params"]):
                            p.requires_grad = True
                            if h_anchor[group_idx][idx] is not None:
                                param_temp = optimizer_temp.param_groups[group_idx]["params"][idx].data.detach().clone()
                                param_diff = p.data.detach().clone() - param_temp.data.detach().clone()

                                group_diffs.append(param_diff)
                            else:
                                group_diffs.append(None)
                        g_glob.append(group_diffs)

                        # Update h_glob with (1-gamma)*h_glob + gamma*g_glob
                    h_temp = []
                    for group_idx, group in enumerate(h_glob):
                        group_h = []
                        for idx in range(len(group)):
                            if g_glob[group_idx][idx] is not None:
                                h_current = (1 - global_gamma) * h_glob[group_idx][idx] + global_gamma * \
                                            g_glob[group_idx][idx]
                                group_h.append(h_current)
                            else:
                                group_h.append(None)
                        h_temp.append(group_h)

                    h_glob = h_temp
                    h_temp_new = []
                    for group_idx, group in enumerate(h_glob):
                        group_h = []
                        for idx in range(len(group)):
                            if g_glob[group_idx][idx] is not None:
                                h_current = h_glob[group_idx][idx] / (local_lr * args.n_epochs)
                                group_h.append(h_current)
                            else:
                                group_h.append(None)
                        h_temp_new.append(group_h)

                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        for idx, p in enumerate(group["params"]):
                            if h_temp_new[group_idx][idx] is not None:
                                # Update the parameter using the current value and h_glob
                                p.data = p.data.detach().clone() - global_eta * h_temp_new[group_idx][idx]
                elif strategy == 4:

                    w_temp = FedAvg(w_locals)

                    network_temp = copy.deepcopy(network)
                    network_temp.train()
                    network_temp.load_state_dict(w_temp)
                    optimizer_temp = torch.optim.SGD(network_temp.parameters(), lr=1)

                    # Acquire g_glob
                    scale = local_lr * args.n_epochs
                    g_glob = []
                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        group_diffs = []
                        for idx, p in enumerate(group["params"]):

                            p.requires_grad = True
                            if h_glob[group_idx][idx] is not None:

                                param_temp = optimizer_temp.param_groups[group_idx]["params"][idx].data.detach().clone()

                                param_diff = (p.data.detach().clone() - param_temp.data.detach().clone()) / scale
                                group_diffs.append(param_diff)
                            else:
                                group_diffs.append(None)
                        g_glob.append(group_diffs)

                    h_glob = g_glob

                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        for idx, p in enumerate(group["params"]):
                            if h_glob[group_idx][idx] is not None:
                                # Update the parameter using the current value and h_glob
                                p.data = p.data.detach().clone() - scale * h_glob[group_idx][idx]

                elif strategy == 5:
                    global_gamma = 1.0
                    global_rho = 0.0

                    # Acquire the updated global gradient g_glob
                    w_temp = FedAvg(w_locals)
                    network_temp = copy.deepcopy(network)
                    network_temp.train()
                    network_temp.load_state_dict(w_temp)
                    optimizer_temp = torch.optim.SGD(network_temp.parameters(), lr=1)

                    # Acquire g_glob
                    g_glob = []
                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        group_diffs = []
                        for idx, p in enumerate(group["params"]):
                            p.requires_grad = True
                            if h_anchor[group_idx][idx] is not None:
                                param_temp = optimizer_temp.param_groups[group_idx]["params"][idx].data.detach().clone()
                                param_diff = p.data.detach().clone() - param_temp.data.detach().clone()
                                group_diffs.append(param_diff)
                            else:
                                group_diffs.append(None)
                        g_glob.append(group_diffs)

                        # Update h_glob with (1-gamma)*h_glob + gamma*g_glob
                    h_temp = []
                    for group_idx, group in enumerate(h_glob):
                        group_h = []
                        for idx in range(len(group)):
                            if g_glob[group_idx][idx] is not None:
                                h_current = (1 - global_gamma) * h_glob[group_idx][idx] + global_gamma * \
                                            g_glob[group_idx][idx]
                                group_h.append(h_current)
                            else:
                                group_h.append(None)
                        h_temp.append(group_h)

                    h_glob = h_temp

                    # Compute the gradient norm of h_glob
                    grad_norm = 0
                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        for idx, p in enumerate(group["params"]):
                            p.requires_grad = True
                            if h_glob[group_idx][idx] == None:
                                continue
                            else:
                                grad_norm += h_glob[group_idx][idx].norm(p=2)

                                # Use the updated h_glob to update global model parameter
                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        scale = global_rho / (grad_norm + 1e-7)
                        for idx, p in enumerate(group["params"]):
                            if h_glob[group_idx][idx] is not None:
                                p.data = p.data.detach().clone() - h_glob[group_idx][idx]

                elif strategy == 6:

                    c_new = copy.deepcopy(c)


                    c_delta = []
                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        group_h = []
                        for idx, p in enumerate(group["params"]):
                            p.requires_grad = True
                            if p.grad is not None:
                                group_h.append(torch.zeros_like(p.grad))
                            else:
                                group_h.append(None)
                        c_delta.append(group_h)

                    for ele in g_i_lis:
                        for i in range(len(c)):
                            for j in range(len(c[i])):
                                c_delta[i][j] += ele[i][j] / (args.n_epochs * args.lr * args.num_users)

                    for i in range(len(c)):
                        for j in range(len(c[i])):
                            c_new[i][j] = c[i][j] * (args.num_users - args.user_subset) / args.num_users + c_delta[i][j]

                    c = copy.deepcopy(c_new)

                    w_glob = FedAvg(w_locals)
                    # copy weight to network (Update Aggregation)
                    network.load_state_dict(w_glob)

                elif strategy == 7:
                    for ele in c_i_delta_lis:
                        for i in range(len(c)):
                            for j in range(len(c[i])):
                                c[i][j] += ele[i][j].to(args.device) / args.num_users

                    w_glob = FedAvg(w_locals)
                    # copy weight to network (Update Aggregation)
                    network.load_state_dict(w_glob)

                elif strategy == 8:
                    s_temp = []
                    lam_temp = []
                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        group_h = []
                        for idx, p in enumerate(group["params"]):
                            p.requires_grad = True
                            if p.grad is not None:
                                group_h.append(torch.zeros_like(p.grad))
                            else:
                                group_h.append(None)
                        s_temp.append(group_h)

                    lam_temp = copy.deepcopy(s_temp)

                    for ele in s_i_tilde_lis:
                        for i in range(len(s)):
                            for j in range(len(s[i])):
                                s_temp[i][j] += ele[i][j].to(args.device) / args.user_subset


                    s_norm_sq = 0
                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        for idx, p in enumerate(group['params']):
                            if p.grad is not None:
                                s_norm_sq += s_temp[group_idx][idx].norm(p=2).pow(2)

                    s_norm = s_norm_sq.sqrt()


                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        scale = args.r / (s_norm + 1e-7)
                        for idx, p in enumerate(group["params"]):
                            if s[group_idx][idx] is not None:
                                s[group_idx][idx] = s_temp[group_idx][idx] * scale


                    w_temp = FedAvg(w_locals)
                    net_temp = copy.deepcopy(network)
                    net_temp.load_state_dict(w_temp)
                    optimizer_temp = torch.optim.SGD(net_temp.parameters(), lr=1)
                    for group_idx, (group, group_glob) in enumerate(
                            zip(optimizer_temp.param_groups, glob_optimizer.param_groups)):
                        for idx, (p, p_glob) in enumerate(zip(group['params'], group_glob['params'])):
                            lam_temp[group_idx][idx] = lam[group_idx][idx] - args.user_subset / (
                                        args.beta * args.num_users) * (
                                                                   p.data.detach().clone() - p_glob.data.detach().clone())

                    lam = copy.deepcopy(lam_temp)


                    w_glob = FedAvg(w_locals)
                    network.load_state_dict(w_glob)
                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        for idx, p in enumerate(group['params']):
                            p.data = p.data.detach().clone() - args.beta * lam[group_idx][idx]

                elif strategy == 9:

                    for ele in c_i_delta_lis:
                        for i in range(len(c)):
                            for j in range(len(c[i])):
                                c[i][j] += ele[i][j].to(args.device) / (args.num_users * args.mu)

                    w_glob = FedAvg(w_locals)
                    # copy weight to network (Update Aggregation)
                    network.load_state_dict(w_glob)

                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        for idx, p in enumerate(group['params']):
                            p.data = p.data.detach().clone() - c[group_idx][idx]

                elif strategy == 10:

                    # Acquire the updated global gradient g_glob
                    w_temp = FedAvg(w_locals)
                    network_temp = copy.deepcopy(network)
                    network_temp.train()
                    network_temp.load_state_dict(w_temp)
                    optimizer_temp = torch.optim.SGD(network_temp.parameters(), lr=1)

                    # Acquire delta_glob
                    delta_temp = []
                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        group_diffs = []
                        for idx, p in enumerate(group["params"]):
                            p.requires_grad = True
                            if delta_glob[group_idx][idx] is not None:
                                param_temp = optimizer_temp.param_groups[group_idx]["params"][idx].data.detach().clone()
                                param_diff = p.data.detach().clone() - param_temp.data.detach().clone()
                                group_diffs.append(param_diff)
                            else:
                                group_diffs.append(None)
                        delta_temp.append(group_diffs)

                    delta_glob = delta_temp

                    # Use the updated delta_glob to update global model parameter
                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        for idx, p in enumerate(group["params"]):
                            if delta_glob[group_idx][idx] is not None:
                                p.data = p.data.detach().clone() - delta_glob[group_idx][idx]

                    delta_glob_new = []
                    for group_idx, group in enumerate(delta_glob):
                        group_h = []
                        for idx in range(len(group)):
                            if delta_glob[group_idx][idx] is not None:
                                h_current = delta_glob[group_idx][idx] / (args.lr * args.n_epochs)
                                group_h.append(h_current)
                            else:
                                group_h.append(None)
                        delta_glob_new.append(group_h)

                elif strategy == 11:
                    eta_glob = 1

                    # Acquire the updated global gradient g_glob
                    w_temp = FedAvg(w_locals)
                    network_temp = copy.deepcopy(network)
                    network_temp.train()
                    network_temp.load_state_dict(w_temp)
                    optimizer_temp = torch.optim.SGD(network_temp.parameters(), lr=1)

                    # Acquire g_glob
                    g_glob_temp = []
                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        group_diffs = []
                        for idx, p in enumerate(group["params"]):
                            p.requires_grad = True
                            if g_glob[group_idx][idx] is not None:
                                param_temp = optimizer_temp.param_groups[group_idx]["params"][idx].data.detach().clone()
                                param_diff = p.data.detach().clone() - param_temp.data.detach().clone()
                                group_diffs.append(param_diff)
                            else:
                                group_diffs.append(None)
                        g_glob_temp.append(group_diffs)

                    g_glob = g_glob_temp

                    # Use the updated delta_glob to update global model parameter
                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        for idx, p in enumerate(group["params"]):
                            if g_glob[group_idx][idx] is not None:
                                p.data = p.data.detach().clone() - eta_glob * g_glob[group_idx][idx]

                    # Update w_t
                    w_t_temp = []
                    for group_idx, group in enumerate(glob_optimizer.param_groups):
                        group_w = []
                        for idx, p in enumerate(group['params']):
                            param_temp = p.data.detach().clone().cpu()
                            group_w.append(param_temp)
                        w_t_temp.append(group_w)

                    w_t = w_t_temp

                elif strategy == 12:
                    w_glob = FedAvg(w_locals)
                    network.load_state_dict(w_glob)

                elif strategy == 13:
                    w_avg = FedAvg(w_locals)
                    for k in w_glob.keys():
                        delta = w_avg[k] - w_glob[k]
                        m_adam[k] = args.fedadam_beta1 * m_adam[k] + (1 - args.fedadam_beta1) * delta
                        v_adam[k] = args.fedadam_beta2 * v_adam[k] + (1 - args.fedadam_beta2) * (delta * delta)
                        w_glob[k] = w_glob[k] + args.fedadam_eta * m_adam[k] / (torch.sqrt(v_adam[k]) + args.fedadam_tau)
                    network.load_state_dict(w_glob)

                elif strategy == 14:
                    w_avg = FedAvg(w_locals)
                    for k in w_glob.keys():
                        delta = w_avg[k] - anchor_state[k]
                        # FedACG momentum update should be damped to avoid
                        # unbounded accumulation under strong heterogeneity.
                        m_acg[k] = args.fedacg_lambda * m_acg[k] + delta
                        w_glob[k] = w_glob[k] + m_acg[k]
                    network.load_state_dict(w_glob)

                else:
                    w_glob = FedAvg(w_locals)
                    # copy weight to network (Update Aggregation)
                    network.load_state_dict(w_glob)

                loss_avg = sum(loss_locals) / len(loss_locals)
                loss_train.append(loss_avg)
                print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))

                completed_round = iter + 1
                if completed_round % args.eval_interval == 0 or completed_round == args.epochs:
                    acc_test, loss_test = test_img(network, dataset_test, args)
                    record_accuracy(strategy, completed_round, acc_test, metric_history)
                    print("Round, {:3d}, Testing accuracy: {:.2f}".format(completed_round, acc_test))
                local_lr = local_lr * local_lr_gamma

            final_acc, best_acc, rd_target = summarize_history(metric_history, args.target_acc)
            if final_acc is not None:
                summary_rows.append({
                    'dataset': args.dataset,
                    'partition': datatype,
                    'alpha': float(args.alpha),
                    'classes_per_client': int(args.num_classes_per_user),
                    'algorithm': algo_lis[strategy - 1],
                    'final_acc': final_acc,
                    'best_acc': best_acc,
                    'rd_at_target': rd_target,
                    'target_acc': float(args.target_acc),
                    'eval_points': len(metric_history),
                })
                csv_path, md_path = write_summary(summary_rows, args, datatype)
                print(f'Summary updated: {csv_path}')

            # Explicit cleanup: per-strategy objects stay in module scope otherwise.
            for _name in [
                'network', 'w_glob', 'glob_optimizer', 'loader', 'images', 'labels', 'outputs', 'criterion', 'loss',
                'loss_train', 'cv_loss', 'cv_acc', 'val_acc_list', 'net_list', 'test_acc_list',
                'h_glob', 'h_anchor', 'h_temp', 'h_temp_new', 'g_glob', 'g_glob_temp',
                'c', 'c_lis', 'c_new', 'c_delta', 'g_i_lis', 'c_i_delta_lis',
                's', 'lam', 'mu', 'lam_lis', 'mu_lis', 's_i_tilde_lis', 's_temp', 'lam_temp',
                'delta_glob', 'delta_lis', 'delta_glob_new',
                'w_old', 'w_t', 'w_old_list', 'w_temp', 'network_temp', 'optimizer_temp', 'net_temp',
                'w_locals', 'loss_locals', 'local', 'm_adam', 'v_adam', 'm_acg', 'anchor_state', 'prox_anchor',
                'w_avg'
            ]:
                globals().pop(_name, None)
            _release_cached_memory()

    # accuracy average
    fedavg_acc = [i / args.monto_carlo_times for i in fedavg_acc]
    fedsam_acc = [i / args.monto_carlo_times for i in fedsam_acc]
    fedvssam_acc = [i / args.monto_carlo_times for i in fedvssam_acc]
    fedmosam_acc = [i / args.monto_carlo_times for i in fedmosam_acc]
    fedsimpsam_acc = [i / args.monto_carlo_times for i in fedsimpsam_acc]
    scaffold_acc = [i / args.monto_carlo_times for i in scaffold_acc]
    fedgamma_acc = [i / args.monto_carlo_times for i in fedgamma_acc]
    fedsmoo_acc = [i / args.monto_carlo_times for i in fedsmoo_acc]
    feddyn_acc = [i / args.monto_carlo_times for i in feddyn_acc]
    fedcm_acc = [i / args.monto_carlo_times for i in fedcm_acc]
    fedlesam_acc = [i / args.monto_carlo_times for i in fedlesam_acc]
    fedprox_acc = [i / args.monto_carlo_times for i in fedprox_acc]
    fedadam_acc = [i / args.monto_carlo_times for i in fedadam_acc]
    fedacg_acc = [i / args.monto_carlo_times for i in fedacg_acc]

    os.makedirs(args.save_dir, exist_ok=True)
    exp_tag = 'data_{}_set_{}_noniid_{}_c_{}_M_{}_S_{}_lr_{}_n_ep_{}_mtcl_{}'.format(
        datatype, args.dataset, args.alpha, args.num_classes_per_user, args.num_users, args.user_subset, args.lr,
        args.n_epochs, args.monto_carlo_times
    )

    rounds_arr = np.arange(1, args.epochs + 1, dtype=np.int64)

    # Map: algorithm name -> python list curve
    algo_curves = {
        'FedAvg': fedavg_acc,
        'FedSAM': fedsam_acc,
        'FedVSSAM': fedvssam_acc,
        'MoFedSAM': fedmosam_acc,
        'FedLVSSAM': fedsimpsam_acc,
        'Scaffold': scaffold_acc,
        'FedGAMMA': fedgamma_acc,
        'FedSMOO': fedsmoo_acc,
        'FedDyn': feddyn_acc,
        'FedCM': fedcm_acc,
        'FedLESAM': fedlesam_acc,
        'FedProx': fedprox_acc,
        'FedAdam': fedadam_acc,
        'FedACG': fedacg_acc,
    }

    # Helper: decide whether a curve is "valid / actually ran"
    def _curve_is_valid(curve, epochs):
        if curve is None:
            return False
        if len(curve) != epochs:
            return False
        arr = np.asarray(curve, dtype=np.float32)
        # If it is all zeros (common when algorithm not executed), treat as invalid
        if np.all(arr == 0):
            return False
        # If contains NaN everywhere, also invalid
        if np.all(np.isnan(arr)):
            return False
        return True

    # Save each algorithm separately
    meta = {
        'exp_tag': exp_tag,
        'datatype': datatype,
        'dataset': args.dataset,
        'num_users': int(args.num_users),
        'user_subset': int(args.user_subset),
        'lr': float(args.lr),
        'local_epochs_or_steps': int(args.n_epochs),
        'monto_carlo_times': int(args.monto_carlo_times),
        'epochs': int(args.epochs),
    }

    saved_algos = []
    for algo_name, curve in algo_curves.items():
        if not _curve_is_valid(curve, args.epochs):
            continue

        payload = dict(meta)
        payload['algo'] = algo_name
        payload['rounds'] = rounds_arr
        payload['acc'] = np.asarray(curve, dtype=np.float32)

        out_path = os.path.join(args.save_dir, '{}_acc_{}.npy'.format(exp_tag, algo_name))
        np.save(out_path, payload, allow_pickle=True)
        saved_algos.append(algo_name)
    fedavg_acc_ = []
    fedsam_acc_ = []
    fedvssam_acc_ = []
    fedmosam_acc_ = []
    fedsimpsam_acc_ = []
    scaffold_acc_ = []
    fedgamma_acc_ = []
    fedsmoo_acc_ = []
    feddyn_acc_ = []
    fedcm_acc_ = []
    fedlesam_acc_ = []
    fedprox_acc_ = []
    fedadam_acc_ = []
    fedacg_acc_ = []

    iter_ = []

    interval = 0
    if args.dataset == 'cifar10':
        interval = 10
    elif args.dataset == 'cifar100':
        interval = 40
    if interval <= 0:
        interval = args.eval_interval if args.eval_interval > 0 else 1

    for i in range(args.epochs):

        if (i + 1) % interval == 0 or i == args.epochs - 1:
            fedavg_acc_.append(fedavg_acc[i])
            fedsam_acc_.append(fedsam_acc[i])
            fedvssam_acc_.append(fedvssam_acc[i])
            fedmosam_acc_.append(fedmosam_acc[i])
            fedsimpsam_acc_.append(fedsimpsam_acc[i])
            scaffold_acc_.append(scaffold_acc[i])
            fedgamma_acc_.append(fedgamma_acc[i])
            fedsmoo_acc_.append(fedsmoo_acc[i])
            feddyn_acc_.append(feddyn_acc[i])
            fedcm_acc_.append(fedcm_acc[i])
            fedlesam_acc_.append(fedlesam_acc[i])
            fedprox_acc_.append(fedprox_acc[i])
            fedadam_acc_.append(fedadam_acc[i])
            fedacg_acc_.append(fedacg_acc[i])

            iter_.append(i + 1)

    # plot and save acc curve
    figure()

    plot(iter_, fedavg_acc_, linestyle='-', label='FedAvg')
    xlabel('Communication Rounds')
    ylabel('Test Accuracy')

    plot(iter_, fedsam_acc_, linestyle='-', label='FedSAM')
    xlabel('Communication Rounds')
    ylabel('Test Accuracy')

    plot(iter_, fedmosam_acc_, linestyle='-.', label='MoFedSAM')
    xlabel('Communication Rounds')
    ylabel('Test Accuracy')

    plot(iter_, fedsimpsam_acc_, linestyle='-', label='FedLVSSAM')
    xlabel('Communication Rounds')
    ylabel('Test Accuracy')

    plot(iter_, fedvssam_acc_, linestyle='-', label='FedVSSAM')
    xlabel('Communication Rounds')
    ylabel('Test Accuracy')

    plot(iter_, scaffold_acc_, linestyle='--', label='Scaffold')
    xlabel('Communication Rounds')
    ylabel('Test Accuracy')

    plot(iter_, fedgamma_acc_, linestyle='-', label='FedGAMMA')
    xlabel('Communication Rounds')
    ylabel('Test Accuracy')

    plot(iter_, fedsmoo_acc_, linestyle='-', label='FedSMOO')
    xlabel('Communication Rounds')
    ylabel('Test Accuracy')

    plot(iter_, feddyn_acc_, linestyle='-', label='FedDyn')
    xlabel('Communication Rounds')
    ylabel('Test Accuracy')

    plot(iter_, fedcm_acc_, linestyle='-', label='FedCM')
    xlabel('Communication Rounds')
    ylabel('Test Accuracy')

    plot(iter_, fedlesam_acc_, linestyle='-', label='FedLESAM')
    xlabel('Communication Rounds')
    ylabel('Test Accuracy')

    plot(iter_, fedprox_acc_, linestyle='-', label='FedProx')
    xlabel('Communication Rounds')
    ylabel('Test Accuracy')

    plot(iter_, fedadam_acc_, linestyle='-', label='FedAdam')
    xlabel('Communication Rounds')
    ylabel('Test Accuracy')

    plot(iter_, fedacg_acc_, linestyle='-', label='FedACG')
    xlabel('Communication Rounds')
    ylabel('Test Accuracy')
    plt.legend()
    plt.savefig(
        os.path.join(args.save_dir, 'data_{}_set_{}_noniid_{}_c_{}_M_{}_S_{}_lr_{}_n_ep_{}_mtcl_{}.png'.format(
            datatype, args.dataset, args.alpha, args.num_classes_per_user, args.num_users, args.user_subset, args.lr,
            args.n_epochs, args.monto_carlo_times)))
    plt.close()
