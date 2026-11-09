import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.0, help="SGD momentum (should be 0 forever)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], default='cifar10', help="name of dataset")
    parser.add_argument('--data_dir', type=str, default='./data', help='dataset root directory')
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--pathological', action='store_true', help='whether pathological non-iid or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')


    # newly added arguments
    parser.add_argument('--monto_carlo_times', type=int, default=1, help="repeated averaging")
    parser.add_argument('--n_epochs', type=int, default=5, help="average number of update times per round per device")
    parser.add_argument('--user_subset', type=int, default=10, help="number of users drawn per round")
    parser.add_argument('--algos', type=str,
                        default='FedAvg,FedProx,FedAdam,SCAFFOLD,FedACG,FedCM,FedDyn,FedSAM,MoFedSAM,FedGAMMA,FedSMOO,FedLESAM,FedVSSAM',
                        help='comma-separated algorithms to run')
    parser.add_argument('--eval_interval', type=int, default=20, help='evaluate every N completed rounds')
    parser.add_argument('--target_acc', type=float, default=90.0, help='accuracy threshold for round-to-target summary')
    parser.add_argument('--save_dir', type=str, default='./save', help='result output directory')
    parser.add_argument('--prepare_only', action='store_true', help='only download/cache the dataset and exit')


    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet distribution parameter')
    parser.add_argument('--num_classes_per_user', type=int, default=2, help='Pathological distribution parameter')
    parser.add_argument('--r', type=float, default=0.1, help='FedSMOO SAM learning rate')
    parser.add_argument('--beta', type=float, default=10, help='FedSMOO penalized coefficient, 10 for CIFAR10, and 100 for CIFAR100')
    parser.add_argument('--mu', type=float, default=0.01, help='FedDyn regularization parameter')
    parser.add_argument('--alpha_cm', type=float, default=0.1, help='FedCM decay parameter')
    parser.add_argument('--gamma_fedgamma', type=float, default=0.05, help='FedGAMMA perturbation parameter')
    parser.add_argument('--rho_mosam', type=float, default=-1.0,
                        help='FedMOSAM perturbation radius rho; if <0, use dataset-aware default (0.01 for CIFAR100, else 0.1)')
    parser.add_argument('--gamma_mosam', type=float, default=0.1,
                        help='FedMOSAM gradient mixing factor gamma')
    parser.add_argument('--sam_rho', type=float, default=0.1, help='FedSAM perturbation radius')
    parser.add_argument('--fedvssam_rho', type=float, default=0.05, help='FedVSSAM perturbation radius')
    parser.add_argument('--fedvssam_gamma_l', type=float, default=0.1, help='FedVSSAM local mixing coefficient')
    parser.add_argument('--fedvssam_gamma_g', type=float, default=0.6, help='FedVSSAM global mixing coefficient')
    # FedProx (paper: proximal term strength tuned from {1e-3, 1e-2, 1e-1, 1}; we use 1e-2 by default)
    parser.add_argument('--mu_prox', type=float, default=0.1, help='FedProx proximal coefficient mu')
    parser.add_argument('--fedprox_lr', type=float, default=0.1, help='FedProx client learning rate')
    parser.add_argument('--fedprox_gamma_lr', type=float, default=1.0, help='FedProx lr decay per round')

    # FedAdam (ICLR 2021, CIFAR-10 Dirichlet setting in paper appendix)
    parser.add_argument('--fedadam_client_lr', type=float, default=10 ** (-1.5),
                        help='FedAdam client lr eta_l (default 10^-1.5)')
    parser.add_argument('--fedadam_beta1', type=float, default=0.9, help='FedAdam beta1')
    parser.add_argument('--fedadam_beta2', type=float, default=0.99, help='FedAdam beta2')
    parser.add_argument('--fedadam_tau', type=float, default=1e-3, help='FedAdam tau (default 10^-3)')
    parser.add_argument('--fedadam_eta', type=float, default=1e-2,
                        help='FedAdam server lr eta (default 10^-2)')
    parser.add_argument('--fedadam_gamma_lr', type=float, default=1.0, help='FedAdam client lr decay per round')

    # FedACG (CVPR 2024, Table 5 on CIFAR-10 Dir(0.3): lambda=0.85, beta=0.01)
    parser.add_argument('--fedacg_lambda', type=float, default=0.85,
                        help='FedACG lookahead coefficient lambda in [0,1)')
    parser.add_argument('--fedacg_beta', type=float, default=0.01, help='FedACG local proximal coefficient beta')
    parser.add_argument('--fedacg_lr', type=float, default=0.1, help='FedACG client learning rate')
    parser.add_argument('--fedacg_gamma_lr', type=float, default=1.0, help='FedACG lr decay per round')
    parser.add_argument('--local_epochs', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--gamma_lr', type=float, default=0.99995, help='learning rate decay')
    args = parser.parse_args()
    return args
