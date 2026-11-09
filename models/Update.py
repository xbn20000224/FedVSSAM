"""Local client update rules for FedVSSAM and baseline methods."""
import torch
from torch import nn

from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy

from optimizer.sam import SAM
from optimizer.vssam import VSSAM
from optimizer.mosam import MOSAM
from optimizer.simpsam import SIMPSAM
from optimizer.gamma import GAMMA
from optimizer.smoo import SMOO
from optimizer.lesam import LESAM


class DatasetSplit(Dataset):
    """Dataset wrapper that exposes a client-specific index subset."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[(self.idxs[item])]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, update_times=None,decay_iter=None,iteration=None, lr=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        g = torch.Generator()
        g.manual_seed(args.seed + iteration)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, generator=g)
        self.update_times = update_times
        self.decay_iter = decay_iter
        self.lr = args.lr if lr is None else lr

    def train(self, net):
        net.train()
        lr_ = self.lr
        optimizer = torch.optim.SGD(net.parameters(), lr=lr_, momentum=self.args.momentum)

        count=0
        batch_loss=[]
        for iter in range(100):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                count+=1
                if count>self.update_times:
                    break
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
        return net.state_dict(), sum(batch_loss) / len(batch_loss)


class fedprox_LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, update_times=None, iteration=None, global_params=None, lr=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        g = torch.Generator()
        g.manual_seed(args.seed + iteration)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, generator=g)
        self.update_times = update_times
        self.mu_prox = args.mu_prox
        self.global_params = global_params
        self.lr = args.lr if lr is None else lr

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        if self.global_params is None:
            anchor_params = [p.detach().clone() for p in net.parameters()]
        else:
            anchor_params = [p.detach().clone().to(self.args.device) for p in self.global_params]

        count = 0
        batch_loss = []
        while count < self.update_times:
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if count >= self.update_times:
                    break
                count += 1

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                logits = net(images)
                loss = self.loss_func(logits, labels)

                prox_term = 0.0
                for p, p0 in zip(net.parameters(), anchor_params):
                    prox_term += torch.sum((p - p0) ** 2)
                loss = loss + 0.5 * self.mu_prox * prox_term

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

        return net.state_dict(), sum(batch_loss) / len(batch_loss)


class fedsam_LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, update_times=None, iteration=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        g = torch.Generator()
        g.manual_seed(args.seed + iteration)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, generator=g)
        self.update_times = update_times

    def train(self, net):

        net.train()
        lr_ = self.args.lr
        optimizer_original =  torch.optim.SGD(net.parameters(), lr=lr_, momentum=self.args.momentum)


        optimizer_ = SAM(net.parameters(), optimizer_original, self.args.sam_rho, adaptive=False)
        count=0
        batch_loss = []

        while count < self.update_times:
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if count>self.update_times:
                    break
                count = count + 1

                images, labels = images.to(self.args.device), labels.to(self.args.device)

                optimizer_.paras = [images, labels, self.loss_func, net]

                loss = optimizer_.step()

                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)
                optimizer_original.step()
                with torch.no_grad():
                    batch_loss.append(loss.item())
        return net.state_dict(), sum(batch_loss) / len(batch_loss)


class fedvssam_LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, update_times=None, h=None, iteration=None, lr=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        g = torch.Generator()
        g.manual_seed(args.seed + iteration)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, generator=g)
        self.h = h
        self.iteration = iteration
        self.update_times = update_times
        self.lr = lr
        self.rho = args.fedvssam_rho

    def train(self, net):

        net.train()
        lr_ = self.lr
        optimizer_original = torch.optim.SGD(net.parameters(), lr=lr_, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        local_gamma = self.args.fedvssam_gamma_l

        h_ = self.h

        optimizer_ = VSSAM(net.parameters(), optimizer_original, self.rho, gamma=local_gamma, adaptive=False)


        count=0
        batch_loss = []
        while count < self.update_times:
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if count>=self.update_times:
                    break

                count = count + 1

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer_.paras = [images, labels, self.loss_func, net]

                # compute the adjusted gradient
                loss, h_adj = optimizer_.step(h_)

                # an interative form is used if activated

                # grad norm clipping
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)

                # update with h_
                optimizer_original.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))

                with torch.no_grad():
                    batch_loss.append(loss.item())
        return net.state_dict(), sum(batch_loss) / len(batch_loss)


class fedmosam_LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, update_times=None, h=None, iteration=None, lr=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        g = torch.Generator()
        g.manual_seed(args.seed + iteration)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, generator=g)
        self.update_times = update_times
        self.h = h
        self.iteration = iteration
        self.lr = lr
        default_rho = 0.01 if args.dataset == 'cifar100' else 0.1
        self.rho = args.rho_mosam if args.rho_mosam >= 0 else default_rho
        self.local_gamma = args.gamma_mosam

    def train(self, net):

        net.train()
        lr_ = self.lr
        optimizer_original = torch.optim.SGD(net.parameters(), lr=lr_, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        local_gamma = self.local_gamma

        h_ = self.h

        optimizer_ = MOSAM(net.parameters(), optimizer_original, self.rho, gamma=local_gamma, adaptive=False)


        count=0
        batch_loss = []
        for iter in range(100):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                count = count + 1
                if count>self.update_times:
                    break

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer_.paras = [images, labels, self.loss_func, net]

                # compute the adjusted gradient
                loss = optimizer_.step(h_)

                # grad norm clipping
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)

                # update with h_
                optimizer_original.step()

            with torch.no_grad():
                batch_loss.append(loss.item())
        return net.state_dict(), sum(batch_loss) / len(batch_loss)


class fedsimpsam_LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, update_times=None, h=None, iteration=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        g = torch.Generator()
        g.manual_seed(args.seed + iteration)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, generator=g)
        self.update_times = update_times
        self.h = h
        self.iteration = iteration

    def train(self, net):

        net.train()
        lr_ = self.args.lr
        optimizer_original = torch.optim.SGD(net.parameters(), lr=lr_, momentum=self.args.momentum)
        local_gamma = 0.2

        h_ = self.h

        optimizer_ = SIMPSAM(net.parameters(), optimizer_original, 0.5, gamma=local_gamma, adaptive=False)


        count=0
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            count = count + 1
            if count>self.update_times:
                break

            images, labels = images.to(self.args.device), labels.to(self.args.device)
            optimizer_.paras = [images, labels, self.loss_func, net]


            # compute the adjusted gradient
            loss, h_adj = optimizer_.step(h_)

            # an interative form is used if activated
            h_ = h_adj


            # grad norm clipping
            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)

            # update with h_
            optimizer_original.step()

            with torch.no_grad():
                batch_loss.append(loss.item())
        return net.state_dict(), sum(batch_loss) / len(batch_loss)


class scaffold_LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, update_times=None, c_i=None, c_glob=None, iteration=None ):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        g = torch.Generator()
        g.manual_seed(args.seed + iteration)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, generator=g)
        self.update_times = update_times
        self.c_i = [[p.to(args.device, dtype=torch.float32) if p is not None else None for p in group] for group in c_i]
        self.c_glob = c_glob
    def train(self, net):

        c_i_new = copy.deepcopy(self.c_i)

        net.train()
        lr_ = self.args.lr
        n_epochs_ = self.args.n_epochs

        # -------------
        optimizer_ = torch.optim.SGD(net.parameters(), lr=lr_, momentum=self.args.momentum)


        w_glob = []
        for group_idx, group in enumerate(optimizer_.param_groups):
            group_w = []
            for idx, p in enumerate(group['params']):
                group_w.append(p.data.detach().clone())
            w_glob.append(group_w)

        net.zero_grad()

        batch_loss=[]
        count = 0
        for iter in range(100):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                count+=1
                if count>self.update_times:
                    break

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                for group_idx, group in enumerate(optimizer_.param_groups):
                    for idx, p in enumerate(group["params"]):
                        p.requires_grad = True
                        if p.grad is not None:
                            temp = p.grad.clone() - self.c_i[group_idx][idx].clone() + self.c_glob[group_idx][idx]
                            p.grad = temp

                optimizer_.step()
                batch_loss.append(loss.item())

        g_i = []
        for group_idx, group in enumerate(optimizer_.param_groups):
            temp_grad = []
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True

                temp = w_glob[group_idx][idx] - p.data.clone()
                c_i_new[group_idx][idx] = c_i_new[group_idx][idx] - self.c_glob[group_idx][idx] + temp / (n_epochs_ * lr_)
                temp_grad.append(temp)

            g_i.append(temp_grad)





        c_i_new_cpu = [[p.cpu() if p is not None else None for p in group] for group in c_i_new]
        return net.state_dict(), sum(batch_loss)/len(batch_loss) , c_i_new_cpu, g_i


class fedgamma_LocalUpdate(object):
    def __init__(self, args=None, dataset=None, idxs=None, update_times=None, c_i=None, c_glob=None, iteration=None, gamma=0.05):
        self.args = args
        g = torch.Generator()
        g.manual_seed(args.seed + iteration)
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True, generator=g)
        self.loss_func = nn.CrossEntropyLoss()
        self.ci = [[p.to(args.device) if p is not None else None for p in group] for group in c_i]              # List of param_group-shaped control variates
        self.cr = c_glob           # List of param_group-shaped global control variates
        self.update_times = update_times
        self.gamma = gamma

    def train(self, net):
        net.train()

        eta = self.args.lr
        K = self.update_times
        gamma = self.gamma


        w_i_0 = []

        optimizer_original = torch.optim.SGD(net.parameters(), lr=eta, momentum=self.args.momentum)
        optimizer_ = GAMMA(net.parameters(), optimizer_original, gamma, self.ci, self.cr, adaptive=False)

        for group_idx, group in enumerate(optimizer_original.param_groups):
            group_w = []
            for p_idx, p in enumerate(group['params']):
                p.requires_grad=True
                group_w.append(p.data.detach().clone())
            w_i_0.append(group_w)

        count = 0
        batch_loss = []

        for iter in range(100):
            for batch_idx, (images, labels) in enumerate(self.trainloader):

                count = count + 1
                if count > self.update_times:
                    break

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer_.paras = [images, labels, self.loss_func, net]
                loss = optimizer_.step()

                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)

                optimizer_original.step()

                with torch.no_grad():
                    batch_loss.append(loss.item())


        c_i_delta = []
        for group_idx, group in enumerate(optimizer_original.param_groups):
            group_delta = []
            for p_idx, p in enumerate(group['params']):
                p.requires_grad = True
                if p.grad is not None:
                    delta = (w_i_0[group_idx][p_idx] - p.data) / (eta * K) - self.cr[group_idx][p_idx]
                    group_delta.append(delta)
                else:
                    group_delta.append(None)
            c_i_delta.append(group_delta)


        for group_idx in range(len(self.ci)):
            for p_idx in range(len(self.ci[group_idx])):
                self.ci[group_idx][p_idx] += c_i_delta[group_idx][p_idx]

        c_i_delta_cpu = [[p.cpu() if p is not None else None for p in group] for group in c_i_delta]
        c_i_cpu = [[p.cpu() if p is not None else None for p in group] for group in self.ci]

        return net.state_dict(), sum(batch_loss) / len(batch_loss), c_i_cpu, c_i_delta_cpu


class fedsmoo_LocalUpdate(object):
    def __init__(self, args=None, dataset=None, idxs=None, update_times=None, iteration=None, s_global=None, lam_i=None, mu_i=None, r=0.1, beta=10, lr=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        g = torch.Generator()
        g.manual_seed(args.seed + iteration)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True, generator=g)
        self.update_times = update_times
        self.mu_i = [[p.to(args.device, dtype=torch.float32) if p is not None else None for p in group] for group in mu_i]
        self.s_global = [[p.to(args.device, dtype=torch.float32) if p is not None else None for p in group] for group in s_global]
        self.lam_i = [[p.to(args.device, dtype=torch.float32) if p is not None else None for p in group] for group in lam_i]
        self.r = r
        self.beta = beta
        self.lr = lr

    def train(self, net):
        net.train()


        w_i_0 = []


        optimizer_original = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        for group_idx, group in enumerate(optimizer_original.param_groups):
            group_w = []
            for p_idx, p in enumerate(group['params']):
                p.requires_grad=True
                group_w.append(p.data.detach().clone())
            w_i_0.append(group_w)

        optimizer_ = SMOO(net.parameters(), optimizer_original, self.s_global, self.mu_i, self.lam_i, w_i_0, self.r, self.beta)

        batch_loss = []
        count = 0
        for epoch in range(100):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                count += 1
                if count > self.update_times:
                    break

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer_.paras = [images, labels, self.loss_func, net]
                loss = optimizer_.step()

                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)

                optimizer_original.step()

                with torch.no_grad():
                    batch_loss.append(loss.item())
        s_i_tilde = []
        s_i_hat = optimizer_.s_i_hat
        mu_i = optimizer_.mu_i

        for group_idx, group in enumerate(optimizer_original.param_groups):
            group_s_i = []
            for p_idx, p in enumerate(group['params']):
                p.requires_grad = True
                if p.grad is not None:
                    s_i = mu_i[group_idx][p_idx] - s_i_hat[group_idx][p_idx]
                    group_s_i.append(s_i)
                else:
                    group_s_i.append(None)
            s_i_tilde.append(group_s_i)

        lam_i_new = []

        for group_idx, group in enumerate(optimizer_original.param_groups):
            group_lam_i = []
            for p_idx, p in enumerate(group['params']):
                p.requires_grad = True
                if p.grad is not None:
                    new_lam_i = self.lam_i[group_idx][p_idx] - (p.data - w_i_0[group_idx][p_idx]) / self.beta
                    group_lam_i.append(new_lam_i)
                else:
                    group_lam_i.append(None)
            lam_i_new.append(group_lam_i)


        s_i_tilde_cpu = [[p.detach().to('cpu', dtype=torch.float16) if p is not None else None for p in group] for group in s_i_tilde]
        lam_i_cpu = [[p.detach().to('cpu', dtype=torch.float16) if p is not None else None for p in group] for group in lam_i_new]
        mu_i_cpu = [[p.detach().to('cpu', dtype=torch.float16) if p is not None else None for p in group] for group in mu_i]

        return net.state_dict(), sum(batch_loss) / len(batch_loss), s_i_tilde_cpu, lam_i_cpu, mu_i_cpu


class feddyn_LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, update_times=None, c_i=None, c_glob=None, iteration=None, mu=0.02):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        g = torch.Generator()
        g.manual_seed(args.seed + iteration)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True,
                                    generator=g)
        self.update_times = update_times
        self.c_i = [[p.to(args.device) if p is not None else None for p in group] for group in c_i]
        self.c_glob = c_glob
        self.mu = mu
    def train(self, net):

        c_i_new = copy.deepcopy(self.c_i)

        net.train()
        lr_ = self.args.lr

        # -------------
        optimizer_ = torch.optim.SGD(net.parameters(), lr=lr_, momentum=self.args.momentum)


        w_glob = []
        for group_idx, group in enumerate(optimizer_.param_groups):
            group_w = []
            for idx, p in enumerate(group['params']):
                group_w.append(p.data.detach().clone())
            w_glob.append(group_w)

        net.zero_grad()

        batch_loss = []
        count = 0
        for iter in range(100):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                count += 1
                if count > self.update_times:
                    break

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                for group_idx, group in enumerate(optimizer_.param_groups):
                    for idx, p in enumerate(group["params"]):
                        p.requires_grad = True
                        if p.grad is not None:
                            temp = p.grad.clone() - self.c_i[group_idx][idx].clone() - self.mu * (w_glob[group_idx][idx].clone() - p.data.clone())
                            p.grad = temp

                optimizer_.step()
                batch_loss.append(loss.item())

        c_i_delta = []
        for group_idx, group in enumerate(optimizer_.param_groups):
            temp_grad = []
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True

                temp = self.mu * (w_glob[group_idx][idx] - p.data.clone())
                c_i_new[group_idx][idx] += temp
                temp_grad.append(temp)

            c_i_delta.append(temp_grad)

        c_i_delta_cpu = [[p.cpu() if p is not None else None for p in group] for group in c_i_delta]
        c_i_new_cpu = [[p.cpu() if p is not None else None for p in group] for group in c_i_new]

        return net.state_dict(), sum(batch_loss) / len(batch_loss), c_i_new_cpu, c_i_delta_cpu


class fedcm_LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, update_times=None, delta=None, iteration=None, alpha=0.1):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        g = torch.Generator()
        g.manual_seed(args.seed + iteration)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True,
                                    generator=g)
        self.update_times = update_times
        self.delta = delta
        self.alpha = alpha
    def train(self, net):

        delta_i = copy.deepcopy(self.delta)

        net.train()
        lr_ = self.args.lr

        # -------------
        optimizer_ = torch.optim.SGD(net.parameters(), lr=lr_, momentum=self.args.momentum)


        w_glob = []
        for group_idx, group in enumerate(optimizer_.param_groups):
            group_w = []
            for idx, p in enumerate(group['params']):
                group_w.append(p.data.detach().clone())
            w_glob.append(group_w)

        net.zero_grad()

        batch_loss = []
        count = 0
        for iter in range(100):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                count += 1
                if count > self.update_times:
                    break

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                for group_idx, group in enumerate(optimizer_.param_groups):
                    for idx, p in enumerate(group["params"]):
                        p.requires_grad = True
                        if p.grad is not None:
                            temp = self.alpha * p.grad.clone() + (1-self.alpha) * self.delta[group_idx][idx]
                            p.grad = temp

                optimizer_.step()
                batch_loss.append(loss.item())

        return net.state_dict(), sum(batch_loss) / len(batch_loss)


class fedlesam_LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, update_times=None, iteration=None, wi_old=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        g = torch.Generator()
        g.manual_seed(args.seed + iteration)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True,
                                    generator=g)
        self.update_times = update_times
        self.wi_old = wi_old

    def train(self, net):

        net.train()

        lr_ = self.args.lr
        optimizer_original = torch.optim.SGD(net.parameters(), lr=lr_, momentum=self.args.momentum)


        w_glob = []
        w_i_old = []
        for group_idx, group in enumerate(optimizer_original.param_groups):
            group_w = []
            group_w_i = []
            for idx, p in enumerate(group['params']):
                group_w.append(p.data.detach().clone())
                group_w_i.append(self.wi_old[group_idx][idx].to(self.args.device))
            w_glob.append(group_w)
            w_i_old.append(group_w_i)

        optimizer_ = LESAM(net.parameters(), optimizer_original, 0.01, w_glob, w_i_old, adaptive=False)

        count = 0

        batch_loss = []

        while count < self.update_times:
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if count >= self.update_times:
                    break
                count = count + 1

                images, labels = images.to(self.args.device), labels.to(self.args.device)

                optimizer_.paras = [images, labels, self.loss_func, net]

                loss = optimizer_.step()

                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)
                optimizer_original.step()

                with torch.no_grad():
                    batch_loss.append(loss.item())

        return net.state_dict(), sum(batch_loss) / len(batch_loss)


class fedacg_LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, update_times=None, iteration=None, anchor_state=None, lr=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        g = torch.Generator()
        g.manual_seed(args.seed + iteration)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, generator=g)
        self.update_times = update_times
        self.anchor_state = anchor_state
        self.lr = args.lr if lr is None else lr
        self.beta = args.fedacg_beta

    def train(self, net):
        if self.anchor_state is not None:
            net.load_state_dict(self.anchor_state)
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        anchor_params = [p.detach().clone() for p in net.parameters()]

        count = 0
        batch_loss = []
        while count < self.update_times:
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if count >= self.update_times:
                    break
                count += 1

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                logits = net(images)
                loss = self.loss_func(logits, labels)

                reg = 0.0
                for p, pa in zip(net.parameters(), anchor_params):
                    reg += torch.sum((p - pa) ** 2)
                loss = loss + 0.5 * self.beta * reg

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

        return net.state_dict(), sum(batch_loss) / len(batch_loss)
