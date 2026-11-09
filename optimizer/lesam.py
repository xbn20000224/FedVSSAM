import torch

# Design of benchmark FedSAM
# Here, we default the value of beta (gamma) as a constant accross devices


class LESAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, w_glob, wi_old, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(LESAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            # group["adaptive"] = adaptive
        self.paras = None
        self.w_glob = w_glob
        self.wi_old = wi_old

    @torch.no_grad()
    def first_step(self):
        # first order sum
        grad_norm_sq = 0
        delta_w = []
        for group_idx, group in enumerate(self.param_groups):
            group_w = []
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                delta_w_new = self.wi_old[group_idx][idx] - self.w_glob[group_idx][idx]
                grad_norm_sq += delta_w_new.norm(p=2).pow(2)
                group_w.append(delta_w_new)
            delta_w.append(group_w)

        grad_norm = grad_norm_sq.sqrt()

        for group_idx, group in enumerate(self.param_groups):
            scale = group["rho"] / (grad_norm + 1e-7)
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                e_w = delta_w[group_idx][idx] * scale.to(p)
                # climb to the local maximum "w + e(w)"
                p.add_(e_w * 1)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]:
                    continue
                # go back to "w" from "w + e(w)"
                p.sub_(self.state[p]["e_w"])
                self.state[p]["e_w"] = 0

    def step(self, g_update=None):
        inputs, labels, loss_func, model = self.paras

        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()

        self.first_step()

        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        self.zero_grad()

        loss.backward()

        batch_loss = loss

        self.second_step()
        return batch_loss
