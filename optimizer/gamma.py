import torch


class GAMMA(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, gamma, ci, cr, adaptive=False, **kwargs):
        assert gamma >= 0.0, f"Invalid perturbation rate, should be non-negative: {gamma}"
        self.max_norm = 10

        defaults = dict(gamma=gamma, adaptive=adaptive, **kwargs)
        super(GAMMA, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["gamma"] = gamma
            # group["adaptive"] = adaptive
        self.paras = None
        self.ci = ci
        self.cr = cr

    @torch.no_grad()
    def first_step(self):
        # first order sum
        grad_norm_sq = 0
        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                if p.grad is None:
                    continue
                grad_norm_sq += p.grad.norm(p=2).pow(2)
        grad_norm = grad_norm_sq.sqrt()

        for group in self.param_groups:
            scale = group["gamma"]
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                if p.grad is None:
                    continue
                # original SAM
                # ASAM

                e_w = p.grad * torch.tensor(scale).to(p)
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

    def step(self):
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

        for group_idx, group in enumerate(self.param_groups):
            for idx, p in enumerate(group['params']):
                if p.grad is not None:
                    new_grad = p.grad.detach().clone() - self.ci[group_idx][idx].detach().clone() + self.cr[group_idx][idx].detach().clone()
                    p.grad = new_grad

        batch_loss = loss

        self.second_step()

        return batch_loss
