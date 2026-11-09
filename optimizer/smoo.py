import torch


class SMOO(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, s_global, mu_i, lam_i, w_i_0, r=0.1, beta=0.1):
        super(SMOO, self).__init__(params, base_optimizer.defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.paras = None  # [images, labels, loss_func, net]
        self.s_global = s_global
        self.mu_i = mu_i
        self.lam_i = lam_i
        self.w_i_0 = w_i_0
        self.r = r
        self.beta = beta
        self.s_i_hat = [[torch.zeros_like(p) if p is not None else None for p in group] for group in s_global]
        for group in self.param_groups:
            group["r"] = self.r

    @torch.no_grad()
    def first_step(self):
        # first order sum
        grad_norm_sq = 0
        for group_idx, group in enumerate(self.param_groups):
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                if p.grad is None:
                    continue
                new_grad = p.grad.detach().clone() - self.mu_i[group_idx][idx] - self.s_global[group_idx][idx]
                p.grad = new_grad
                grad_norm_sq += p.grad.norm(p=2).pow(2)

        grad_norm = grad_norm_sq.sqrt()

        for group_idx, group in enumerate(self.param_groups):
            scale = group["r"] / (grad_norm + 1e-7)
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                if p.grad is None:
                    continue


                new_s_i_hat = scale * p.grad.detach().clone()
                self.s_i_hat[group_idx][idx] = new_s_i_hat


                self.mu_i[group_idx][idx] += new_s_i_hat - self.s_global[group_idx][idx]

                e_w = p.grad * scale.to(p)
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

        model.train()
        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()

        self.first_step()

        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()
        self.second_step()

        for group_idx, group in enumerate(self.param_groups):
            for idx, p in enumerate(group['params']):
                if p.grad is not None:
                    new_grad = p.grad.detach().clone() - self.lam_i[group_idx][idx] + (p.data.detach().clone() - self.w_i_0[group_idx][idx]) / self.beta
                    p.grad = new_grad

        batch_loss = loss
        self.paras = None
        return batch_loss
