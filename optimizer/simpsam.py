"""Simplified variance-suppressed SAM optimizer."""

import torch


class SIMPSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, adaptive=False, gamma=None, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SIMPSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.g_update=None
        for group in self.param_groups:
            group["rho"] = rho
        self.paras = None

        self.gamma = gamma


    @torch.no_grad()
    def first_step(self,g_update):
        updated_grad =[]
        for group_idx, group in enumerate(self.param_groups):
            group_grads = []
            for idx,p in enumerate(group["params"]):

                p.requires_grad = True
                if p.grad is not None:

                    gradient = p.grad.detach().clone()*self.gamma + g_update[group_idx][idx]*(1-self.gamma)
                    group_grads.append(gradient)
                else:
                    group_grads.append(None)
            updated_grad.append(group_grads)
        grad_norm_sq = 0
        for group in updated_grad:
            for ele in group:
                if ele != None:
                    grad_norm_sq += ele.norm(p=2).pow(2)

        grad_norm = grad_norm_sq.sqrt()
        for group_idx, group in enumerate(self.param_groups):
            scale = group["rho"] / (grad_norm + 1e-7)
            for idx,p in enumerate(group["params"]):
                p.requires_grad = True
                if p.grad == None:
                    continue
                e_w = ( updated_grad[group_idx][idx] ) * scale.to(p)

                # climb to the local maximum "w + e(w)"
                p.add_(e_w * 1)
                self.state[p]["e_w"] = e_w

        return updated_grad


    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]:
                    continue
                # go back to "w" from "w + e(w)"
                p.sub_(self.state[p]["e_w"])
                self.state[p]["e_w"] = 0

    def step(self,g_update=None):

        inputs, labels, loss_func, model = self.paras
        predictions = model(inputs)
        loss_1 = loss_func(predictions, labels)
        self.zero_grad()
        loss_1.backward()

        # Use first_step to acquire the adjusted model para and the adjusted gradient
        h_adj = self.first_step(g_update)

        # compute the batch_gradient again with the adjusted model para
        inputs, labels, loss_func, model = self.paras
        predictions = model(inputs)
        loss_2 = loss_func(predictions, labels)
        self.zero_grad()
        loss_2.backward()
        batch_loss = loss_2

        self.second_step()


        return batch_loss, h_adj
