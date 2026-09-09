import torch
import torch.optim as optim


class norm_clipping_SGD(optim.Optimizer):
    def __init__(self, params, lr: float, C: float):
        defaults = dict(lr=lr, C=C)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        for group in self.param_groups:
            lr = group["lr"]
            C = group["C"]

            total_norm_sq = torch.zeros((), device=group["params"][0].device)

            for p in group["params"]:
                if p.grad is not None:
                    total_norm_sq += torch.sum(p.grad.detach() ** 2)

            total_norm = torch.sqrt(total_norm_sq)

            clip_coef = min(1.0, C / (total_norm + 1e-12))

            for p in group["params"]:
                if p.grad is not None:
                    p.sub_(p.grad, alpha=lr * clip_coef)

        return loss