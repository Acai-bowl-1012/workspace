import torch
import torch.optim as optim


class SGD(optim.Optimizer):
    def __init__(self, params, lr: float):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                p.sub_(grad, alpha=lr)

        return loss