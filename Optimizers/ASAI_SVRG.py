import torch
import torch.optim as optim
import math


class ASAI_SVRG(optim.Optimizer):
    def __init__(self, params, lr:float):
        defaults = dict(lr = lr,)
        super().__init__(params, defaults)


    def init_epoch(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                if "a" not in state:
                    state["a"] = torch.zeros_like(p, requires_grad=False)
                    state["hat_a"] = torch.zeros_like(p, requires_grad=False)
                    state["x"] = p.detach().clone()
                    state["hat_x"] = p.detach().clone()

                state["a"].zero_()
                state["x"].zero_()
                state["t"] = 0


    def calc_snapshot_grads(self, model, X, T, loss_func):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    state["w"] = torch.clone(p).detach()
                    p.copy_(state["hat_x"])

        model.zero_grad()
        Y = model(X)
        loss = loss_func(Y, T)
        loss.backward()

        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    state["snapshot_grad"] = torch.clone(p.grad.detach())
                    p.copy_(state["w"])


    def end_epoch(self):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]

                    state["hat_a"].copy_(state["a"])
                    state['hat_x'].copy_(state["x"])
                    p.copy_(state['x'])


    def calc_diff_norm(self, model, dataloader, loss_func):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_data = 0

        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    state["full_grad"] = torch.zeros_like(p)

                    if "hat_x" in state:
                        state["w"] = torch.clone(p.detach())
                        p.copy_(state["hat_x"])

        model.train()
        for X, T in dataloader:
            X, T = X.to(device), T.to(device)

            model.zero_grad()
            Y = model(X)
            loss = loss_func(Y, T) * len(X)
            loss.backward()

            total_data += len(X)

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad = p.grad.detach()
                    state = self.state[p]
                    state["full_grad"].add_(grad)

        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    state["full_grad"].div_(total_data)

                    if "current_p_temp" in state:
                        p.copy_(state["w"])
                        del state["w"]

        diff_norm_sum = 0.0
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]

                    if "a" in state and "full_grad" in state:
                        diff = state["a"] - state["full_grad"]
                        diff_norm_sum += torch.sum(diff ** 2).item()

        diff_norm = math.sqrt(diff_norm_sum)

        return diff_norm


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                state = self.state[p]
                a = state['a']
                hat_a = state["hat_a"]
                snapshot_grad = state["snapshot_grad"]
                x = state["x"]
                t = state["t"]

                x.mul_(t / (t+1)).add_(p, alpha = 1 / (t+1))
                a.mul_(t / (t+1)).add_(grad, alpha = 1 / (t+1))
                v = grad - snapshot_grad + hat_a
                p.sub_(v, alpha = lr)

                state["t"] += 1

        return loss