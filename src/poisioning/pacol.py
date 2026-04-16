# needs to be re-worked, ignore for time being
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class PACOL:
    def __init__(
        self,
        eps: float,
        steps: int,
        step_size: float,
        iterations: int,
        dist_metric: str,
        model: nn.Module,
        loss_fn: nn.Module,
        opt: torch.optim.Optimizer,
        batch_size: int,
    ):
        self.eps: float = eps
        self.S: int = steps
        self.alpha: float = step_size
        self.K: int = iterations
        self.dist_metric: str = dist_metric
        self.model: nn.Module = model
        self.loss_fn: nn.Module = loss_fn
        self.opt: torch.optim.Optimizer = opt
        self.batch_size: int = batch_size

    def _dist(self, label_grad: torch.Tensor, x_grad: torch.Tensor):

        label_grad = label_grad.detach()
        if self.dist_metric == "cosine":
            H = F.cosine_similarity(label_grad.unsqueeze(0), x_grad.unsqueeze(0))
        else:
            H = torch.norm(label_grad - x_grad)
        return H

    def _grads(self, X: torch.Tensor, y: torch.Tensor):
        self.model.zero_grad()
        loss = self.loss_fn(self.model(X), y)

        grads = torch.autograd.grad(
            outputs=loss, inputs=[p for p in self.model.parameters()], create_graph=True
        )

        return torch.cat([g.flatten() for g in grads])

    def __call__(
        self,
        flip_data: Dataset[torch.Tensor],
        adv_data: list[torch.Tensor],
    ):

        collate_fn = getattr(flip_data, "collate_fn", None)

        flip_data = DataLoader(
            flip_data, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn
        )

        device = next(self.model.parameters()).device

        for i, (X_f, _, y_f) in enumerate(flip_data):
            if i > self.K:
                break
            X_f = X_f.to(device)
            y_f = y_f.to(device)
            grad_flip = self._grads(X_f, y_f)

            for idx in range(self.S):
                start: int = idx * self.batch_size
                end: int = start + self.batch_size
                X = adv_data[0][start:end].to(device)
                y = adv_data[1][start:end].to(device)
                X_orig = X.clone().detach()
                X = X.detach().requires_grad_(True)

                grad_adv: torch.Tensor = self._grads(X, y)

                H: torch.Tensor = self._dist(grad_flip, grad_adv)
                H.backward()

                with torch.no_grad():
                    X: torch.Tensor = torch.clamp(
                        (X + self.alpha * X.grad.sign()),
                        min=X_orig - self.eps,
                        max=X_orig + self.eps,
                    )
                    # X: torch.Tensor = torch.clamp(X, 0.0, 1.0)

                adv_data[0][start:end] = X.clone().detach()
                adv_data[1][start:end] = y.clone().detach()

            self.model.zero_grad()
            X = adv_data[0][0 : self.batch_size].to(device)
            y = adv_data[1][0 : self.batch_size].to(device)
            loss = self.loss_fn(self.model(X), y)
            loss.backward()
            self.opt.step()
        return adv_data[0], adv_data[1]
