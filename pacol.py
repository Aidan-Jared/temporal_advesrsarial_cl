import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class PACOL:
    def __init__(
            self,
            eps: float,
            steps: int,
            step_size: int,
            iterations: int,
            dist_metric: str,
            loss_fn: nn.Module,
            opt: torch.optim.Optimizer,
            batch_size: int,
    ):
        self.eps = eps
        self.S = steps
        self.alpha = step_size
        self.K = iterations
        self.dist_metric = dist_metric
        self.loss_fn = loss_fn
        self.opt = opt
        self.batch_size = batch_size
        

    def _dist(self, label_grad: torch.Tensor, x_grad: torch.Tensor):

        label_grad = label_grad.detach()
        if self.dist_metric == "cosine":
            H = F.cosine_similarity(label_grad.unsqueeze(0), x_grad.unsqueeze(0))
        else:
            H = torch.norm(label_grad - x_grad)
        return H
    
    def _grads(
            self,
            X: torch.Tensor,
            y: torch.Tensor,
            model: nn.Module
    ):      
        model.zero_grad()  
        loss = self.loss_fn(model(X), y)
        loss.backward(retain_graph = True)

        return torch.cat([p.grad.flatten() for p in model.parameters()])


    
    def __call__(
            self,
            flip_data: Dataset,
            adv_data: Dataset,
            model: nn.Module,
    ):

        flip_data = DataLoader(
            flip_data,
            batch_size=self.batch_size,
            shuffle=True,
            )
    
        for _ in range(self.K):
            X, y = next(iter(flip_data))
            grad_flip = self._grads(X, y)

            for idx in range(self.S):
                start = idx * self.batch_size
                end = start + self.batch_size
                X, y = adv_data[start:end]
                X_orig = X.clone().detach()
                X = X.detach().requires_grad_(True)


                grad_adv = self._grads(X, y)

                H = self._dist(grad_flip, grad_adv)
                H.backward()

                with torch.no_grad():
                    X = torch.clamp(
                        (X + self.alpha * X.grad.sign()), 
                        min=X_orig - self.eps,
                        max=X_orig+ self.eps
                    )
                    X = torch.clamp(X, 0.,1.)
                adv_data.X[start:end] = X
            
            model.zero_grad()
            X, y = adv_data[0:self.batch_size]
            loss = self.loss_fn(model(X), y)
            loss.backward()
            self.opt.step()
        return adv_data
                