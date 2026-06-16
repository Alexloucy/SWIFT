import torch
import torch.nn as nn


class GradNorm:
    def __init__(self, loss_weights, num_tasks):
        self.loss_weights = loss_weights
        self.num_tasks = num_tasks

    def update_weights(self, grads):
        # norm_losses = torch.tensor(losses) / sum(losses)
        # norm_grads = [torch.norm(grad) for grad in grads]
        norm_grads = []
        for grad_set in grads:
            grad_norm = torch.norm(
                torch.stack(
                    [
                        torch.norm(g)
                        for g in grad_set
                        if g is not None  # Ensure None gradients are skipped
                    ]
                )
            )
            norm_grads.append(grad_norm)

        self.loss_weights = [1.0 / (1e-8 + norm_grad) for norm_grad in norm_grads]
        self.loss_weights = torch.tensor(self.loss_weights)
        self.loss_weights = self.loss_weights / self.loss_weights.sum()

    def get_weights(self):
        return self.loss_weights


class UncertaintyWeighting(nn.Module):
    def __init__(self):
        super(UncertaintyWeighting, self).__init__()
        self.log_sigma1 = nn.Parameter(torch.tensor(0.0))
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))

    def forward(self, loss_invariant, loss_specific):
        sigma1 = torch.exp(self.log_sigma1)
        sigma2 = torch.exp(self.log_sigma2)

        inv_spc_loss = (
            (1.0 / (sigma1**2)) * loss_invariant
            + ((1.0) / sigma2**2) * loss_specific
            + torch.log(sigma1 + sigma2)
        )
        return inv_spc_loss


class ParetoMTL(nn.Module):
    def __init__(self, num_tasks):
        super(ParetoMTL, self).__init__()
        self.num_tasks = num_tasks

    @staticmethod
    def _get_gradient(loss, params):
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return torch.cat([g.view(-1) for g in grads if g is not None])

    def calculate_pareto_grad(self, losses, params):
        grads = [self._get_gradient(loss, params) for loss in losses]

        G = torch.stack(grads)
        G_norm = torch.norm(G, dim=1, keepdim=True) + 1e-8
        G_normalized = G / G_norm

        weights = torch.ones(self.num_tasks).to(G.device) / self.num_tasks

        pareto_grad = G_normalized.t() @ weights

        return pareto_grad
