import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class NoettePonderLoss(nn.Module):
    """
    Probabilistic Ponder Loss for Noette.
    Combines expectation-weighted cross entropy over latent ponder steps
    with a Kullback-Leibler divergence penalty against a Geometric compute budget prior.
    """
    def __init__(self, lambda_prior: float = 0.3, beta_reg: float = 0.01, ignore_index: int = -100):
        super().__init__()
        self.lambda_prior = lambda_prior
        self.beta_reg = beta_reg
        self.ignore_index = ignore_index

    def _get_geometric_prior(self, max_steps: int, device: torch.device) -> torch.Tensor:
        """
        Constructs the prior distribution p*(t) = lambda * (1 - lambda)^(t-1)
        """
        steps = torch.arange(1, max_steps + 1, device=device, dtype=torch.float32)
        prior = self.lambda_prior * ((1.0 - self.lambda_prior) ** (steps - 1))
        # Assign remainder of distribution to the final step K so sum equals 1.0
        prior[-1] = 1.0 - prior[:-1].sum()
        return prior.view(-1, 1, 1, 1) # [T, 1, 1, 1]

    def forward(
        self,
        step_logits: torch.Tensor,   # [T, B, S, V]
        step_q: torch.Tensor,        # [T, B, S, 1]
        targets: torch.Tensor        # [B, S]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        T, B, S, V = step_logits.shape
        device = step_logits.device
        
        # 1. Compute Cross Entropy for each ponder step t
        targets_flat = targets.view(-1) # [B * S]
        step_losses = []
        for t in range(T):
            logits_t_flat = step_logits[t].view(-1, V) # [B*S, V]
            # Cross entropy with reduction='none'
            loss_t = F.cross_entropy(logits_t_flat, targets_flat, ignore_index=self.ignore_index, reduction='none')
            step_losses.append(loss_t.view(B, S))
            
        stacked_step_losses = torch.stack(step_losses, dim=0) # [T, B, S]
        
        # 2. Weighted Task Loss over all ponder steps
        # Ignore loss at masked positions
        mask = (targets != self.ignore_index).float() # [B, S]
        valid_tokens = mask.sum().clamp(min=1.0)
        
        # q_t is [T, B, S, 1] -> squeeze to [T, B, S]
        q_squeezed = step_q.squeeze(-1) # [T, B, S]
        weighted_step_loss = (q_squeezed * stacked_step_losses).sum(dim=0) # [B, S]
        task_loss = (weighted_step_loss * mask).sum() / valid_tokens
        
        # 3. Compute KL Divergence against Geometric Prior: KL(q || prior)
        prior = self._get_geometric_prior(T, device) # [T, 1, 1, 1]
        prior_squeezed = prior.squeeze(-1) # [T, 1, 1]
        
        eps = 1e-8
        kl_div = (q_squeezed * (torch.log(q_squeezed + eps) - torch.log(prior_squeezed + eps))).sum(dim=0) # [B, S]
        reg_loss = (kl_div * mask).sum() / valid_tokens
        
        # 4. Total Combined Loss
        total_loss = task_loss + (self.beta_reg * reg_loss)
        
        metrics = {
            "total_loss": total_loss.detach(),
            "task_loss": task_loss.detach(),
            "reg_loss": reg_loss.detach(),
            "avg_step": ((torch.arange(1, T + 1, device=device, dtype=torch.float32).view(-1, 1, 1) * q_squeezed).sum(dim=0) * mask).sum().detach() / valid_tokens
        }
        
        return total_loss, metrics
