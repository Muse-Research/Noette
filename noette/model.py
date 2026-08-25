import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from .config import NoetteConfig
from .modules import RMSNorm, CausalSelfAttention, SwiGLU, LatentAttractorMomentum, EpistemicHaltingHead

class NoetteRecurrentBlock(nn.Module):
    """
    A single universal Transformer block cycled recurrently during latent pondering.
    """
    def __init__(self, config: NoetteConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN Self Attention with residual
        x = x + self.attn(self.attn_norm(x), mask=mask)
        # Pre-LN SwiGLU with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x

class NoetteLM(nn.Module):
    """
    Noette: Recurrent Latent Ponder-Loop Small Language Model.
    Performs zero-token multi-step reasoning in continuous latent space.
    """
    def __init__(self, config: NoetteConfig):
        super().__init__()
        self.config = config
        
        # Token Embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Recurrent Core Blocks
        self.recurrent_blocks = nn.ModuleList([
            NoetteRecurrentBlock(config) for _ in range(config.num_recurrent_blocks)
        ])
        
        # Latent Attractor Momentum Stabilization
        if config.use_attractor_momentum:
            self.attractor = LatentAttractorMomentum(config.d_model, config.momentum_decay)
        else:
            self.attractor = None
            
        # Epistemic Halting Head (p_t)
        self.halting_head = EpistemicHaltingHead(config.d_model)
        
        # Final Norm and Language Model Prediction Head
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying for extreme parameter efficiency
        self.lm_head.weight = self.tok_embeddings.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
        return mask.unsqueeze(0).unsqueeze(0) # [1, 1, S, S]

    def forward(
        self,
        input_ids: torch.Tensor,
        max_steps: Optional[int] = None,
        return_ponder_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adaptive latent pondering.
        During training: computes complete distribution over 1..K steps.
        During inference: supports dynamic early exit based on cumulative halting probability.
        """
        B, S = input_ids.shape
        device = input_ids.device
        causal_mask = self._create_causal_mask(S, device)
        K = max_steps or self.config.max_ponder_steps
        
        # Step 0: Initial embedding projection
        h = self.drop(self.tok_embeddings(input_ids)) # [B, S, D]
        h_prev = h
        velocity = torch.zeros_like(h)
        
        step_logits: List[torch.Tensor] = []
        step_p: List[torch.Tensor] = []          # Raw halting probs p_t
        step_q: List[torch.Tensor] = []          # Unnormalized/Normalized halting weights q_t
        step_states: List[torch.Tensor] = []
        
        unhalted_prob = torch.ones((B, S, 1), device=device) # Track cumulative remaining prob
        cumulative_q = torch.zeros((B, S, 1), device=device)
        ponder_steps_taken = torch.ones((B, S), device=device)
        
        for t in range(1, K + 1):
            # 1. Recurrent Core Transformation
            h_in = h
            for block in self.recurrent_blocks:
                h_in = block(h_in, mask=causal_mask)
            
            # 2. Attractor Momentum Stabilization
            if self.attractor is not None:
                h, velocity = self.attractor(h_in, h_prev, velocity)
            else:
                h = h_in
            h_prev = h
            
            # 3. Halting probability at step t
            p_t = self.halting_head(h) # [B, S, 1]
            
            # If final step K, force remaining probability to halt
            if t == K:
                p_t = torch.ones_like(p_t)
                
            q_t = unhalted_prob * p_t # Actual halting probability mass assigned to step t
            unhalted_prob = unhalted_prob * (1.0 - p_t + 1e-7)
            cumulative_q = cumulative_q + q_t
            
            # 4. Language Model output projection for step t
            normed_h = self.final_norm(h)
            logits_t = self.lm_head(normed_h) # [B, S, V]
            
            step_logits.append(logits_t)
            step_p.append(p_t)
            step_q.append(q_t)
            step_states.append(h)
            
            # In inference evaluation mode, check for early convergence
            if not self.training and (cumulative_q >= self.config.halting_threshold).all():
                ponder_steps_taken = torch.clamp(ponder_steps_taken, min=1, max=t)
                break

        # Stack over step dimension: [K, B, S, ...]
        all_logits = torch.stack(step_logits, dim=0) # [T, B, S, V]
        all_q = torch.stack(step_q, dim=0)           # [T, B, S, 1]
        all_p = torch.stack(step_p, dim=0)           # [T, B, S, 1]
        
        # Expected prediction weighted across all ponder steps:
        # y_pred = \sum_t q_t * logits_t
        expected_logits = (all_logits * all_q).sum(dim=0) # [B, S, V]
        
        # Expected ponder depth per token: \sum_t t * q_t
        step_indices = torch.arange(1, all_q.shape[0] + 1, device=device, dtype=torch.float32).view(-1, 1, 1, 1)
        expected_steps = (step_indices * all_q).sum(dim=0).squeeze(-1) # [B, S]
        
        out = {
            "logits": expected_logits,
            "step_logits": all_logits,
            "step_p": all_p,
            "step_q": all_q,
            "expected_steps": expected_steps,
            "final_state": h
        }
        
        if return_ponder_details:
            out["step_states"] = torch.stack(step_states, dim=0)
            
        return out


class BaselineTransformerLM(nn.Module):
    """
    Standard fixed-depth multi-layer Transformer for fair baseline comparison.
    Configured to match parameter count and hidden dimensions.
    """
    def __init__(self, config: NoetteConfig, num_layers: int = 4):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            NoetteRecurrentBlock(config) for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, S = input_ids.shape
        causal_mask = self._create_causal_mask(S, input_ids.device)
        h = self.drop(self.tok_embeddings(input_ids))
        for block in self.blocks:
            h = block(h, mask=causal_mask)
        normed = self.final_norm(h)
        logits = self.lm_head(normed)
        return {"logits": logits}
