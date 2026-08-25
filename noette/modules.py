import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k shape: [B, H, S, D_h]
    # cos, sin shape: [S, D_h] -> unsqueeze to [1, 1, S, D_h]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(x, S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v) # [B, H, S, D_h]
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)

class LatentAttractorMomentum(nn.Module):
    """
    Stabilizes recurrent latent ponder steps by tracking continuous velocity
    towards stable attractor states in representation space.
    """
    def __init__(self, d_model: int, momentum_decay: float = 0.85):
        super().__init__()
        self.d_model = d_model
        self.momentum_decay = momentum_decay
        self.velocity_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, h_curr: torch.Tensor, h_prev: torch.Tensor, velocity: torch.Tensor):
        # Compute dynamic displacement
        raw_velocity = h_curr - h_prev
        gate = self.velocity_gate(torch.cat([h_curr, velocity], dim=-1))
        new_velocity = self.momentum_decay * velocity + (1.0 - self.momentum_decay) * (gate * raw_velocity)
        stabilized_h = h_curr + new_velocity
        return stabilized_h, new_velocity

class EpistemicHaltingHead(nn.Module):
    """
    Estimates the halting probability p_t at recurrent ponder step t,
    signaling convergence of the internal latent thought.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        # Returns halting probability p_t in range (0, 1) of shape [B, S, 1]
        return self.proj(self.norm(h_t))
