from dataclasses import dataclass

@dataclass
class NoetteConfig:
    vocab_size: int = 128
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    max_seq_len: int = 256
    
    # Recurrent Latent Pondering Parameters
    num_recurrent_blocks: int = 1     # Number of unique shared recurrent Transformer blocks
    max_ponder_steps: int = 8         # Maximum intra-token ponder loops (K)
    min_ponder_steps: int = 1         # Minimum ponder loops
    ponder_lambda: float = 0.25       # Geometric prior parameter (expected compute cost)
    ponder_beta: float = 0.05         # Regularization weight for KL divergence against prior
    halting_threshold: float = 0.95   # Inference early halting confidence threshold tau
    
    # Attractor momentum & stability
    use_attractor_momentum: bool = True
    momentum_decay: float = 0.85
    dropout: float = 0.05
