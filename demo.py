import time
import torch
from torch.utils.data import DataLoader
from noette.config import NoetteConfig
from noette.model import NoetteLM
from noette.loss import NoettePonderLoss
from noette.data import SimpleCharTokenizer, MultiHopDeductionDataset, collate_reasoning_batch
from noette.trainer import train_noette_epoch

def format_ponder_bar(depth: float, max_depth: int = 6) -> str:
    # ASCII intensity bar
    filled = int(round(depth))
    empty = max(0, max_depth - filled)
    bar = "#" * filled + "-" * empty
    return f"[{bar}] {depth:.2f}x"

def visualize_inference(model: NoetteLM, prompt: str, tokenizer: SimpleCharTokenizer, device: torch.device):
    model.eval()
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    
    with torch.no_grad():
        out = model(input_ids, return_ponder_details=True)
        logits = out["logits"] # [1, S, V]
        expected_steps = out["expected_steps"][0] # [S]
        step_q = out["step_q"][:, 0, :, 0] # [K, S]
        
    next_token_id = torch.argmax(logits[0, -1]).item()
    predicted_char = tokenizer.decode([next_token_id])
    
    print("\n" + "=" * 80)
    print(f" INPUT PROMPT:  {prompt}")
    print(f" PREDICTED ANS: {predicted_char}")
    print("=" * 80)
    print(f"{'Pos':<5}{'Token':<10}{'Latent Ponder Depth':<25}{'Ponder Prob Distribution (q_1 .. q_K)'}")
    print("-" * 80)
    
    chars = [tokenizer.id_to_char.get(i, "?") for i in encoded]
    for i, ch in enumerate(chars):
        depth = expected_steps[i].item()
        bar = format_ponder_bar(depth, model.config.max_ponder_steps)
        q_dist = " ".join([f"{step_q[t, i].item():.2f}" for t in range(step_q.shape[0])])
        display_ch = repr(ch)[1:-1] if ch != " " else "<sp>"
        print(f"{i:<5}{display_ch:<10}{bar:<25}{q_dist}")
        
    print("-" * 80)
    print(f">> Notice: Static syntax tokens use ~1.00 step; Deduction tokens scale latent pondering depth automatically!\n")

def run_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "#" * 80)
    print("   NOETTE: ZERO-TOKEN LATENT PONDER-LOOP ARCHITECTURE DEMO")
    print("#" * 80)
    
    # 1. Dataset & Tokenizer
    dataset = MultiHopDeductionDataset(num_samples=1800, min_hops=2, max_hops=4, seed=42)
    tokenizer = dataset.tokenizer
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=lambda b: collate_reasoning_batch(b, pad_token_id=tokenizer.pad_token_id)
    )
    
    # 2. Config & Model
    config = NoetteConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=96,
        n_heads=3,
        d_ff=384,
        max_ponder_steps=6,
        ponder_lambda=0.25,
        ponder_beta=0.015,
        halting_threshold=0.95
    )
    model = NoetteLM(config).to(device)
    loss_fn = NoettePonderLoss(lambda_prior=config.ponder_lambda, beta_reg=config.ponder_beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n[+] Initialized Noette (Parameters: {param_count:,} | Max Latent Ponder Depth: {config.max_ponder_steps})")
    print("[+] Training model to converge internal thought attractors (6 fast epochs)...")
    
    for epoch in range(1, 7):
        t0 = time.time()
        metrics = train_noette_epoch(model, loader, optimizer, loss_fn, device)
        print(f"    Epoch {epoch}/6 [{time.time()-t0:.1f}s] - Loss: {metrics['task_loss']:.4f} | Avg Latent Depth: {metrics['avg_ponder_depth']:.2f}")
        
    print("\n[+] Visualizing Internal Latent Thought Depth on Unseen Reasoning Problems:")
    
    test_cases = [
        "[X=5, Y=X] Q:Y?=",
        "[A=9, B=A, C=B] Q:C?=",
        "[M=4, N=M, P=N, K=P] Q:K?=",
        "[Z=7, W=Z, V=W, U=V, T=U] Q:T?="
    ]
    
    for prompt in test_cases:
        visualize_inference(model, prompt, tokenizer, device)

if __name__ == "__main__":
    run_demo()
