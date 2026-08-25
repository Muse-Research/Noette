import time
import torch
from torch.utils.data import DataLoader
from .config import NoetteConfig
from .model import NoetteLM, BaselineTransformerLM
from .loss import NoettePonderLoss
from .data import MultiHopDeductionDataset, AlgorithmicCarryDataset, collate_reasoning_batch
from .trainer import train_noette_epoch, train_baseline_epoch, evaluate_accuracy

def run_comparative_benchmark(epochs: int = 15, batch_size: int = 64, num_samples: int = 2500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"   NOETTE vs. STANDARD SLM COMPARATIVE BENCHMARK")
    print(f"   Hardware Target: {device}")
    print(f"=======================================================\n")
    
    # 1. Prepare Datasets
    print("[1/4] Generating Multi-Hop Symbolic Deduction Dataset (2 to 5 Hops)...")
    train_dataset = MultiHopDeductionDataset(num_samples=num_samples, min_hops=2, max_hops=5, seed=42)
    val_dataset = MultiHopDeductionDataset(num_samples=600, min_hops=2, max_hops=5, seed=1337)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_reasoning_batch(b, pad_token_id=train_dataset.tokenizer.pad_token_id)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_reasoning_batch(b, pad_token_id=val_dataset.tokenizer.pad_token_id)
    )
    
    # 2. Configure Models
    vocab_size = train_dataset.tokenizer.vocab_size
    config = NoetteConfig(
        vocab_size=vocab_size,
        d_model=96,
        n_heads=3,
        d_ff=384,
        max_ponder_steps=6,
        ponder_lambda=0.3,
        ponder_beta=0.01,
        halting_threshold=0.95
    )
    
    noette_model = NoetteLM(config).to(device)
    baseline_model = BaselineTransformerLM(config, num_layers=2).to(device)
    
    noette_params = sum(p.numel() for p in noette_model.parameters())
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    
    print(f"[2/4] Model Architecture Specifications:")
    print(f"  - Noette Parameters:    {noette_params:,} (1 Recurrent Latent Ponder Core, Dynamic Depth 1..{config.max_ponder_steps})")
    print(f"  - Baseline Parameters:  {baseline_params:,} (2-Layer Fixed-Depth Transformer)")
    print(f"\n[3/4] Training Models on Zero-Token Reasoning Task...")
    
    # Optimizers & Loss
    noette_opt = torch.optim.AdamW(noette_model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    baseline_opt = torch.optim.AdamW(baseline_model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    loss_fn = NoettePonderLoss(lambda_prior=config.ponder_lambda, beta_reg=config.ponder_beta)
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        noette_metrics = train_noette_epoch(noette_model, train_loader, noette_opt, loss_fn, device)
        baseline_metrics = train_baseline_epoch(baseline_model, train_loader, baseline_opt, device)
        dt = time.time() - t0
        
        if epoch % 3 == 0 or epoch == epochs:
            noette_eval = evaluate_accuracy(noette_model, val_loader, train_dataset.tokenizer, device, is_noette=True)
            base_eval = evaluate_accuracy(baseline_model, val_loader, train_dataset.tokenizer, device, is_noette=False)
            print(f"Epoch {epoch:02d}/{epochs:02d} [{dt:.1f}s] | "
                  f"Noette Loss: {noette_metrics['task_loss']:.3f}, Acc: {noette_eval['accuracy']:.1f}%, Ponder Depth: {noette_eval['mean_ponder_depth']:.2f} | "
                  f"Baseline Loss: {baseline_metrics['total_loss']:.3f}, Acc: {base_eval['accuracy']:.1f}%")
                  
    print(f"\n[4/4] Evaluating Detailed Performance by Reasoning Difficulty (Hop Count)...")
    print(f"{'Hops':<8}{'Noette Accuracy':<20}{'Noette Ponder Steps':<25}{'Baseline Accuracy':<20}")
    print("-" * 75)
    
    for h in range(2, 6):
        hop_dataset = MultiHopDeductionDataset(num_samples=300, min_hops=h, max_hops=h, seed=100 + h)
        hop_loader = DataLoader(
            hop_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_reasoning_batch(b, pad_token_id=hop_dataset.tokenizer.pad_token_id)
        )
        n_eval = evaluate_accuracy(noette_model, hop_loader, hop_dataset.tokenizer, device, is_noette=True)
        b_eval = evaluate_accuracy(baseline_model, hop_loader, hop_dataset.tokenizer, device, is_noette=False)
        print(f"{h:<8}{n_eval['accuracy']:>6.1f}%{'':<13}{n_eval['mean_ponder_depth']:>6.2f} steps{'':<12}{b_eval['accuracy']:>6.1f}%")
        
    print("\nBenchmark Complete!")
    return noette_model, baseline_model

if __name__ == "__main__":
    run_comparative_benchmark(epochs=12, batch_size=64, num_samples=2000)
