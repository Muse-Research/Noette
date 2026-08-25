import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
from .model import NoetteLM, BaselineTransformerLM
from .loss import NoettePonderLoss

def train_noette_epoch(
    model: NoetteLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: NoettePonderLoss,
    device: torch.device
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_reg_loss = 0.0
    total_avg_step = 0.0
    num_batches = 0
    
    for input_ids, targets, _ in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        out = model(input_ids)
        
        loss, metrics = loss_fn(
            step_logits=out["step_logits"],
            step_q=out["step_q"],
            targets=targets
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += metrics["total_loss"].item()
        total_task_loss += metrics["task_loss"].item()
        total_reg_loss += metrics["reg_loss"].item()
        total_avg_step += metrics["avg_step"].item()
        num_batches += 1
        
    return {
        "total_loss": total_loss / max(1, num_batches),
        "task_loss": total_task_loss / max(1, num_batches),
        "reg_loss": total_reg_loss / max(1, num_batches),
        "avg_ponder_depth": total_avg_step / max(1, num_batches)
    }

def train_baseline_epoch(
    model: BaselineTransformerLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for input_ids, targets, _ in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        out = model(input_ids)
        logits = out["logits"]
        
        # Standard Cross-Entropy
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.view(-1),
            ignore_index=-100
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return {"total_loss": total_loss / max(1, num_batches)}

def evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    is_noette: bool = True
) -> Dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    all_ponder_depths = []
    
    with torch.no_grad():
        for input_ids, targets, prompt_lens in dataloader:
            input_ids = input_ids.to(device)
            B, S = input_ids.shape
            
            if is_noette:
                out = model(input_ids)
                logits = out["logits"] # [B, S, V]
                expected_steps = out["expected_steps"] # [B, S]
            else:
                out = model(input_ids)
                logits = out["logits"]
                expected_steps = None
                
            predictions = torch.argmax(logits, dim=-1) # [B, S]
            
            for b in range(B):
                p_len = prompt_lens[b]
                # Target slice
                target_slice = targets[b, p_len - 1:]
                valid_mask = target_slice != -100
                if valid_mask.sum() == 0:
                    continue
                    
                pred_slice = predictions[b, p_len - 1:][valid_mask]
                true_slice = target_slice[valid_mask]
                
                is_correct = torch.equal(pred_slice, true_slice)
                if is_correct:
                    correct += 1
                total += 1
                
                if is_noette and expected_steps is not None:
                    # Average ponder depth on reasoning tokens
                    depth_on_ans = expected_steps[b, p_len - 1:][valid_mask].mean().item()
                    all_ponder_depths.append(depth_on_ans)
                    
    acc = (correct / total) * 100.0 if total > 0 else 0.0
    mean_ponder = sum(all_ponder_depths) / len(all_ponder_depths) if all_ponder_depths else 1.0
    return {"accuracy": acc, "mean_ponder_depth": mean_ponder, "total_samples": total}
