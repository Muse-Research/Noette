"""
Hugging Face Space Gradio Web Demo for Noette.
Visualizes real-time token-by-token latent ponder depth heatmaps in the browser.
"""

import torch
from noette.config import NoetteConfig
from noette.model import NoetteLM
from noette.loss import NoettePonderLoss
from noette.data import SimpleCharTokenizer, MultiHopDeductionDataset, collate_reasoning_batch
from noette.trainer import train_noette_epoch
from torch.utils.data import DataLoader

def init_demo_model():
    dataset = MultiHopDeductionDataset(num_samples=1500, min_hops=2, max_hops=4, seed=42)
    tokenizer = dataset.tokenizer
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=lambda b: collate_reasoning_batch(b, pad_token_id=tokenizer.pad_token_id)
    )
    
    config = NoetteConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=96,
        n_heads=3,
        d_ff=384,
        max_ponder_steps=6,
        ponder_lambda=0.25,
        ponder_beta=0.015
    )
    model = NoetteLM(config)
    loss_fn = NoettePonderLoss(lambda_prior=config.ponder_lambda, beta_reg=config.ponder_beta)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    
    # Fast 5-epoch training
    for _ in range(5):
        train_noette_epoch(model, loader, opt, loss_fn, torch.device("cpu"))
        
    return model, tokenizer

def analyze_prompt(prompt_text: str, model, tokenizer):
    model.eval()
    encoded = tokenizer.encode(prompt_text)
    input_ids = torch.tensor([encoded], dtype=torch.long)
    
    with torch.no_grad():
        out = model(input_ids, return_ponder_details=True)
        logits = out["logits"]
        expected_steps = out["expected_steps"][0]
        step_q = out["step_q"][:, 0, :, 0]
        
    pred_id = torch.argmax(logits[0, -1]).item()
    pred_char = tokenizer.decode([pred_id])
    
    html_output = ["<div style='font-family: monospace; font-size: 14px;'>"]
    html_output.append(f"<h3>Predicted Deduction: <span style='color: #10b981;'>{pred_char}</span></h3>")
    html_output.append("<table style='width:100%; border-collapse: collapse;'>")
    html_output.append("<tr style='background: #1f2937; color: #fff;'><th style='padding: 6px;'>Pos</th><th>Token</th><th>Latent Depth</th><th>Ponder Heatmap</th><th>Distribution (q1..qK)</th></tr>")
    
    chars = [tokenizer.id_to_char.get(i, "?") for i in encoded]
    for i, ch in enumerate(chars):
        depth = expected_steps[i].item()
        # Heatmap color from blue (low compute) to orange/red (deep ponder)
        intensity = min(1.0, max(0.0, (depth - 1.0) / 4.0))
        r = int(59 + intensity * (239 - 59))
        g = int(130 - intensity * 60)
        b = int(246 - intensity * 180)
        color = f"rgb({r},{g},{b})"
        
        display_ch = ch if ch != " " else "&nbsp;"
        q_dist = " ".join([f"{step_q[t, i].item():.2f}" for t in range(step_q.shape[0])])
        
        html_output.append(f"<tr style='border-bottom: 1px solid #374151;'>")
        html_output.append(f"<td style='padding: 6px; text-align: center;'>{i}</td>")
        html_output.append(f"<td style='font-weight: bold; text-align: center;'>{display_ch}</td>")
        html_output.append(f"<td style='text-align: center; color: {color}; font-weight: bold;'>{depth:.2f}x</td>")
        html_output.append(f"<td><div style='background: {color}; width: {int(depth * 15)}%; height: 12px; border-radius: 3px;'></div></td>")
        html_output.append(f"<td style='font-size: 12px; color: #9ca3af;'>{q_dist}</td>")
        html_output.append("</tr>")
        
    html_output.append("</table></div>")
    return "".join(html_output)

if __name__ == "__main__":
    print("Noette Hugging Face Space app entrypoint initialized.")
