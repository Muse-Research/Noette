import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

class SimpleCharTokenizer:
    """
    Compact character-level tokenizer for symbolic reasoning tasks.
    """
    def __init__(self):
        chars = (
            ["<pad>", "<eos>", "<bos>", "<unk>", "=", "+", "-", ">", "?", ",", ";", "[", "]", "(", ")", " "]
            + [str(i) for i in range(10)]
            + [chr(c) for c in range(ord('A'), ord('Z') + 1)]
            + [chr(c) for c in range(ord('a'), ord('z') + 1)]
        )
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        self.pad_token_id = self.char_to_id["<pad>"]
        self.eos_token_id = self.char_to_id["<eos>"]
        self.bos_token_id = self.char_to_id["<bos>"]
        self.vocab_size = len(chars)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(ch, self.char_to_id["<unk>"]) for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join([self.id_to_char.get(i, "") for i in ids if i not in (self.pad_token_id, self.eos_token_id, self.bos_token_id)])


class MultiHopDeductionDataset(Dataset):
    """
    Multi-Hop Symbolic Pointer Dereferencing.
    Example: "[A=8, B=A, C=B, D=C] Q: D? Ans: 8"
    Requires D hops of internal pointer chasing.
    """
    def __init__(self, num_samples: int = 5000, min_hops: int = 2, max_hops: int = 5, seed: int = 42):
        super().__init__()
        self.tokenizer = SimpleCharTokenizer()
        self.samples = []
        random.seed(seed)
        
        letters = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
        
        for _ in range(num_samples):
            hops = random.randint(min_hops, max_hops)
            chosen_vars = random.sample(letters, hops + 1)
            base_val = random.randint(0, 9)
            
            # Construct chain: var0 = base_val, var1 = var0, var2 = var1 ...
            clauses = [f"{chosen_vars[0]}={base_val}"]
            for i in range(1, len(chosen_vars)):
                clauses.append(f"{chosen_vars[i]}={chosen_vars[i-1]}")
            
            random.shuffle(clauses) # Shuffle context to prevent trivial positional bias
            
            context = "[" + ", ".join(clauses) + "]"
            target_var = chosen_vars[-1]
            prompt = f"{context} Q:{target_var}?="
            target_ans = f"{base_val}"
            
            full_text = f"{prompt}{target_ans}"
            prompt_ids = self.tokenizer.encode(prompt)
            full_ids = self.tokenizer.encode(full_text)
            
            self.samples.append({
                "prompt": prompt,
                "target_ans": target_ans,
                "full_text": full_text,
                "prompt_len": len(prompt_ids),
                "full_ids": full_ids,
                "hops": hops
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class AlgorithmicCarryDataset(Dataset):
    """
    Multi-digit addition with complex carry chains.
    Form: "ADD: 489 + 527 = 1016"
    Forces the network to compute carry ripples across digits.
    """
    def __init__(self, num_samples: int = 5000, num_digits: int = 4, seed: int = 42):
        super().__init__()
        self.tokenizer = SimpleCharTokenizer()
        self.samples = []
        random.seed(seed)
        
        for _ in range(num_samples):
            # Create numbers that deliberately trigger long carry chains (e.g. 999 + 1)
            if random.random() < 0.4:
                a = random.randint(10 ** (num_digits - 1), 10 ** num_digits - 1)
                b = 10 ** num_digits - a + random.randint(0, 5)
            else:
                a = random.randint(10 ** (num_digits - 1), 10 ** num_digits - 1)
                b = random.randint(10 ** (num_digits - 1), 10 ** num_digits - 1)
                
            ans = a + b
            prompt = f"ADD: {a}+{b}="
            target_ans = f"{ans}"
            full_text = f"{prompt}{target_ans}"
            
            prompt_ids = self.tokenizer.encode(prompt)
            full_ids = self.tokenizer.encode(full_text)
            
            self.samples.append({
                "prompt": prompt,
                "target_ans": target_ans,
                "full_text": full_text,
                "prompt_len": len(prompt_ids),
                "full_ids": full_ids
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_reasoning_batch(batch: List[Dict], pad_token_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Collates samples, pads sequences, and masks prompt tokens with -100 so loss is only computed on target answers.
    """
    max_len = max(len(s["full_ids"]) for s in batch)
    B = len(batch)
    
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    targets = torch.full((B, max_len), -100, dtype=torch.long)
    prompt_lens = [s["prompt_len"] for s in batch]
    
    for i, s in enumerate(batch):
        ids = torch.tensor(s["full_ids"], dtype=torch.long)
        L = len(ids)
        input_ids[i, :L] = ids
        
        # Targets are shifted by 1 for causal autoregressive prediction
        if L > 1:
            targets[i, :L-1] = ids[1:]
            # Mask out prompt portion so we only score answering tokens
            p_len = s["prompt_len"]
            targets[i, :p_len - 1] = -100
            
    return input_ids, targets, prompt_lens
