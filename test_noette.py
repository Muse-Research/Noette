import unittest
import torch
from noette.config import NoetteConfig
from noette.model import NoetteLM, BaselineTransformerLM
from noette.loss import NoettePonderLoss
from noette.data import SimpleCharTokenizer, MultiHopDeductionDataset, AlgorithmicCarryDataset, collate_reasoning_batch

class TestNoette(unittest.TestCase):
    def setUp(self):
        self.config = NoetteConfig(
            vocab_size=64,
            d_model=64,
            n_heads=2,
            d_ff=128,
            max_seq_len=64,
            max_ponder_steps=4,
            ponder_lambda=0.3,
            ponder_beta=0.01
        )
        self.model = NoetteLM(self.config)
        self.loss_fn = NoettePonderLoss(lambda_prior=self.config.ponder_lambda, beta_reg=self.config.ponder_beta)

    def test_forward_shapes(self):
        B, S = 4, 16
        input_ids = torch.randint(0, self.config.vocab_size, (B, S))
        out = self.model(input_ids)
        
        self.assertEqual(out["logits"].shape, (B, S, self.config.vocab_size))
        self.assertEqual(out["step_logits"].shape, (self.config.max_ponder_steps, B, S, self.config.vocab_size))
        self.assertEqual(out["step_q"].shape, (self.config.max_ponder_steps, B, S, 1))
        self.assertEqual(out["expected_steps"].shape, (B, S))

    def test_halting_distribution_sums_to_one(self):
        B, S = 2, 8
        input_ids = torch.randint(0, self.config.vocab_size, (B, S))
        out = self.model(input_ids)
        q_sum = out["step_q"].sum(dim=0) # [B, S, 1]
        # Should sum to 1.0 (within numerical float precision)
        self.assertTrue(torch.allclose(q_sum, torch.ones_like(q_sum), atol=1e-4))

    def test_loss_and_gradients(self):
        B, S = 2, 12
        input_ids = torch.randint(0, self.config.vocab_size, (B, S))
        targets = torch.randint(0, self.config.vocab_size, (B, S))
        targets[:, :4] = -100 # Mask prompt
        
        out = self.model(input_ids)
        loss, metrics = self.loss_fn(
            step_logits=out["step_logits"],
            step_q=out["step_q"],
            targets=targets
        )
        
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)
        
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")

    def test_dataset_generation(self):
        dataset = MultiHopDeductionDataset(num_samples=10, min_hops=2, max_hops=4)
        self.assertEqual(len(dataset), 10)
        sample = dataset[0]
        self.assertIn("prompt", sample)
        self.assertIn("target_ans", sample)
        self.assertIn("full_ids", sample)

if __name__ == "__main__":
    unittest.main()
