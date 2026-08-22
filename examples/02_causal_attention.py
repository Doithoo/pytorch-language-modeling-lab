"""Show that changing future tokens cannot change earlier Transformer logits."""

import torch

from language_model.config import ModelConfig
from language_model.models import TransformerLanguageModel

cfg = ModelConfig(vocab_size=16, context_length=8, n_layers=2, n_heads=2, d_model=32, d_ff=64, dropout=0.0)
model = TransformerLanguageModel(cfg, pad_id=0).eval()
left = torch.tensor([[1, 4, 5, 6]])
right = left.clone()
right[0, 3] = 9
with torch.inference_mode():
    left_logits = model(left).logits
    right_logits = model(right).logits
print("past logits unchanged:", torch.allclose(left_logits[:, :3], right_logits[:, :3]))
