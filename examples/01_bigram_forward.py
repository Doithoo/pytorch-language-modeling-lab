"""Inspect the bigram model's input, logits, and next-token loss."""

import torch

from language_model.models import BigramLanguageModel

model = BigramLanguageModel(vocab_size=8, pad_id=0)
input_ids = torch.tensor([[1, 4, 5, 6]])
targets = torch.tensor([[4, 5, 6, 2]])
output = model(input_ids, targets)
print("input:", tuple(input_ids.shape))
print("logits:", tuple(output.logits.shape))
assert output.loss is not None
print("loss:", output.loss.item())
