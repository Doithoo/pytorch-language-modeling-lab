import torch

from language_model.config import ModelConfig
from language_model.models import BigramLanguageModel, TransformerLanguageModel


def model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=24,
        context_length=8,
        n_layers=2,
        n_heads=2,
        d_model=16,
        d_ff=32,
        dropout=0.0,
    )


def test_bigram_shapes_and_loss() -> None:
    model = BigramLanguageModel(vocab_size=12, pad_id=0)
    inputs = torch.tensor([[1, 4, 5]])
    output = model(inputs, torch.tensor([[4, 5, 2]]))
    assert output.logits.shape == (1, 3, 12)
    assert output.loss is not None
    assert output.token_count == 3


def test_transformer_is_causal() -> None:
    torch.manual_seed(0)
    model = TransformerLanguageModel(model_config(), pad_id=0).eval()
    left = torch.tensor([[1, 5, 6, 7, 8]])
    right = left.clone()
    right[0, 4] = 9
    with torch.inference_mode():
        left_logits = model(left).logits
        right_logits = model(right).logits
    assert torch.allclose(left_logits[:, :4], right_logits[:, :4], atol=1e-6)


def test_kv_cache_matches_full_forward() -> None:
    torch.manual_seed(0)
    model = TransformerLanguageModel(model_config(), pad_id=0).eval()
    ids = torch.tensor([[1, 5, 6, 7]])
    with torch.inference_mode():
        full = model(ids).logits[:, -1]
        prefix = model(ids[:, :3], use_cache=True)
        cached = model(ids[:, 3:], past_key_values=prefix.past_key_values, use_cache=True).logits[:, -1]
    assert torch.allclose(full, cached, atol=1e-5)
