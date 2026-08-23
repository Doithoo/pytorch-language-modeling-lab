from dataclasses import replace

import torch

from language_model.config import ModelConfig
from language_model.models import BigramLanguageModel, TransformerLanguageModel
from language_model.models.attention import CausalSelfAttention
from language_model.models.transformer import RotaryEmbedding


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


def test_sdpa_attention_matches_readable_attention() -> None:
    readable = model_config()
    sdpa = replace(readable, attention_implementation="sdpa")
    torch.manual_seed(11)
    readable_model = TransformerLanguageModel(readable, pad_id=0).eval()
    torch.manual_seed(11)
    sdpa_model = TransformerLanguageModel(sdpa, pad_id=0).eval()
    sdpa_model.load_state_dict(readable_model.state_dict())
    inputs = torch.tensor([[1, 5, 6, 7]])
    with torch.inference_mode():
        expected = readable_model(inputs).logits
        actual = sdpa_model(inputs).logits
    assert torch.allclose(expected, actual, atol=1e-5)


def test_rotary_cache_matches_dynamic_formula_without_persistent_state() -> None:
    attention = CausalSelfAttention(16, 2, 0.0, rotary=True)
    rotary_embedding = RotaryEmbedding(8, 8)
    tensor = torch.randn(1, 2, 4, 8)
    positions = torch.arange(4)
    actual = attention._apply_rotary(tensor, rotary_embedding(positions, tensor.dtype))

    half = tensor.shape[-1] // 2
    frequencies = torch.arange(half, dtype=torch.float32)
    inverse = 1.0 / (10000.0 ** (frequencies / half))
    angles = positions.to(torch.float32).unsqueeze(-1) * inverse.unsqueeze(0)
    cos = angles.cos().view(1, 1, -1, half)
    sin = angles.sin().view(1, 1, -1, half)
    even = tensor[..., ::2]
    odd = tensor[..., 1::2]
    expected = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    assert torch.allclose(actual, expected, atol=1e-7)
    assert not rotary_embedding.state_dict()


def test_rotary_cache_uses_autocast_dtype() -> None:
    rotary = replace(model_config(), positional_encoding="rotary", attention_implementation="sdpa")
    model = TransformerLanguageModel(rotary, pad_id=0).eval()
    assert model.rotary_embedding is not None
    captured: list[torch.dtype] = []

    def record_dtype(_module, _inputs, output) -> None:
        captured.append(output[0].dtype)

    handle = model.rotary_embedding.register_forward_hook(record_dtype)
    try:
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            model(torch.tensor([[1, 5, 6, 7]]))
    finally:
        handle.remove()
    assert captured == [torch.bfloat16]


def test_rotary_cache_matches_full_forward() -> None:
    rotary = replace(model_config(), positional_encoding="rotary")
    torch.manual_seed(23)
    model = TransformerLanguageModel(rotary, pad_id=0).eval()
    ids = torch.tensor([[1, 5, 6, 7]])
    with torch.inference_mode():
        full = model(ids).logits[:, -1]
        prefix = model(ids[:, :3], use_cache=True)
        cached = model(ids[:, 3:], past_key_values=prefix.past_key_values, use_cache=True).logits[:, -1]
    assert torch.allclose(full, cached, atol=1e-5)


def test_rotary_is_causal() -> None:
    rotary = replace(model_config(), positional_encoding="rotary")
    torch.manual_seed(29)
    model = TransformerLanguageModel(rotary, pad_id=0).eval()
    left = torch.tensor([[1, 5, 6, 7, 8]])
    right = left.clone()
    right[0, 4] = 9
    with torch.inference_mode():
        left_logits = model(left).logits
        right_logits = model(right).logits
    assert torch.allclose(left_logits[:, :4], right_logits[:, :4], atol=1e-6)
