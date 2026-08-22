import torch

from language_model.generation import sample_next_token


def test_greedy_sampling() -> None:
    logits = torch.tensor([[0.0, 1.0, 3.0, 2.0]])
    assert sample_next_token(logits, temperature=0.0, top_k=None, top_p=1.0).item() == 2


def test_top_k_never_samples_removed_tokens() -> None:
    logits = torch.tensor([[0.0, 1.0, 9.0, 8.0]])
    generator = torch.Generator().manual_seed(4)
    values = {
        sample_next_token(logits, temperature=1.0, top_k=2, top_p=1.0, generator=generator).item() for _ in range(30)
    }
    assert values <= {2, 3}
