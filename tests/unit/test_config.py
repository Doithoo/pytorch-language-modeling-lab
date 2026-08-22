import pytest

from language_model.config import ConfigError, config_from_dict, config_to_dict, parse_overrides


def test_config_rejects_unknown_key() -> None:
    with pytest.raises(ConfigError, match="unknown config key"):
        config_from_dict({"model": {"hidden_size": 32}})


def test_config_rejects_inconsistent_attention_dimensions() -> None:
    with pytest.raises(ConfigError, match="divisible"):
        config_from_dict({"model": {"d_model": 30, "n_heads": 8}})


def test_config_roundtrip_and_overrides() -> None:
    cfg = config_from_dict({"train": {"max_steps": 10, "warmup_steps": 0}, "runtime": {"device": "cpu"}})
    assert config_from_dict(config_to_dict(cfg)) == cfg
    assert parse_overrides(["train.max_steps=2", "train.amp=true"]) == {
        "train.max_steps": 2,
        "train.amp": True,
    }
