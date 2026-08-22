"""Typed and validated experiment configuration."""

from __future__ import annotations

import dataclasses
import math
import types
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class DataConfig:
    corpus_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    manifest_dir: Path = Path("data/manifests")
    dataset_name: str = "corpus"
    tokenizer_name: str = "char"
    context_length: int = 128
    stride: int = 128
    split_ratios: list[float] = field(default_factory=lambda: [0.9, 0.05, 0.05])
    seed: int = 42
    num_workers: int = 0
    window_mode: str = "document"
    split_strategy: str = "document_count"
    group_by_source: bool = False


@dataclass(frozen=True)
class TokenizerConfig:
    name: str = "char"
    vocab_size: int | None = None
    min_frequency: int = 1
    special_tokens: list[str] = field(default_factory=lambda: ["<pad>", "<bos>", "<eos>", "<unk>"])


@dataclass(frozen=True)
class ModelConfig:
    name: str = "transformer_decoder"
    vocab_size: int | None = None
    context_length: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024
    dropout: float = 0.1
    positional_encoding: str = "learned"
    tie_embeddings: bool = True
    attention_implementation: str = "readable"


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 5000
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    amp: bool = False
    eval_interval: int = 250
    eval_batches: int = 50
    save_interval: int = 250
    log_interval: int = 10
    best_metric: str = "val_loss"
    beta1: float = 0.9
    beta2: float = 0.95


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int | None = 40
    top_p: float = 0.95
    use_cache: bool = True
    seed: int = 42
    prompts: list[str] = field(default_factory=lambda: ["The "])


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    seed: int = 42
    output_dir: Path = Path("artifacts")
    run_name: str = "language-model"
    log_level: str = "info"
    deterministic: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


class ConfigError(ValueError):
    """Raised when configuration is unknown, inconsistent, or unsafe."""


T = TypeVar("T")
_PATH_FIELDS = {
    DataConfig: {"corpus_dir", "processed_dir", "manifest_dir"},
    RuntimeConfig: {"output_dir"},
}
_SECTIONS: dict[str, type[Any]] = {
    "data": DataConfig,
    "tokenizer": TokenizerConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "generation": GenerationConfig,
    "runtime": RuntimeConfig,
}


def _construct(cls: type[T], raw: Mapping[str, Any], section: str) -> T:
    valid = {item.name for item in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - valid
    if unknown:
        raise ConfigError(f"unknown config key: {section}.{sorted(unknown)[0]}")
    values = dict(raw)
    for name in _PATH_FIELDS.get(cls, set()):
        if name in values:
            if not isinstance(values[name], (str, Path)) or not str(values[name]):
                raise ConfigError(f"{section}.{name} must be a non-empty path")
            values[name] = Path(values[name])
    hints = get_type_hints(cls)
    for name, value in values.items():
        if not _matches_type(value, hints[name]):
            raise ConfigError(f"{section}.{name} has invalid type; expected {_type_name(hints[name])}")
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"invalid config section {section}: {exc}") from exc


def _matches_type(value: Any, expected: Any) -> bool:
    origin = get_origin(expected)
    if origin is types.UnionType:
        return any(_matches_type(value, option) for option in get_args(expected))
    if origin is list:
        arguments = get_args(expected)
        return isinstance(value, list) and (not arguments or all(_matches_type(item, arguments[0]) for item in value))
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is bool:
        return isinstance(value, bool)
    return isinstance(value, expected)


def _type_name(expected: Any) -> str:
    origin = get_origin(expected)
    if origin is types.UnionType:
        return " or ".join(_type_name(option) for option in get_args(expected))
    if origin is list:
        arguments = get_args(expected)
        return f"list[{_type_name(arguments[0])}]" if arguments else "list"
    return getattr(expected, "__name__", str(expected))


def config_from_dict(raw: Mapping[str, Any]) -> ExperimentConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError("config root must be a mapping")
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ConfigError(f"unknown config section: {sorted(unknown)[0]}")
    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        value = raw.get(name, {})
        if not isinstance(value, Mapping):
            raise ConfigError(f"config section {name} must be a mapping")
        sections[name] = _construct(cls, value, name)
    cfg = ExperimentConfig(**sections)
    validate_config(cfg)
    return cfg


def _deep_set(raw: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if len(parts) != 2 or parts[0] not in _SECTIONS:
        raise ConfigError(f"override must be section.field, got {dotted_key!r}")
    section = raw.setdefault(parts[0], {})
    if not isinstance(section, dict):
        raise ConfigError(f"cannot override non-mapping section {parts[0]}")
    section[parts[1]] = value


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError("config root must be a mapping")
    if overrides:
        for key, value in overrides.items():
            _deep_set(loaded, key, value)
    return config_from_dict(loaded)


def validate_config(cfg: ExperimentConfig) -> None:
    d, t, m, train, gen, runtime = (
        cfg.data,
        cfg.tokenizer,
        cfg.model,
        cfg.train,
        cfg.generation,
        cfg.runtime,
    )
    if not d.dataset_name.strip():
        raise ConfigError("data.dataset_name must not be empty")
    if d.context_length < 2:
        raise ConfigError("data.context_length must be at least 2")
    if not 1 <= d.stride <= d.context_length:
        raise ConfigError("data.stride must be between 1 and context_length")
    if len(d.split_ratios) != 3 or any(value <= 0 for value in d.split_ratios):
        raise ConfigError("data.split_ratios must contain three positive values")
    if not math.isclose(sum(d.split_ratios), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ConfigError("data.split_ratios must sum to 1")
    if d.num_workers < 0:
        raise ConfigError("data.num_workers must be non-negative")
    if d.window_mode != "document":
        raise ConfigError("only data.window_mode=document is supported")
    if d.split_strategy not in {"document_count", "character_balanced"}:
        raise ConfigError("data.split_strategy must be document_count or character_balanced")
    if t.name not in {"char", "bpe"}:
        raise ConfigError("tokenizer.name must be char or bpe")
    if t.name != d.tokenizer_name:
        raise ConfigError("data.tokenizer_name must match tokenizer.name")
    if len(t.special_tokens) != 4 or len(set(t.special_tokens)) != 4:
        raise ConfigError("tokenizer.special_tokens must contain four unique tokens")
    if t.min_frequency < 1:
        raise ConfigError("tokenizer.min_frequency must be positive")
    if t.name == "bpe" and (t.vocab_size is None or t.vocab_size < 32):
        raise ConfigError("BPE tokenizer.vocab_size must be at least 32")
    if m.name not in {"bigram", "transformer_decoder"}:
        raise ConfigError("model.name must be bigram or transformer_decoder")
    if m.context_length != d.context_length:
        raise ConfigError("model.context_length must equal data.context_length")
    if m.vocab_size is not None and m.vocab_size < len(t.special_tokens) + 1:
        raise ConfigError("model.vocab_size is too small")
    if m.n_layers < 1 or m.n_heads < 1 or m.d_model < 1 or m.d_ff < 1:
        raise ConfigError("model dimensions must be positive")
    if m.d_model % m.n_heads:
        raise ConfigError("model.d_model must be divisible by model.n_heads")
    if not 0 <= m.dropout < 1:
        raise ConfigError("model.dropout must be in [0, 1)")
    if m.positional_encoding not in {"learned", "rotary"}:
        raise ConfigError("model.positional_encoding must be learned or rotary")
    if m.positional_encoding == "rotary" and (m.d_model // m.n_heads) % 2:
        raise ConfigError("rotary positional encoding requires an even attention head dimension")
    if m.attention_implementation not in {"readable", "sdpa"}:
        raise ConfigError("model.attention_implementation must be readable or sdpa")
    for name in ("max_steps", "batch_size", "gradient_accumulation_steps", "eval_interval", "save_interval"):
        if getattr(train, name) < 1:
            raise ConfigError(f"train.{name} must be positive")
    if train.eval_batches < 1 or train.log_interval < 1:
        raise ConfigError("train.eval_batches and train.log_interval must be positive")
    if train.learning_rate <= 0 or train.weight_decay < 0 or train.max_grad_norm < 0:
        raise ConfigError("invalid optimizer settings")
    if not 0 <= train.warmup_steps <= train.max_steps:
        raise ConfigError("train.warmup_steps must be between 0 and max_steps")
    if train.best_metric != "val_loss":
        raise ConfigError("only train.best_metric=val_loss is supported")
    if gen.max_new_tokens < 1 or gen.temperature < 0:
        raise ConfigError("invalid generation length or temperature")
    if gen.top_k is not None and gen.top_k < 1:
        raise ConfigError("generation.top_k must be positive or null")
    if not 0 < gen.top_p <= 1:
        raise ConfigError("generation.top_p must be in (0, 1]")
    if not gen.prompts or any(not prompt for prompt in gen.prompts):
        raise ConfigError("generation.prompts must contain non-empty strings")
    if runtime.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ConfigError("runtime.device must be auto, cpu, cuda, or mps")
    if not runtime.run_name.strip():
        raise ConfigError("runtime.run_name must not be empty")


def resolve_vocab_size(cfg: ExperimentConfig, vocab_size: int) -> ExperimentConfig:
    if vocab_size < 1:
        raise ConfigError("resolved vocabulary must be positive")
    if cfg.model.vocab_size is not None and cfg.model.vocab_size != vocab_size:
        raise ConfigError(
            f"configured model.vocab_size={cfg.model.vocab_size} does not match tokenizer vocab_size={vocab_size}"
        )
    if cfg.tokenizer.vocab_size is not None and cfg.tokenizer.name == "char" and cfg.tokenizer.vocab_size != vocab_size:
        raise ConfigError(
            f"configured tokenizer.vocab_size={cfg.tokenizer.vocab_size} does not match learned size={vocab_size}"
        )
    resolved = replace(
        cfg,
        tokenizer=replace(cfg.tokenizer, vocab_size=vocab_size),
        model=replace(cfg.model, vocab_size=vocab_size),
    )
    validate_config(resolved)
    return resolved


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if dataclasses.is_dataclass(value):
            return {item.name: convert(getattr(value, item.name)) for item in fields(value)}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    return convert(cfg)


def parse_overrides(values: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ConfigError(f"override must be KEY=VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        result[key] = yaml.safe_load(raw)
    return result
