from pathlib import Path

from language_model.evaluation.comparison import compare_runs
from language_model.utils import write_json


def _write_run(directory: Path, *, manifest: str, tokenizer: str, perplexity: float, bpc: float) -> None:
    directory.mkdir()
    write_json(
        directory / "run.json",
        {
            "status": "completed",
            "manifest_identity": manifest,
            "tokenizer_fingerprint": tokenizer,
            "test": {
                "perplexity": perplexity,
                "bits_per_character": bpc,
                "bits_per_byte": bpc,
            },
        },
    )


def test_compare_runs_only_ranks_perplexity_for_matching_contracts(tmp_path: Path) -> None:
    char = tmp_path / "char"
    bpe = tmp_path / "bpe"
    _write_run(char, manifest="manifest-char", tokenizer="tokenizer-char", perplexity=3.0, bpc=1.5)
    _write_run(bpe, manifest="manifest-bpe", tokenizer="tokenizer-bpe", perplexity=2.0, bpc=1.2)

    result = compare_runs([char, bpe])

    assert result["ranking_by_test_perplexity"] == []
    assert result["ranking_by_test_bits_per_character"] == [str(bpe), str(char)]
    assert result["comparability"]["test_perplexity"]["comparable"] is False


def test_compare_runs_ranks_perplexity_for_matching_contracts(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(first, manifest="manifest", tokenizer="tokenizer", perplexity=3.0, bpc=1.5)
    _write_run(second, manifest="manifest", tokenizer="tokenizer", perplexity=2.0, bpc=1.6)

    result = compare_runs([first, second])

    assert result["ranking_by_test_perplexity"] == [str(second), str(first)]
    assert result["comparability"]["test_perplexity"]["comparable"] is True
