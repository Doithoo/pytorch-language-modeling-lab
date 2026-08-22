from pathlib import Path

from language_model.config import TokenizerConfig
from language_model.tokenization import BPETokenizer, CharTokenizer, load_tokenizer, train_tokenizer


def test_character_tokenizer_roundtrip_and_unknown(tmp_path: Path) -> None:
    tokenizer = CharTokenizer.train(["abc cab"], ["<pad>", "<bos>", "<eos>", "<unk>"])
    ids = tokenizer.encode("abc", add_bos=True, add_eos=True)
    assert tokenizer.decode(ids) == "abc"
    assert tokenizer.encode("z") == [tokenizer.unk_id]
    tokenizer.save(tmp_path)
    loaded = load_tokenizer(tmp_path)
    assert loaded.to_payload() == tokenizer.to_payload()
    assert loaded.fingerprint() == tokenizer.fingerprint()


def test_bpe_tokenizer_roundtrip_and_payload(tmp_path: Path) -> None:
    cfg = TokenizerConfig(name="bpe", vocab_size=64, min_frequency=1)
    tokenizer = train_tokenizer(["hello world", "hello language model", "world model"], cfg)
    assert isinstance(tokenizer, BPETokenizer)
    ids = tokenizer.encode("hello world", add_bos=True, add_eos=True)
    assert tokenizer.decode(ids) == "hello world"
    tokenizer.save(tmp_path)
    loaded = load_tokenizer(tmp_path)
    assert loaded.fingerprint() == tokenizer.fingerprint()
